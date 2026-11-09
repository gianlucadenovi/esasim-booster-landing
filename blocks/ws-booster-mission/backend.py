# ── EsaSim block ──────────────────────────────────────────────────────
# Booster mission (ws-booster-mission) · version 5 · machine block, Python: def run(ctx)
# Launch and land a Falcon 9-class booster live: the precomputed launch, flip and entry burn, then the learned policy lands it on the drone ship.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-09
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policies · The booster_policy*.pt files from Booster landing: train (wire its dataset: every training adds one); the newest that has finished its lessons is flown unless Policy file names one
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   flights     Table (csv)               Flights · flights.csv: each flight's wind, sea, result, distance from the centre, touchdown speed and lean
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   wind        number    Wind (m/s) = 6
#   jet         number    Jet stream (m/s) = 25
#   sea         number    Sea (0-1) = 0.4
#   warp        number    Time warp = 1
#   offset      number    Ship offset (m) = 0
#   policy_file text      Policy file = ""
#   flights_per_run number    Flights per run = 1
#   auto_launch bool      Launch at once = false
#   channel     text      Flight channel = "rocket"
#   events_channel text      Events channel = "event"
#   control_channel text      Controls channel = "control"
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch[cuda] (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Booster mission. Launch and landing of a Falcon 9-class first stage, live in the 3D view: the launch is computed
beforehand (nine engines, max-Q throttle-down, gravity turn, cut-off and separation); then the booster flips with its
thrusters, coasts up to about 120 km, lights three engines for the entry burn and hands over to the landing policy
learned by Booster landing: train, which flies the grid fins and the landing burn to the drone ship. Launch starts a
flight (Wind, Sea and Time warp take effect at the next one); the run waits for the next Launch until it is stopped."""
import math

try:
    import torch
except ImportError:  # the block says how to install it
    torch = None

# ── where to learn ─────────────────────────────────────────────────────────────────────────────────────────
def pick_device(want):
    """(device, label, kind): an NVIDIA GPU or the CPU. auto takes the GPU when this PyTorch can use one."""
    if torch is None:
        raise RuntimeError("This block needs PyTorch on the worker. With an NVIDIA GPU: pip install torch --index-url "
                           "https://download.pytorch.org/whl/cu124 (CPU only: pip install torch).")
    w = str(want or "auto").strip().lower()
    if w.startswith("nvidia") or w == "auto":
        if torch.cuda.is_available():
            n = torch.cuda.get_device_name(0)
            return torch.device("cuda"), (n if n.upper().startswith("NVIDIA") else "NVIDIA " + n), "nvidia"
        if w.startswith("nvidia"):
            raise RuntimeError("NVIDIA GPU asked, but this PyTorch (%s) cannot use one: install the CUDA build of PyTorch "
                               "(pip install torch --index-url https://download.pytorch.org/whl/cu124)." % torch.__version__)
    if w.startswith("tenstorrent"):
        raise RuntimeError("Tenstorrent: this block does not run on the Tenstorrent card yet. Choose auto, NVIDIA GPU or CPU.")
    import shutil
    note = "" if not shutil.which("nvidia-smi") else (" - this machine has an NVIDIA GPU but this PyTorch (%s) cannot use it: install the CUDA build "
                                                      "(pip install torch --index-url https://download.pytorch.org/whl/cu124)" % torch.__version__)
    return torch.device("cpu"), "CPU (%d threads)%s" % (torch.get_num_threads(), note), "cpu"


# ── booster physics (shared by "Booster landing: train" and "Booster mission") ─────────────────────────────
# A Falcon 9-class first stage, from public figures (approximate): 3.66 m wide, about 45 m long with the
# interstage, 25.6 t empty, nine Merlin-class engines (845 kN each at sea level, 914 kN in vacuum, Isp 282 / 311 s),
# one or three of them relit for the entry and landing burns, throttle down to 40 % (one engine at its lowest still
# lifts more than the empty booster weighs: it cannot hover, so the landing burn must end exactly at the deck),
# gimbal +-5 deg, four grid fins at the top, nitrogen thrusters for attitude in vacuum, four landing legs.
# World frame: metres, x east (downrange), y north, z up; the launch pad at the origin. Gravity falls with height;
# the ground is flat for the physics (the view curves it). Roll is held by the thrusters, so the body is described by
# its axis b (a unit vector from the engines to the top) and an angular velocity across it.


G0 = 9.80665
R_EARTH = 6371000.0
LEN = 45.0                 # engine exit to the top of the interstage
AREA = math.pi * 1.83 ** 2
DRY = 25600.0
T_SL, T_VAC = 845e3, 914e3  # one engine
ISP_VAC = 311.0
MDOT1 = T_VAC / (ISP_VAC * G0)  # kg/s of one engine at full throttle
THR_MIN = 0.40
GIMBAL = math.radians(5.0)
Z_ENG, Z_CP, Z_FIN = 1.5, 21.0, 42.0   # gimbal point, body centre of pressure, grid fins (from the engine exit)
S_FIN, CN_FIN, CL_FIN, FIN_MAX = 3.6, 2.0, 2.2, math.radians(20.0)
CN_BODY = 1.6
TAU_RCS = 150e3             # N m, the cold-gas thrusters at the top
LEG_R, LEG_DROP = 9.0, 1.2   # leg feet: radius from the axis, below the engine exit
DECK_X, DECK_Y, DECK_Z = 45.5, 26.0, 2.5   # half length, half width, height above the sea
WN, ZETA = 1.6, 0.85        # attitude loop
P0 = 101325.0


def atmosphere(h):
    """density kg/m3, pressure Pa, speed of sound m/s at height h (tensor), a smooth US-76-like profile."""
    h = h.clamp(min=0.0)
    T = torch.where(h < 11000.0, 288.15 - 0.0065 * h,
                    torch.where(h < 20000.0, torch.full_like(h, 216.65),
                                torch.where(h < 32000.0, 216.65 + 0.001 * (h - 20000.0), 228.65 + 0.0028 * (h - 32000.0).clamp(max=15000.0))))
    rho = torch.where(h < 11000.0, 1.225 * (T / 288.15).clamp(min=0.5) ** 4.2559,
                      torch.where(h < 20000.0, 0.3639 * torch.exp(-(h - 11000.0) / 6341.6),
                                  torch.where(h < 40000.0, 0.08803 * torch.exp(-(h - 20000.0) / 6500.0),
                                              0.003948 * torch.exp(-(h - 40000.0) / 7500.0))))
    p = rho * 287.05 * T
    return rho, p, torch.sqrt(1.4 * 287.05 * T)


def cg_of(m):
    """height of the centre of mass above the engine exit: the engines are heavy, the propellant left sits low."""
    prop = (m - DRY).clamp(min=0.0)
    return (DRY * 15.0 + prop * 11.0) / m


def inertia(m):
    return m * 10.5 ** 2 + 6.0e5


def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def dot(a, b):
    return (a * b).sum(-1, keepdim=True)


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def perp(v, b):
    return v - dot(v, b) * b


class Wind:
    """Per run: a surface wind and its direction, growing with height (power law), a jet stream near 11 km, and gusts
    (a random process with a few seconds of memory, stronger near the sea)."""

    def __init__(self, n, dev, dtype, gen=None, surface=(0.0, 12.0), jet=(0.0, 35.0)):
        self.n, self.dev, self.dt = n, dev, dtype
        self.gen = gen
        self.u0 = torch.zeros(n, device=dev, dtype=dtype)
        self.d0 = torch.zeros(n, device=dev, dtype=dtype)
        self.j = torch.zeros(n, device=dev, dtype=dtype)
        self.dj = torch.zeros(n, device=dev, dtype=dtype)
        self.g = torch.zeros(n, 2, device=dev, dtype=dtype)
        self.surface, self.jet = surface, jet

    def rnd(self, *shape):
        return torch.rand(*shape, device=self.dev, dtype=self.dt, generator=self.gen)

    def reset(self, idx, u0=None, d0=None):
        k = idx.numel()
        if k == 0:
            return
        lo, hi = self.surface
        self.u0[idx] = lo + (hi - lo) * self.rnd(k) if u0 is None else u0
        self.d0[idx] = 2 * math.pi * self.rnd(k) if d0 is None else d0
        self.j[idx] = self.jet[0] + (self.jet[1] - self.jet[0]) * self.rnd(k)
        self.dj[idx] = self.d0[idx] + (self.rnd(k) - 0.5) * 1.2
        self.g[idx] = 0.0

    def step(self, dt):
        tau = 4.0
        s = 0.18 * self.u0 + 0.4
        n = torch.randn(self.n, 2, device=self.dev, dtype=self.dt, generator=self.gen)
        self.g = self.g + (-self.g / tau) * dt + (s * math.sqrt(2.0 * dt / tau)).unsqueeze(-1) * n

    def at(self, h):
        h = h.clamp(min=1.0)
        prof = (h / 10.0) ** 0.14
        low = self.u0 * prof.clamp(max=2.4) * torch.exp(-((h - 9000.0).clamp(min=0.0) / 12000.0) ** 2)
        jet = self.j * torch.exp(-((h - 11500.0) / 4500.0) ** 2)
        gust = torch.exp(-h / 3000.0).unsqueeze(-1) * self.g
        w = torch.stack([low * torch.cos(self.d0) + jet * torch.cos(self.dj), low * torch.sin(self.d0) + jet * torch.sin(self.dj)], -1) + gust
        return torch.cat([w, torch.zeros_like(w[:, :1])], -1)


class Deck:
    """The drone ship's deck: it holds its station but heaves, pitches and rolls with the swell."""

    def __init__(self, n, dev, dtype, gen=None):
        z = lambda: torch.zeros(n, device=dev, dtype=dtype)
        self.n, self.dev, self.dt, self.gen = n, dev, dtype, gen
        self.c = torch.zeros(n, 3, device=dev, dtype=dtype)   # centre of the deck at rest
        self.amp_h, self.amp_p, self.amp_r = z(), z(), z()
        self.w_h, self.w_p, self.w_r = z(), z(), z()
        self.ph = torch.zeros(n, 3, device=dev, dtype=dtype)

    def reset(self, idx, centre, sea=None):
        k = idx.numel()
        if k == 0:
            return
        r = lambda: torch.rand(k, device=self.dev, dtype=self.dt, generator=self.gen)
        s = r() if sea is None else torch.full((k,), float(sea), device=self.dev, dtype=self.dt)
        self.c[idx] = centre
        # a 90 m barge: about a metre of heave and a couple of degrees of roll in a moderate sea, twice that in a rough one
        self.amp_h[idx] = 0.2 + 1.8 * s
        self.amp_p[idx] = math.radians(0.3) + math.radians(2.4) * s * (0.5 + 0.5 * r())
        self.amp_r[idx] = math.radians(0.4) + math.radians(3.6) * s * (0.5 + 0.5 * r())
        self.w_h[idx] = 2 * math.pi / (7.0 + 5.0 * r())
        self.w_p[idx] = 2 * math.pi / (6.0 + 5.0 * r())
        self.w_r[idx] = 2 * math.pi / (8.0 + 6.0 * r())
        self.ph[idx] = 2 * math.pi * torch.rand(k, 3, device=self.dev, dtype=self.dt, generator=self.gen)

    def state(self, t):
        """heave (m), pitch, roll (rad) and the heave rate at time t"""
        a = self.w_h * t + self.ph[:, 0]
        heave = self.amp_h * torch.sin(a)
        pitch = self.amp_p * torch.sin(self.w_p * t + self.ph[:, 1])
        roll = self.amp_r * torch.sin(self.w_r * t + self.ph[:, 2])
        return heave, pitch, roll, self.amp_h * self.w_h * torch.cos(a)

    def height(self, t, xy):
        heave, pitch, roll, _ = self.state(t)
        dx, dy = xy[:, 0] - self.c[:, 0], xy[:, 1] - self.c[:, 1]
        return self.c[:, 2] + DECK_Z + heave - torch.tan(pitch) * dx + torch.tan(roll) * dy


class Booster:
    """N boosters stepped together. Commands: throttle (0.4..1), ignition (once, then the engine runs until touchdown
    or the propellant ends), engines (1 for landing, 3 for the entry burn) and the attitude wanted (an axis b_des);
    the flight computer turns the attitude into gimbal, grid fins and thrusters, each within its limits."""

    def __init__(self, n, dev, dtype=None, gen=None):
        dtype = dtype or torch.float32
        self.n, self.dev, self.dt, self.gen = n, dev, dtype, gen
        v3 = lambda: torch.zeros(n, 3, device=dev, dtype=dtype)
        s = lambda v=0.0: torch.full((n,), v, device=dev, dtype=dtype)
        self.pos, self.vel, self.w = v3(), v3(), v3()
        self.b = v3(); self.b[:, 2] = 1.0
        self.m, self.t = s(DRY + 8000.0), s()
        self.lit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.engines = s(1.0)
        self.ign = s()               # seconds since ignition
        self.thr, self.thr_cmd = s(), s(THR_MIN)
        self.u_g, self.u_f = v3(), v3()   # actuator states (fraction of the torque each can give, as a vector)
        self.fins, self.legs = s(1.0), s()   # deployed 0..1
        self.legs_cmd = torch.zeros(n, dtype=torch.bool, device=dev)
        self.b_des = self.b.clone()
        self.hold = s(1.0)            # 1: the flight computer holds b_des; 0: free (thrusters off)
        self.wind = Wind(n, dev, dtype, gen)
        self.deck = Deck(n, dev, dtype, gen)
        # what the view and the reward need
        self.q_dyn, self.mach, self.p_amb = s(), s(), s(P0)
        self.thrust, self.heat = s(), s()
        self.gim, self.finv, self.rcs = v3(), v3(), v3()

    # --------------------------------------------------------------- the step
    STATE = ("pos", "vel", "w", "b", "m", "t", "ign", "thr", "u_g", "u_f", "legs", "lit")

    def step(self, dt, mask=None):
        """dt seconds of flight; with a mask, only those boosters move (the others keep their state)"""
        if mask is not None:
            old = {k: getattr(self, k) for k in self.STATE}
            oldg = self.wind.g
        self._step(dt)
        if mask is not None:
            for k, v in old.items():
                nv = getattr(self, k)
                setattr(self, k, torch.where(mask.view(-1, *([1] * (nv.dim() - 1))), nv, v))
            self.wind.g = torch.where(mask.unsqueeze(-1), self.wind.g, oldg)

    def _step(self, dt):
        b, m = self.b, self.m
        h = self.pos[:, 2]
        rho, p, a_snd = atmosphere(h)
        self.wind.step(dt)
        wv = self.wind.at(h)
        va = self.vel - wv
        V = va.norm(dim=-1).clamp(min=1e-3)
        vh = va / V.unsqueeze(-1)
        q = 0.5 * rho * V * V
        self.q_dyn, self.mach, self.p_amb = q, V / a_snd, p
        zc = cg_of(m)
        I = inertia(m)
        # engine: thrust grows over the first 0.8 s after ignition; the propellant runs out
        prop = (m - DRY).clamp(min=0.0)
        self.lit = self.lit & (prop > 1.0)
        self.ign = torch.where(self.lit, self.ign + dt, torch.zeros_like(self.ign))
        self.thr_cmd = self.thr_cmd.clamp(THR_MIN, 1.0)
        self.thr = self.thr + (self.thr_cmd - self.thr) * min(1.0, dt / 0.35)
        spool = (self.ign / 0.8).clamp(0.0, 1.0)
        T = torch.where(self.lit, self.engines * self.thr * spool * (T_VAC - (T_VAC - T_SL) * p / P0), torch.zeros_like(h))
        mdot = torch.where(self.lit, self.engines * self.thr * spool * MDOT1, torch.zeros_like(h))
        self.thrust = T
        # flight computer: the angular acceleration that turns b to b_des, then the torque each actuator can give
        lg = (zc - Z_ENG)
        tg = lg * T * math.sin(GIMBAL)
        tf = (Z_FIN - zc) * q * S_FIN * CL_FIN * FIN_MAX * self.fins
        tr = torch.full_like(tg, TAU_RCS)
        tmax = (tg + tf + tr).clamp(min=1.0)
        # turn rate wanted: proportional near the target, and never faster than the torque available can stop
        err = cross(b, self.b_des)
        s_ang = err.norm(dim=-1)
        ang = torch.atan2(s_ang, (b * self.b_des).sum(-1))
        axis = torch.where((s_ang > 1e-6).unsqueeze(-1), err / s_ang.clamp(min=1e-9).unsqueeze(-1), torch.zeros_like(err))
        amax = tmax / I
        rate = torch.minimum(WN * ang, torch.sqrt(2.0 * 0.55 * amax * ang))
        alpha_cmd = (2.0 * ZETA * WN) * (axis * rate.unsqueeze(-1) - self.w)
        tau_cmd = perp(alpha_cmd, b) * I.unsqueeze(-1) * self.hold.unsqueeze(-1)
        u = tau_cmd / tmax.unsqueeze(-1)
        un = u.norm(dim=-1, keepdim=True)
        u = torch.where(un > 1.0, u / un.clamp(min=1e-9), u)
        self.u_g = perp(self.u_g + (u - self.u_g) * min(1.0, dt / 0.08), b)
        self.u_f = perp(self.u_f + (u - self.u_f) * min(1.0, dt / 0.15), b)
        # forces (world) and torques about the centre of mass
        e_g = cross(b, self.u_g) * math.tan(GIMBAL)              # gimbal: thrust leans so the torque points along u
        d = unit(b + e_g)
        F = T.unsqueeze(-1) * d
        tau = cross(-(zc - Z_ENG).unsqueeze(-1) * b, F)
        ax = vh.mul(b).sum(-1, keepdim=True)
        Ca = 1.35 + 0.45 * torch.exp(-((self.mach - 1.25) / 0.6) ** 2) + 0.15 * self.legs
        Ca = torch.where(self.lit, Ca * 0.55, Ca)                 # the plume shields the base
        F = F - (q * AREA * Ca).unsqueeze(-1) * ax * b
        vp = perp(vh, b)
        Fb = -(q * AREA * CN_BODY).unsqueeze(-1) * vp
        Ff = -(q * S_FIN * CN_FIN * self.fins).unsqueeze(-1) * vp
        Fc = (q * S_FIN * CL_FIN * FIN_MAX * self.fins).unsqueeze(-1) * cross(self.u_f, b)
        F = F + Fb + Ff + Fc
        tau = tau + cross((Z_CP - zc).unsqueeze(-1) * b, Fb) + cross((Z_FIN - zc).unsqueeze(-1) * b, Ff + Fc)
        tau = tau + TAU_RCS * u * self.hold.unsqueeze(-1)
        tau = tau - (0.6 * q * AREA * LEN * LEN / V.clamp(min=30.0)).unsqueeze(-1) * self.w
        self.gim, self.finv, self.rcs = e_g, self.u_f * FIN_MAX * self.fins.unsqueeze(-1), u * (tr / tmax).unsqueeze(-1) * self.hold.unsqueeze(-1)
        g = G0 * (R_EARTH / (R_EARTH + h.clamp(min=0.0))) ** 2
        acc = F / m.unsqueeze(-1)
        acc[:, 2] -= g
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        wdot = perp(tau, b) / I.unsqueeze(-1)
        self.w = perp(self.w + wdot * dt, b)
        self.b = unit(b + cross(self.w, b) * dt)
        self.w = perp(self.w, self.b)
        self.m = (m - mdot * dt).clamp(min=DRY)
        self.legs = torch.where(self.legs_cmd, (self.legs + dt / 2.5).clamp(max=1.0), self.legs)
        self.heat = q * V / 1e7    # a glow index for the view (heating goes with q V)
        self.t = self.t + dt

    # --------------------------------------------------------------- geometry
    def feet(self):
        """the centre of the leg feet and the height of the lowest foot"""
        zc = cg_of(self.m)
        ring = self.pos - (zc + LEG_DROP * self.legs).unsqueeze(-1) * self.b
        tilt_s = (1.0 - self.b[:, 2] ** 2).clamp(min=0.0).sqrt()
        return ring, ring[:, 2] - (2.0 + (LEG_R - 2.0) * self.legs) * tilt_s

    def quat(self):
        """orientation as a quaternion (x, y, z, w) turning +z onto b (no roll)"""
        b = self.b
        w = 1.0 + b[:, 2]
        q = torch.stack([-b[:, 1], b[:, 0], torch.zeros_like(w), w], -1)
        flip = w < 1e-6
        q = torch.where(flip.unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0, 0.0], device=b.device, dtype=b.dtype).expand_as(q), q)
        return unit(q)


# ── the launch, computed once: the full stack from the pad to stage separation ────────────────────────────
# Nine engines, throttled to 66 % through the thickest air just after max-Q, a pitch kick a few seconds after
# lift-off and then a gravity turn (the thrust follows the velocity). Main engine cut-off comes when the booster
# still has RESERVE kg of propellant for its entry and landing burns. The second stage then fires its vacuum engine.
STACK0 = 549054.0          # lift-off mass
PROP1 = 411000.0           # first stage propellant
RESERVE = 27000.0          # kept for coming back
S2_THRUST, S2_ISP = 981e3, 348.0


def ascent(kick_t=8.0, kick_deg=2.5, dt=0.02):   # 2.5 deg: cut-off at T+2:34, about 71 km and 2.4 km/s, 23 deg up, so the booster tops out near 119 km (as a Falcon 9 on a drone-ship mission)
    """list of frames (t, x, z, vx, vz, pitch_deg_from_vertical, throttle, mass) and the cut-off state"""
    import math
    m = STACK0
    upper = STACK0 - 25600.0 - PROP1
    x = z = vx = vz = 0.0
    t, frames = 0.0, []
    pitch = 0.0
    mdot9 = 9 * 914e3 / (311.0 * 9.80665)
    while True:
        h = max(z, 0.0)
        T_ = 288.15 - 0.0065 * h if h < 11000 else 216.65 if h < 20000 else 216.65 + 0.001 * (h - 20000) if h < 32000 else 228.65 + 0.0028 * min(h - 32000, 15000)
        rho = (1.225 * (max(T_ / 288.15, 0.5)) ** 4.2559 if h < 11000 else 0.3639 * math.exp(-(h - 11000) / 6341.6) if h < 20000
               else 0.08803 * math.exp(-(h - 20000) / 6500.0) if h < 40000 else 0.003948 * math.exp(-(h - 40000) / 7500.0))
        p = rho * 287.05 * T_
        a = math.sqrt(1.4 * 287.05 * T_)
        V = math.hypot(vx, vz)
        q = 0.5 * rho * V * V
        thr = 0.66 if 66.0 < t < 88.0 else 1.0     # through the thickest air just after max-Q (about T+1:02, 11 km, Mach 1.5)
        if t > 125.0:
            thr = 0.85
        T = 9 * thr * (914e3 - (914e3 - 845e3) * p / 101325.0)
        if t < kick_t:
            pitch = 0.0
        elif t < kick_t + 4.0:
            pitch = math.radians(kick_deg) * (t - kick_t) / 4.0
        else:
            pitch = math.atan2(vx, vz) if V > 1 else pitch
        M = V / a
        cd = 0.32 + 0.28 * math.exp(-((M - 1.15) / 0.45) ** 2)
        D = q * 10.52 * cd
        ux, uz = (vx / V, vz / V) if V > 1 else (0.0, 1.0)
        g = 9.80665 * (6371000.0 / (6371000.0 + h)) ** 2
        ax = (T * math.sin(pitch) - D * ux) / m
        az = (T * math.cos(pitch) - D * uz) / m - g
        if t < 0.5 and az < 0:
            az = 0.0
        frames.append((t, x, z, vx, vz, math.degrees(pitch), thr, m, q, M))
        prop_left = m - upper - 25600.0
        if prop_left <= RESERVE:
            break
        vx += ax * dt; vz += az * dt
        x += vx * dt; z += vz * dt
        m -= thr * mdot9 * dt
        t += dt
    return frames, {"t": t, "x": x, "z": z, "vx": vx, "vz": vz, "m_booster": 25600.0 + RESERVE, "m_upper": upper}


# ── after separation: the flip, the coast, the entry burn (scripted, like the real flight computer) ─────────
# The booster turns its engines toward its direction of flight with the thrusters (the flip), deploys the grid fins,
# coasts over its highest point and, falling back through about 55 km, relights three engines for the entry burn that
# slows it down before the thick air. Then the landing controller (the policy that learned) takes over.
H_ENTRY = 56000.0
ENTRY_KEEP = 6500.0        # propellant kept for the landing burn
V_ENTRY_END = 1300.0


def after_meco(n, dev, dtype, gen=None, noise=True, meco=None):
    """n boosters just separated, at cut-off (with a little spread when noise)"""
    import math
    import torch
    if meco is None:
        _, meco = ascent()
    B = Booster(n, dev, dtype, gen)
    r = lambda: (torch.rand(n, device=dev, dtype=dtype, generator=gen) - 0.5) * 2.0 if noise else torch.zeros(n, device=dev, dtype=dtype)
    V = math.hypot(meco["vx"], meco["vz"]) * (1.0 + 0.006 * r())
    gam = math.atan2(meco["vz"], meco["vx"]) + math.radians(0.4) * r()
    az = math.radians(0.3) * r()
    B.pos[:, 0], B.pos[:, 1], B.pos[:, 2] = meco["x"], 0.0, meco["z"]
    B.vel[:, 0] = V * torch.cos(gam) * torch.cos(az)
    B.vel[:, 1] = V * torch.cos(gam) * torch.sin(az)
    B.vel[:, 2] = V * torch.sin(gam)
    B.b = unit(B.vel.clone())
    B.m[:] = meco["m_booster"] * (1.0 + 0.01 * r())
    B.t[:] = meco["t"]
    B.fins[:] = 0.0
    B.hold[:] = 0.0
    B.thr[:] = 0.0
    B.engines[:] = 1.0
    B.wind.reset(torch.arange(n, device=dev))
    B.phase = torch.zeros(n, dtype=torch.long, device=dev)   # 0 separation, 1 flip, 2 coast, 3 entry burn, 4 handed over
    B.t_sep = meco["t"]
    return B


def script_step(B, dt):
    """one step of the scripted part for every booster not yet handed over; returns the mask of those handed over"""
    import torch
    ph = B.phase
    t_rel = B.t - B.t_sep
    retro = -unit(B.vel)
    # separation: drift apart for 3 s, thrusters off
    to_flip = (ph == 0) & (t_rel > 3.0)
    ph = torch.where(to_flip, torch.ones_like(ph), ph)
    flipping = ph == 1
    B.hold = torch.where(ph >= 1, torch.ones_like(B.hold), B.hold)
    B.b_des = torch.where((ph >= 1).unsqueeze(-1), retro, B.b_des)
    done_flip = flipping & ((B.b * retro).sum(-1) > 0.985) & (B.w.norm(dim=-1) < 0.02)
    ph = torch.where(done_flip, torch.full_like(ph, 2), ph)
    B.fins = torch.where(ph >= 2, (B.fins + dt / 3.0).clamp(max=1.0), B.fins)
    start = (ph == 2) & (B.vel[:, 2] < 0) & (B.pos[:, 2] < H_ENTRY)
    ph = torch.where(start, torch.full_like(ph, 3), ph)
    burning = ph == 3
    B.engines = torch.where(burning, torch.full_like(B.engines, 3.0), B.engines)
    B.lit = B.lit | burning
    B.thr_cmd = torch.where(burning, torch.ones_like(B.thr_cmd), B.thr_cmd)
    V = B.vel.norm(dim=-1)
    stop = burning & (B.ign > 3.0) & ((B.m - DRY <= ENTRY_KEEP) | (V <= V_ENTRY_END))
    B.lit = B.lit & ~stop
    B.engines = torch.where(stop, torch.ones_like(B.engines), B.engines)
    ph = torch.where(stop, torch.full_like(ph, 4), ph)
    B.phase = ph
    return ph == 4


def to_handover(B, dt=0.05, record=None, every=0.25):
    """run the script until every booster is handed over; record(B) every `every` seconds of flight if given"""
    import torch
    nxt = float(B.t[0])
    for _ in range(40000):
        if bool((B.phase == 4).all()):
            break
        script_step(B, dt)
        quiet = bool((B.q_dyn < 30.0).all()) and not bool(B.lit.any()) and bool((B.phase == 2).all())
        h = 0.2 if quiet else dt
        live = (B.phase != 4)
        B.step(h, live)
        if record is not None and float(B.t[0]) >= nxt:
            nxt += every
            record(B)
    return B


def s2_path(meco, seconds=620.0, dt=0.1):
    """the second stage after separation, for the view: it holds the stack's attitude through separation, lights its
    vacuum engine 7 s after cut-off and burns for about six minutes, steering to level off near 200 km, until it
    reaches orbital speed (about 7.8 km/s) or its propellant ends (SECO, near T+8:30); then it coasts in orbit.
    Returns the rows (t, x, 0, z, axis x, 0, axis z, lit) and the time of SECO."""
    import math
    x, z, vx, vz = meco["x"], meco["z"], meco["vx"], meco["vz"]
    m = meco["m_upper"]
    prop = m - 16000.0                    # the stage's own dry mass and the payload stay
    gam0 = math.atan2(vz, vx)
    out, t, seco = [], 0.0, None
    mdot = S2_THRUST / (S2_ISP * 9.80665) * 0.92      # it runs a little below full thrust for most of the burn
    pitch = gam0
    while t <= seconds:
        lit = t >= 7.0 and seco is None
        g = 9.80665 * (6371000.0 / (6371000.0 + z)) ** 2
        cen = (vx * vx) / (6371000.0 + z)   # the flight bends over the curved Earth
        a = mdot * S2_ISP * 9.80665 / m if lit else 0.0
        if lit:
            # steering: climb toward about 200 km and level off there, the rest of the thrust going into speed
            vz_want = max(-60.0, min(900.0, (200000.0 - z) / 90.0))
            need = g - cen + (vz_want - vz) / 12.0
            steer = math.asin(max(-0.35, min(0.75, need / a)))
            u = min(1.0, max(0.0, (t - 8.0) / 25.0)); u = u * u * (3.0 - 2.0 * u)
            pitch = gam0 + (steer - gam0) * u
        out.append((meco["t"] + t, x, 0.0, z, math.cos(pitch), 0.0, math.sin(pitch), 1.0 if lit else 0.0))
        vx += a * math.cos(pitch) * dt
        vz += (a * math.sin(pitch) - g + cen) * dt
        x += vx * dt; z += vz * dt
        if lit:
            m -= mdot * dt
            prop -= mdot * dt
            if math.hypot(vx, vz) >= 7800.0 or prop <= 0.0:
                seco = meco["t"] + t
        else:
            if seco is not None:
                pitch = math.atan2(vz, vx)   # coasting: it keeps pointing along its path
        t += dt
    return out, seco


# ── the landing, as a learning problem ──────────────────────────────────────────────────────────────────
# From the end of the entry burn (or, while learning, from lower down: a curriculum) the controller sees where the
# deck is, how the booster moves and leans, its engine and propellant, and decides 10 times a second: the throttle,
# when to light the landing burn (once: it then burns until touchdown) and which way to lean (up to 24 deg), which
# the flight computer turns into gimbal, grid fins and thrusters. The legs come out 400 m above the deck.
# Landed: on the deck, under 2.5 m/s down and 1.5 m/s sideways, leaning under 8 deg, turning slowly.
OBS = 20
ACT = 4
CTRL_DT, SUBSTEPS = 0.1, 2
TILT_MAX = 0.45            # tan of the lean the controller may ask for (24 deg)
# A reverse curriculum: the first lessons start close to the deck in the middle of a landing burn (states taken along
# rough reference descents flown by a simple guidance law), the next ones further back, the last from the hand-over.
STAGES = [("burning", 30.0, 400.0), ("burning", 400.0, 2200.0), ("falling", 2500.0, 5000.0), ("falling", 5000.0, 12000.0),
          ("falling", 12000.0, 22000.0), ("falling", 22000.0, 34000.0)]
T_LIMIT = 220.0
GAMMA = 0.999


def symlog(x, s):
    import torch
    return torch.sign(x) * torch.log1p(x.abs() / s)


def observe(B):
    import torch
    ring, low = B.feet()
    dc = B.deck.c
    heave, pitch, roll, hdot = B.deck.state(B.t)
    deck_z = dc[:, 2] + DECK_Z + heave
    r = ring - torch.stack([dc[:, 0], dc[:, 1], deck_z], -1)
    v = B.vel - torch.stack([torch.zeros_like(hdot), torch.zeros_like(hdot), hdot], -1)
    _, p, _ = atmosphere(B.pos[:, 2])
    tav = (T_VAC - (T_VAC - T_SL) * p / P0) / (B.m * G0)      # what one engine at full throttle lifts, in weights
    prop = (B.m - DRY).clamp(min=0.0)
    o = torch.cat([
        symlog(r, 10.0) / 4.0,
        symlog(v, 5.0) / 4.0,
        B.b,
        B.w * 5.0,
        B.lit.to(B.m.dtype).unsqueeze(-1),
        B.thr.unsqueeze(-1),
        (prop / 10000.0).unsqueeze(-1),
        B.legs.unsqueeze(-1),
        (torch.log1p(B.q_dyn / 1000.0) / 3.0).unsqueeze(-1),
        tav.unsqueeze(-1),
        (hdot / 2.0).unsqueeze(-1),
        (B.ign_req / 3.0).unsqueeze(-1),
    ], -1)
    return o


def command(B, a):
    """apply the controller's 4 numbers (each -1..1)"""
    import torch
    a = a.clamp(-1.0, 1.0)
    B.thr_cmd = THR_MIN + (1.0 - THR_MIN) * (a[:, 0] + 1.0) * 0.5
    want = a[:, 1] > 0.0
    B.ign_req = torch.where(want, (B.ign_req + 1.0).clamp(max=3.0), torch.zeros_like(B.ign_req))
    light = (B.ign_req >= 3.0) & ~B.lit & ~B.used
    B.used = B.used | light
    B.lit = B.lit | light
    B.engines = torch.ones_like(B.engines)
    up = torch.zeros_like(B.b); up[:, 2] = 1.0
    B.b_des = unit(up + TILT_MAX * torch.stack([a[:, 2], a[:, 3], torch.zeros_like(a[:, 2])], -1))
    ring, low = B.feet()
    B.legs_cmd = B.legs_cmd | ((ring[:, 2] - B.deck.c[:, 2] - DECK_Z) < 400.0)


def contact(B):
    """(touched, landed, on_deck, metrics): the lowest foot reached the deck or the sea"""
    import torch
    ring, low = B.feet()
    dc = B.deck.c
    deck_h = B.deck.height(B.t, ring[:, :2])
    rel = ring[:, :2] - dc[:, :2]
    on = (rel[:, 0].abs() < DECK_X - 6.0) & (rel[:, 1].abs() < DECK_Y - 6.0)
    surf = torch.where(on, deck_h, torch.zeros_like(deck_h))
    touched = low <= surf
    _, _, _, hdot = B.deck.state(B.t)
    vz = B.vel[:, 2] - torch.where(on, hdot, torch.zeros_like(hdot))
    vh = B.vel[:, :2].norm(dim=-1)
    tilt = torch.rad2deg(torch.acos(B.b[:, 2].clamp(-1.0, 1.0)))
    landed = touched & on & (vz > -2.5) & (vh < 1.5) & (tilt < 8.0) & (B.w.norm(dim=-1) < 0.12) & (B.legs > 0.99)
    return touched, landed, on, {"miss": rel.norm(dim=-1), "vz": vz, "vh": vh, "tilt": tilt}


# ── learning to land: PPO (proximal policy optimisation) over thousands of boosters at once ─────────────────
def make_nets(torch, obs=None, act=None, hidden=256):
    import torch.nn as nn
    obs, act = obs or OBS, act or ACT

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act))
            self.v = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.log_std = nn.Parameter(torch.full((act,), -0.6))
            nn.init.orthogonal_(self.pi[-1].weight, 0.01)
            nn.init.zeros_(self.pi[-1].bias)
            # start leaning on the lowest throttle: a lit engine at full thrust from far up only climbs
            with torch.no_grad():
                self.pi[-1].bias[0] = -0.3
                self.pi[-1].bias[1] = -0.5     # lighting asks for 3 decisions in a row: rare at first, not never

        def forward(self, x):
            return self.pi(x), self.v(x).squeeze(-1)
    return Policy()


# ── what the 3D view gets: frames of the flight (esasim.rocket/1) ─────────────────────────────────────────
FIELDS = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "thrust", "engines", "gx", "gy", "gz",
          "fx", "fy", "fz", "legs", "fins", "rx", "ry", "rz", "q", "mach", "heat", "prop", "heave", "pitch", "roll", "wx", "wy"]


def frame_of(B, i, t0=0.0):
    """one booster's state as a row of FIELDS (rounded for the wire)"""
    q = B.quat()[i]
    heave, pitch, roll, _ = B.deck.state(B.t)
    w = B.wind.at(B.pos[:, 2])[i]
    T = float(B.thrust[i])
    r2, r3, r4 = (lambda v: round(float(v), 2)), (lambda v: round(float(v), 3)), (lambda v: round(float(v), 4))
    return [r2(B.t[i] - t0), r2(B.pos[i, 0]), r2(B.pos[i, 1]), r2(B.pos[i, 2]),
            r4(q[0]), r4(q[1]), r4(q[2]), r4(q[3]),
            r2(B.vel[i, 0]), r2(B.vel[i, 1]), r2(B.vel[i, 2]),
            r3(T / T_VAC), int(B.engines[i]) if bool(B.lit[i]) else 0,
            r4(B.gim[i, 0]), r4(B.gim[i, 1]), r4(B.gim[i, 2]),
            r4(B.finv[i, 0]), r4(B.finv[i, 1]), r4(B.finv[i, 2]),
            r3(B.legs[i]), r3(B.fins[i]),
            r3(B.rcs[i, 0]), r3(B.rcs[i, 1]), r3(B.rcs[i, 2]),
            round(float(B.q_dyn[i])), r2(B.mach[i]), r3(B.heat[i]), round(float(B.m[i] - DRY)),
            r3(heave[i]), r4(pitch[i]), r4(roll[i]), r2(w[0]), r2(w[1])]


def chunks(rows, size=200):
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


MOUNT = 10.0   # the engines' exit stands 10 m above the ground on the launch mount


def load_policy(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    net = make_nets(torch, st.get("fields") or OBS, st.get("act") or ACT)
    net.load_state_dict(st["net"])
    net.eval()
    mean, var = st["mean"].double(), st["var"].double()
    net = net.double()

    def act(obs):
        with torch.no_grad():
            x = ((obs - mean) / (var + 1e-8).sqrt()).clamp(-6.0, 6.0)
            mu, _ = net(x)
            return mu
    return act, st


def mission(ctx, act, seed, wind, jet, sea, offset):
    """one flight, computed up to the hand-over at once, then flown by the policy; returns a list of rows to play
    (with the events at their time) and a generator that yields the landing part as it is flown"""
    gen = torch.Generator().manual_seed(seed)
    dt64 = torch.float64
    frames, meco = ascent(dt=0.025)
    rows, events = [], []
    t_lift = 0.0
    qmax_t = max(frames, key=lambda f: f[8])[0]
    for i, f in enumerate(frames[::2]):
        t, x, z, vx, vz, pitch, thr, m, q, M = f
        p = math.radians(pitch)
        qq = [0.0, math.sin(p), 0.0, 1.0 + math.cos(p)]
        nq = math.sqrt(sum(c * c for c in qq))
        prop = m - meco["m_upper"] - DRY
        cg = float(cg_of(torch.tensor(DRY + prop, dtype=torch.float64)))   # rows give the booster's centre of mass
        x, z = x + cg * math.sin(p), z + MOUNT + cg * math.cos(p)
        rows.append([round(t, 2), round(x, 2), 0.0, round(z, 2), 0.0, round(qq[1] / nq, 4), 0.0, round(qq[3] / nq, 4),
                     round(vx, 2), 0.0, round(vz, 2), round(9 * thr * (1.0 if t > 0 else 0.0), 3), 9, 0, 0, 0, 0, 0, 0, 0.0, 0.0,
                     0, 0, 0, round(q), round(M, 2), 0.0, round(prop), 0, 0, 0, 0, 0])
    events += [(0.0, "Liftoff"), (qmax_t, "Max-Q: the highest aerodynamic pressure"), (meco["t"], "Main engine cut-off"),
               (meco["t"] + 3.0, "Stage separation"), (meco["t"] + 7.0, "Second stage engine ignition")]
    B = after_meco(1, "cpu", dt64, gen, noise=False, meco=meco)
    B.pos[:, 2] += MOUNT + float(cg_of(B.m[0])) * float(B.b[0, 2])     # the same reference as the launch rows
    B.pos[:, 0] += float(cg_of(B.m[0])) * float(B.b[0, 0])
    B.wind.reset(torch.arange(1), u0=torch.tensor([float(wind)], dtype=dt64))
    B.wind.j[:] = float(jet)
    last = {"ph": 0}

    def rec(Bx):
        rows.append(frame_of(Bx, 0))
        ph = int(Bx.phase[0])
        if ph != last["ph"]:
            tt = float(Bx.t[0])
            events.append((tt, {1: "The booster flips with its nitrogen thrusters", 2: "Grid fins deployed",
                                3: "Entry burn: three engines", 4: "Entry burn shut-down: the landing controller takes over"}.get(ph, "")))
            last["ph"] = ph
    to_handover(B, dt=0.05, record=rec, every=0.05)
    # the drone ship where the booster would fall without steering, moved by the offset asked
    C = Booster(1, "cpu", dt64, gen)
    for k in Booster.STATE:
        setattr(C, k, getattr(B, k).clone())
    C.fins, C.hold, C.engines = B.fins.clone(), B.hold.clone(), B.engines.clone()
    C.wind.u0, C.wind.d0, C.wind.j, C.wind.dj, C.wind.g = B.wind.u0, B.wind.d0, B.wind.j, B.wind.dj, B.wind.g.clone()
    for _ in range(8000):
        if float(C.pos[0, 2]) <= 0:
            break
        C.b_des = -unit(C.vel - C.wind.at(C.pos[:, 2]))
        C.step(0.05)
    ship = [float(C.pos[0, 0]) + float(offset), float(C.pos[0, 1])]
    B.deck.reset(torch.arange(1), torch.tensor([[ship[0], ship[1], 0.0]], dtype=dt64), sea)
    dk = B.deck   # the swell's model, for the view to move the ship at any moment (the same numbers as the physics)
    meco["deck"] = [round(float(v), 5) for v in (dk.amp_h[0], dk.w_h[0], dk.ph[0, 0], dk.amp_p[0], dk.w_p[0], dk.ph[0, 1], dk.amp_r[0], dk.w_r[0], dk.ph[0, 2])]
    for r in rows:   # the deck's swell in the frames already made
        tt = torch.tensor([r[0]], dtype=dt64)
        hv, pi_, ro, _ = B.deck.state(tt)
        r[28], r[29], r[30] = round(float(hv[0]), 3), round(float(pi_[0]), 4), round(float(ro[0]), 4)
    B.ign_req = torch.zeros(1, dtype=dt64)
    B.used = torch.zeros(1, dtype=torch.bool)
    B.thr_cmd[:] = THR_MIN
    t_hand = float(B.t[0])

    def settle(B, result, seconds=6.0, h=0.001):
        """the booster on the deck after touchdown, as a rigid body on four spring-damper feet (and its engine skirt
        and body if they touch), with friction, on the moving deck; returns the frames (every 0.05 s) and the result"""
        m = float(B.m[0]); zc = float(cg_of(B.m[0]))
        It, Ia = float(inertia(B.m[0])), 3.0e5
        p = [float(B.pos[0, i]) for i in range(3)]; v = [float(B.vel[0, i]) for i in range(3)]
        q = [float(c) for c in B.quat()[0]]                      # x, y, z, w (body z = the booster's axis)
        w = [float(B.w[0, i]) for i in range(3)]                 # world angular velocity
        dk = B.deck
        cx, cy, cz = float(dk.c[0, 0]), float(dk.c[0, 1]), float(dk.c[0, 2])
        ah, wh, ap, wp, ar, wr = float(dk.amp_h[0]), float(dk.w_h[0]), float(dk.amp_p[0]), float(dk.w_p[0]), float(dk.amp_r[0]), float(dk.w_r[0])
        p0, p1, p2 = float(dk.ph[0, 0]), float(dk.ph[0, 1]), float(dk.ph[0, 2])

        def deck(t, x, y):
            hv, pt, rl = ah * math.sin(wh * t + p0), ap * math.sin(wp * t + p1), ar * math.sin(wr * t + p2)
            tp, tr = math.tan(pt), math.tan(rl)
            z = cz + DECK_Z + hv - tp * (x - cx) + tr * (y - cy)
            n = (tp, -tr, 1.0); nl = math.sqrt(tp * tp + tr * tr + 1.0)
            return z, (n[0] / nl, n[1] / nl, n[2] / nl), hv, pt, rl

        def rot(q, a):    # rotate vector a by quaternion q
            x, y, z, w_ = q; ax, ay, az = a
            tx, ty, tz = 2 * (y * az - z * ay), 2 * (z * ax - x * az), 2 * (x * ay - y * ax)
            return (ax + w_ * tx + y * tz - z * ty, ay + w_ * ty + z * tx - x * tz, az + w_ * tz + x * ty - y * tx)

        def crs(a, b):
            return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        # contact points in the body frame (from the centre of mass): the feet 9 m out, 1.2 m below the engine exit,
        # where the view draws them; the engine skirt; points up the side of the tank (it hits them if it falls over)
        pts = [((LEG_R * math.cos(math.radians(45 + 90 * k)), LEG_R * math.sin(math.radians(45 + 90 * k)), -(zc + LEG_DROP)), "foot") for k in range(4)]
        pts.append(((0.0, 0.0, -zc), "body"))
        for k in range(8):
            a = math.radians(45 * k)
            pts += [((1.83 * math.cos(a), 1.83 * math.sin(a), z_ - zc), "body") for z_ in (12.0, 25.0, 40.0)]
        # each leg: a stiff spring until its crushable core yields (FY), then the core crushes for good, up to STROKE;
        # beyond that the leg fails. The body (engine skirt, tank) is much stiffer and never gives
        KF, CF, KB, CB, MU, FY, STROKE = 1.5e6, 1.5e5, 6.0e6, 4.0e5, 0.6, 4.2e5, 0.6
        crushed = [0.0, 0.0, 0.0, 0.0]
        t = float(B.t[0]); t0 = t
        rows, crush, tipped, broke_t, vz0 = [], 0.0, False, None, float(result.get("vz") or 0.0)
        steps, every = int(seconds / h), int(round(0.05 / h))
        tmpl = frame_of(B, 0)
        legs_ok = float(B.legs[0]) >= 0.99                               # legs not locked out: it stands on its engines
        for k in range(steps):
            F = [0.0, 0.0, -G0 * m]; T = [0.0, 0.0, 0.0]
            for j, (bp, kind) in enumerate(pts):
                if kind == "foot" and (not legs_ok or crushed[j] > STROKE):   # a failed leg holds nothing up any more
                    continue
                r = rot(q, bp); x, y, z = p[0] + r[0], p[1] + r[1], p[2] + r[2]
                zs, n, _, _, _ = deck(t, x, y)
                pen = (zs - z) * n[2]
                if pen <= 0.0:
                    continue
                zs2, _, _, _, _ = deck(t + 0.002, x, y)
                vd = (zs2 - zs) / 0.002                                  # the deck's speed up at that point
                vp = crs(w, r); vr = (v[0] + vp[0], v[1] + vp[1], v[2] + vp[2] - vd)
                vn = vr[0] * n[0] + vr[1] * n[1] + vr[2] * n[2]
                if kind == "foot":
                    if KF * (pen - crushed[j]) > FY:
                        crushed[j] = pen - FY / KF               # the core crushes
                    fn = min(FY * 1.1, max(0.0, KF * max(0.0, pen - crushed[j]) - CF * vn))
                    crush = max(crush, crushed[j])
                else:
                    fn = max(0.0, KB * pen - CB * vn)
                    if r[2] > -zc + 1.0 or not legs_ok:      # the tank on the deck, or no legs: it is lost
                        tipped = True
                vt = (vr[0] - vn * n[0], vr[1] - vn * n[1], vr[2] - vn * n[2]); vtl = math.sqrt(vt[0] ** 2 + vt[1] ** 2 + vt[2] ** 2)
                ft = min(MU * fn, 4.0e5 * vtl) / vtl if vtl > 1e-6 else 0.0
                f = (fn * n[0] - ft * vt[0], fn * n[1] - ft * vt[1], fn * n[2] - ft * vt[2])
                F = [F[0] + f[0], F[1] + f[1], F[2] + f[2]]
                tq = crs(r, f); T = [T[0] + tq[0], T[1] + tq[1], T[2] + tq[2]]
            v = [v[i] + F[i] / m * h for i in range(3)]
            p = [p[i] + v[i] * h for i in range(3)]
            # rotation: torque in the body frame, the body's inertia, back to the world
            qi = (-q[0], -q[1], -q[2], q[3])
            wb = rot(qi, w); tb = rot(qi, T)
            Iw = (It * wb[0], It * wb[1], Ia * wb[2]); gy = crs(wb, Iw)
            wb = (wb[0] + (tb[0] - gy[0]) / It * h, wb[1] + (tb[1] - gy[1]) / It * h, wb[2] + (tb[2] - gy[2]) / Ia * h)
            wb = (wb[0] * (1 - 0.3 * h), wb[1] * (1 - 0.3 * h), wb[2] * (1 - 0.3 * h))
            w = list(rot(q, wb))
            x, y, z, w_ = q; ox, oy, oz = w
            q = [x + 0.5 * h * (w_ * ox + y * oz - z * oy), y + 0.5 * h * (w_ * oy + z * ox - x * oz), z + 0.5 * h * (w_ * oz + x * oy - y * ox), w_ + 0.5 * h * (-x * ox - y * oy - z * oz)]
            ql = math.sqrt(sum(c * c for c in q)); q = [c / ql for c in q]
            t += h
            ax_ = rot(q, (0.0, 0.0, 1.0)); tilt = math.degrees(math.acos(max(-1.0, min(1.0, ax_[2]))))
            if broke_t is None and (tipped or tilt > 35.0 or crush > STROKE):
                broke_t = t                                              # the legs give way, or it topples onto the deck
            if (k + 1) % every == 0:
                _, _, hv, pt, rl = deck(t, cx, cy)
                row = list(tmpl)
                row[0] = round(t, 2); row[1], row[2], row[3] = round(p[0], 3), round(p[1], 3), round(p[2], 3)
                row[4], row[5], row[6], row[7] = round(q[0], 5), round(q[1], 5), round(q[2], 5), round(q[3], 5)
                row[8], row[9], row[10] = round(v[0], 3), round(v[1], 3), round(v[2], 3)
                row[11], row[12] = 0.0, 0
                row[21] = row[22] = row[23] = 0.0
                row[24], row[25] = 0, 0.0
                row[28], row[29], row[30] = round(hv, 3), round(pt, 4), round(rl, 4)
                rows.append(row)
        ax_ = rot(q, (0.0, 0.0, 1.0)); tilt_end = math.degrees(math.acos(max(-1.0, min(1.0, ax_[2]))))
        res = dict(result)
        res.update({"physics": True, "crush_m": round(crush, 3), "tilt_end": round(tilt_end, 1)})
        if broke_t is not None:
            res.update({"landed": False, "boom_t": round(broke_t, 2)})
        else:
            res["landed"] = tilt_end < 10.0                               # standing on its legs: landed
        B.t = torch.tensor([t], dtype=B.t.dtype)
        return rows, res

    def landing():
        """yields (rows, events, result) every 0.05 s of flight while the policy flies"""
        result = None
        after = []
        n = 0
        legs_said = burn_said = False
        while result is None:
            obs = observe(B)
            a = act(obs)
            command(B, a)
            ev = []
            if bool(B.used[0]) and not burn_said:
                burn_said = True
                ev.append((float(B.t[0]), "Landing burn"))
            if bool(B.legs_cmd[0]) and not legs_said:
                legs_said = True
                ev.append((float(B.t[0]), "Landing legs deployed"))
            out = []
            for _ in range(SUBSTEPS):
                B.step(CTRL_DT / SUBSTEPS)
                out.append(frame_of(B, 0))
                touched, ok, on, met = contact(B)
                if bool(touched[0]):
                    result = {"landed": bool(ok[0]), "on_deck": bool(on[0]), "miss": round(float(met["miss"][0]), 1),
                              "vz": round(float(met["vz"][0]), 2), "vh": round(float(met["vh"][0]), 2), "tilt": round(float(met["tilt"][0]), 1),
                              "t_touch": round(float(B.t[0]), 2)}
                    break
            n += 1
            if result is not None and result.get("on_deck"):
                B.lit[:] = False
                after, result = settle(B, result)     # what the contact does decides the outcome, before it is told
            if result is None and (float(B.t[0]) - t_hand > T_LIMIT or float(B.b[0, 2]) < 0.0):
                result = {"landed": False, "on_deck": False, "miss": None, "vz": None, "lost": True}
            yield out, ev, result
        # after touchdown: the engine stops. On the deck, a rigid-body contact simulation decides what happens: the four
        # feet (crushable shock absorbers) take the impact, the booster settles and rides the deck's heave, pitch and
        # roll, or tips over and breaks up if it came in too fast or too tilted. In the sea it is simply lost.
        B.lit[:] = False
        if result.get("physics"):
            for r in after:
                yield [r], [], result
            return
        land_pos = B.pos.clone()
        for k in range(120):   # 6 s after touchdown, then the flight ends
            t = B.t + 0.05
            B.t = t
            yield [frame_of(B, 0)], [], result
    return rows, sorted(events), landing, ship, meco, t_hand


def run(ctx):
    import json
    import random
    import time
    p = ctx.params
    if torch is None:
        pick_device("cpu")
    torch.set_num_threads(1)
    # every policy the trainings left (in their dataset): the one asked by name, else the newest (the date and time in
    # its name, booster_policy_<YYYYmmdd_HHMM>_<NN>pct.pt; a file without them counts as the oldest)
    import os
    import re
    pols = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                pols.append((m_.group(1) if m_ else "", os.path.getmtime(f.path), float(h.get("landed_full") or 0), float(h.get("minutes") or 0), f))
            except Exception as e:  # a damaged file: say so, use the others
                ctx.log("Skipped %s: %s" % (f.name, e))
    if not pols:
        raise RuntimeError("No policy: wire the Policy output of Booster landing: train (its dataset of booster_policy_*.pt) to this block.")
    pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
    pols = [(l, mi, f) for _, _, l, mi, f in pols]
    ctx.log("%d policies, newest first: %s" % (len(pols), ", ".join("%s (%.0f%% landed, %.0f min)" % (f.name, 100 * l, mi) for l, mi, f in pols)))
    want = str(p.get("policy_file") or "").strip()
    pick = [x for x in pols if want and want in str(x[2].name)]
    if want and not pick:
        ctx.log("No policy named like %s: using the newest." % want)
    ready = [x for x in pols if x[0] >= 0.3]      # trainings that got through every lesson and land full flights
    if not pick and ready and ready[0] is not pols[0]:
        ctx.log("The newest policy, %s, has not finished its lessons yet (%.0f%% of full flights land): flying %s, the newest that lands." % (
            pols[0][2].name, 100 * pols[0][0], ready[0][2].name))
    pol = (pick or ready or pols)[0][2]
    act, st = load_policy(pol.path)
    ctx.log("Landing policy %s: trained %.0f minutes on %s, %s of full flights landed when it was saved." % (
        pol.name, float(st.get("minutes") or 0), st.get("device") or "?", "%.0f%%" % (100 * float(st.get("landed_full") or 0))))
    early = float(st.get("landed_full") or 0) < 0.3
    if early:
        ctx.log("Warning: this policy is an early one (it landed %.0f%% of full flights when saved, after %.0f minutes of learning): "
                "expect the booster to be lost. Train longer and use the policy written at the end of training." % (100 * float(st.get("landed_full") or 0), float(st.get("minutes") or 0)))
    ch, ch_ev, ch_ctl = p.get("channel") or "rocket", p.get("events_channel") or "event", p.get("control_channel") or "control"
    S = {"wind": float(p.get("wind") if p.get("wind") is not None else 6), "jet": float(p.get("jet") if p.get("jet") is not None else 25),
         "sea": float(p.get("sea") if p.get("sea") is not None else 0.4), "warp": float(p.get("warp") or 1), "offset": float(p.get("offset") or 0)}
    auto = bool(p.get("auto_launch"))
    per_run = int(p.get("flights_per_run") if p.get("flights_per_run") is not None else 1)
    flights = 0
    log = open(ctx.output("flights", "flights.csv"), "w", encoding="utf-8")
    log.write("flight,seed,wind,jet,sea,landed,on_deck,miss_m,touchdown_mps,tilt_deg\n")

    def controls(wait):
        go = False
        for m in ctx.messages(ch_ctl, wait=wait):
            d = m.get("data") if isinstance(m, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c == "launch":
                go = True
            elif c in ("wind", "jet", "sea", "warp", "offset"):
                try:
                    S[c] = float(v)
                    if c == "sea" and S[c] > 1:
                        S[c] = S[c] / 100.0
                except (TypeError, ValueError):
                    pass
        return go

    ctx.telemetry(ch_ev, {"text": "Ready: press Launch" + (" (warning: an early landing policy, it will rarely land)" if early else "")})
    ctx.telemetry(ch, {"format": "esasim.rocket/1", "kind": "idle", "ep": 0, "frames": [], "fields": FIELDS, "phase": "On the pad", "status": "Ready for launch"})
    while True:
        if not auto and not controls(0.5):
            continue
        auto = False
        flights += 1
        seed = random.randrange(1, 10 ** 6)
        ctx.log("Flight %d: wind %.0f m/s at the sea, jet stream %.0f m/s, sea state %.1f, time x%.0f." % (flights, S["wind"], S["jet"], S["sea"], S["warp"]))
        rows, events, landing, ship, meco, t_hand = mission(ctx, act, seed, S["wind"], S["jet"], S["sea"], S["offset"])
        s2, seco_t = s2_path(meco)
        if seco_t is not None:
            events.append((seco_t, "Second stage engine cut-off: in orbit"))
            events.sort()
        base = {"format": "esasim.rocket/1", "kind": "mission", "ep": flights, "fields": FIELDS, "ship": [round(ship[0], 1), round(ship[1], 1)],
                "pad": [0.0, 0.0], "meco_t": round(meco["t"], 2), "sep_t": round(meco["t"] + 3.0, 2), "handover_t": round(t_hand, 2),
                "label": "Flight %d" % flights, "wind": S["wind"], "sea": S["sea"], "deck": meco.get("deck")}
        # countdown, then the flight in real time (times the warp)
        warp = max(0.25, min(20.0, S["warp"]))
        for k in range(10, 0, -1):
            ctx.telemetry(ch, dict(base, frames=[], phase="Countdown", clock=-k, warp=warp, status="T-%d s" % k))
            if k == 10:
                ctx.telemetry(ch_ev, {"text": "Flight %d: T-10 s" % flights})
            if k == 3:
                ctx.telemetry(ch_ev, {"text": "Engine ignition sequence"})
            time.sleep(1.0)        # the countdown always runs in real seconds
        ev_i = 0
        buf = []
        t_wall = time.time()
        t_sim0 = rows[0][0]
        seq = 0
        result = None

        def phase_of(t):
            if t < meco["t"]:
                return "Ascent"
            if t < meco["t"] + 3:
                return "Main engine cut-off"
            if t < t_hand:
                return "Coast and entry burn"
            return "Landing"

        def flush(final=False):
            nonlocal buf, seq
            if not buf:
                return
            last = buf[-1]
            t0_, t1_ = buf[0][0], last[0]
            s2rows = [[round(r[0], 2), round(r[1], 1), 0.0, round(r[3], 1), round(r[4], 4), 0.0, round(r[6], 4), r[7]] for r in s2 if t0_ - 0.05 <= r[0] <= t1_ + 0.05]
            V = math.sqrt(last[8] ** 2 + last[9] ** 2 + last[10] ** 2)
            item = dict(base, seq=seq, frames=buf, s2=s2rows, warp=warp, clock=round(last[0], 1), phase=phase_of(last[0]),
                        alt_km=round(last[3] / 1000.0, 2), speed_kmh=round(V * 3.6), throttle=round(100 * last[11] / max(1, last[12] or 1) if last[12] else 0),
                        prop_t=round(last[27] / 1000.0, 2), downrange_km=round(last[1] / 1000.0, 1), q_kpa=round(last[24] / 1000.0, 1),
                        to_ship_km=round(math.hypot(last[1] - ship[0], last[2] - ship[1]) / 1000.0, 2))
            if result is not None:
                item["result"] = result
            if final:
                item["last"] = True
            ctx.telemetry(ch, item)
            seq += 1
            buf = []

        def pace(t):
            nonlocal t_wall
            ahead = (t - t_sim0) / warp - (time.time() - t_wall)
            if ahead > 0:
                time.sleep(ahead)

        def play(rs):
            nonlocal ev_i
            for r in rs:
                while ev_i < len(events) and events[ev_i][0] <= r[0]:
                    if events[ev_i][1]:
                        ctx.telemetry(ch_ev, {"text": events[ev_i][1], "t": events[ev_i][0]})
                    ev_i += 1
                buf.append(r)
                if len(buf) >= 20:
                    pace(r[0])
                    flush()
                    controls(0)
        play(rows)
        for out, ev, res in landing():
            for e in ev:
                events.append(e)
            events.sort()
            if res is not None and result is None:
                result = res
                txt = ("The booster has landed: %.1f m from the centre, %.1f m/s at touchdown" % (res["miss"], abs(res["vz"])) if res.get("landed")
                       else "Lost the booster: %s" % ("hard landing on the deck" if res.get("on_deck") else "in the sea" if not res.get("lost") else "out of control"))
                if res.get("boom_t") is not None:
                    txt = "Lost the booster on the deck: %s" % ("it came down too hard and the legs gave way" if res.get("crush_m", 0) > 0.59 else "it toppled over")
                events.append((res.get("boom_t") or out[-1][0], txt))
                log.write("%d,%d,%.1f,%.1f,%.2f,%s,%s,%s,%s,%s\n" % (flights, seed, S["wind"], S["jet"], S["sea"], res.get("landed"), res.get("on_deck"), res.get("miss"), res.get("vz"), res.get("tilt")))
                log.flush()
                ctx.log(txt + ".")
            play(out)
        flush(final=True)
        if per_run > 0 and flights >= per_run:   # the run ends by itself
            ctx.telemetry(ch_ev, {"text": "Flight over: run again for another one"})
            ctx.log("%d flight%s flown: the run ends." % (flights, "" if flights == 1 else "s"))
            log.close()
            return
        ctx.telemetry(ch_ev, {"text": "Ready: press Launch for the next flight"})