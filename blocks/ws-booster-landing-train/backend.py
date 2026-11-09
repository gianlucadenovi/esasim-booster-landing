# ── EsaSim block ──────────────────────────────────────────────────────
# Booster landing: train (ws-booster-landing-train) · version 3 · machine block, Python: def run(ctx)
# A Falcon 9-class booster learns to land on a drone ship by reinforcement learning (PPO), thousands of boosters at once on the GPU.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-09
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Continue from · optional · The policies' dataset (booster_policy*.pt): training goes on from the furthest one (the lesson it reached, then its landing rate)
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · booster_policy_<date>_<NN>pct.pt: the landing policy (network, input scaling, lesson reached, landing rate), written when learning ends
#   curve       Table (csv)               Learning curve · training.csv: minutes, lesson, landing rates, miss, touchdown speed, steps per second
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   accelerator select    Accelerator = "auto" of auto | NVIDIA GPU | CPU
#   boosters    number    Boosters at once = 4096
#   minutes     number    Minutes at most = 60
#   target      number    Stop at landing rate = 0.9
#   replay_every number    Replay every (s) = 15
#   from_scratch bool      Start from scratch = false
#   seed        number    Seed = 1
#   channel     text      Training channel = "train"
#   rocket_channel text      Flight channel = "rocket"
#   events_channel text      Events channel = "event"
#   control_channel text      Controls channel = "control"
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch[cuda] (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Booster landing: train. A Falcon 9-class first stage learns to land on a drone ship at sea, by reinforcement
learning (PPO) over thousands of boosters flown at once on the GPU. Each attempt starts where the scripted flight
hands over (after the launch, the flip and the entry burn, about 37 km up at 1.3 km/s) or, while it learns, closer to
the deck: six lessons, the first two starting in the middle of a landing burn (30-400 m, then 0.4-2.2 km), the next
ones falling from 2.5-5 km, 5-12 km, 12-22 km and the hand-over; a lesson opens when half the attempts of the one
before land. Attempts land when they touch the deck under 2.5 m/s down and 1.5 m/s sideways. The physics: gravity, a standard
atmosphere, drag and lift of the body and grid fins, one Merlin-class engine throttling 40-100 % that can be lit once,
gimbal, cold-gas thrusters, wind growing with height with a jet stream and gusts, and a deck that heaves, pitches and
rolls. Every few seconds one attempt is replayed in the 3D view; the curve goes to the Training channel and a CSV. The
policy (booster_policy.pt) is written when learning ends (the time limit, the target, or Finish now), with the best one
seen on full flights."""
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
# Nine engines, throttled to 72 % through the region of highest dynamic pressure, a pitch kick a few seconds after
# lift-off and then a gravity turn (the thrust follows the velocity). Main engine cut-off comes when the booster
# still has RESERVE kg of propellant for its entry and landing burns. The second stage then fires its vacuum engine.
STACK0 = 549054.0          # lift-off mass
PROP1 = 411000.0           # first stage propellant
RESERVE = 27000.0          # kept for coming back
S2_THRUST, S2_ISP = 981e3, 348.0


def ascent(kick_t=8.0, kick_deg=2.2, dt=0.02):   # 2.2 deg: cut-off at about 69 km and 2.4 km/s, 23 deg up, so the booster tops out near 117 km (as a Falcon 9 on a drone-ship mission)
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
        thr = 0.72 if 48.0 < t < 74.0 else 1.0
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


def s2_path(meco, seconds=180.0, dt=0.1):
    """the second stage after separation, for the view: it fires 7 s after cut-off and pitches down to the horizon"""
    import math
    x, z, vx, vz = meco["x"], meco["z"], meco["vx"], meco["vz"]
    m = meco["m_upper"]
    gam0 = math.atan2(vz, vx)
    out, t = [], 0.0
    mdot = S2_THRUST / (S2_ISP * 9.80665)
    while t <= seconds:
        lit = t >= 7.0
        g = 9.80665 * (6371000.0 / (6371000.0 + z)) ** 2
        gam = math.atan2(vz, vx)
        steer = gam - math.radians(4.0) * min(1.0, t / 60.0) + math.radians(10.0) * max(0.0, 1.0 - t / 90.0)
        # it keeps the stack's attitude through separation (the booster slides away behind it) and only starts to steer
        # once its engine is running, smoothly
        u = min(1.0, max(0.0, (t - 8.0) / 25.0)); u = u * u * (3.0 - 2.0 * u)
        pitch = gam0 + (steer - gam0) * u
        a = S2_THRUST / m if lit else 0.0
        cen = (vx * vx) / (6371000.0 + z)   # the flight bends over the curved Earth
        out.append((meco["t"] + t, x, 0.0, z, math.cos(pitch), 0.0, math.sin(pitch), 1.0 if lit else 0.0))
        vx += a * math.cos(pitch) * dt
        vz += (a * math.sin(pitch) - g + cen) * dt
        x += vx * dt; z += vz * dt
        if lit:
            m -= mdot * dt
        t += dt
    return out


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


def reference(B):
    """a rough guidance law (not the learner): light when the stopping distance at 75 % of full thrust reaches the
    height, throttle to the deceleration needed, lean toward the deck; it only provides start states"""
    import torch
    ring, low = B.feet()
    h = (ring[:, 2] - B.deck.c[:, 2] - DECK_Z).clamp(min=0.0)
    r = ring[:, :2] - B.deck.c[:, :2]
    _, p, _ = atmosphere(B.pos[:, 2])
    tmax = T_VAC - (T_VAC - T_SL) * p / P0
    amax = tmax / B.m - G0
    vz = -B.vel[:, 2]
    need = vz * vz / (2 * 0.75 * amax.clamp(min=1.0))
    ign = ((need >= h) & (h < 9000.0)) | B.lit
    acc = vz * vz / (2 * h.clamp(min=1.0)) + G0
    thr = (acc * B.m / tmax).clamp(THR_MIN, 1.0)
    a0 = (thr - THR_MIN) / (1 - THR_MIN) * 2 - 1
    tgo = (h / vz.clamp(min=5.0)).clamp(min=2.0)
    acc_h = -2 * (r + B.vel[:, :2] * tgo.unsqueeze(-1)) / (tgo * tgo).unsqueeze(-1)
    lean = (acc_h / G0).clamp(-1, 1) / TILT_MAX * (h / 60.0).clamp(max=1.0).unsqueeze(-1)
    lean = torch.where(B.lit.unsqueeze(-1), lean, -lean * 0.5)
    return torch.stack([a0, torch.where(ign, torch.ones_like(a0), -torch.ones_like(a0)), lean[:, 0], lean[:, 1]], -1)


def offset_radius(h):
    """how far from the deck a start may aim (the controller has to fly the difference)"""
    return 30.0 + 0.025 * h


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


def potential(B):
    """how close to a good landing a state is (higher is better): the distance to the deck and the gap to a descent
    speed that suits the height, both on a log scale, plus lean and turning near the deck. Climbing always lowers it."""
    import torch
    ring, low = B.feet()
    dc = B.deck.c
    heave, _, _, hdot = B.deck.state(B.t)
    rz = ring[:, 2] - dc[:, 2] - DECK_Z - heave
    h = rz.clamp(min=0.0)
    rh = ring[:, :2] - dc[:, :2]
    dist = torch.sqrt((rh ** 2).sum(-1) + rz ** 2)
    vz_t = -(torch.sqrt(2.0 * 7.0 * h) + 1.0)
    tau = (h / 60.0).clamp(min=4.0)
    vh_t = -rh / tau.unsqueeze(-1)
    vh_t = vh_t * (150.0 / vh_t.norm(dim=-1).clamp(min=150.0)).unsqueeze(-1)
    ev = torch.sqrt((B.vel[:, 2] - hdot - vz_t) ** 2 + ((B.vel[:, :2] - vh_t) ** 2).sum(-1))
    tilt = torch.acos(B.b[:, 2].clamp(-1.0, 1.0))
    low_w = (1.0 - h / 600.0).clamp(min=0.0)
    return -(2.0 * torch.log1p(dist / 30.0) + 2.0 * torch.log1p(ev / 15.0) + (2.0 * tilt + 0.5 * B.w.norm(dim=-1)) * low_w)


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


class LandingEnv:
    def __init__(self, n, dev, dtype=None, seed=0, pool=None):
        import torch
        self.n, self.dev = n, torch.device(dev)
        self.dtype = dtype or torch.float32
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.B = Booster(n, self.dev, self.dtype, self.gen)
        self.B.ign_req = torch.zeros(n, device=self.dev, dtype=self.dtype)
        self.B.used = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.B.t0 = torch.zeros(n, device=self.dev, dtype=self.dtype)
        self.pools = {}
        self.mix = [1.0, 0.0, 0.0, 0.0]
        self.ret = torch.zeros(n, device=self.dev, dtype=self.dtype)
        self.stage_of = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.meco = None

    # ------------------------------------------------------------ start states
    def make_pool(self, stage, size, log=None):
        """start states for one lesson. Falling lessons: real hand-overs (after the scripted flight) flown passively down
        to the lesson's height band, the deck where that fall would hit, give or take offset_radius. Burning lessons:
        from 3-5 km, the reference guidance flies toward the deck and each start is taken where it crosses its height."""
        import torch
        dev, dt = self.dev, self.dtype
        if self.meco is None:
            _, self.meco = ascent()
        kind, lo, hi = STAGES[stage]
        B = after_meco(size, dev, dt, self.gen, noise=True, meco=self.meco)
        to_handover(B)
        flo, fhi = (3000.0, 5000.0) if kind == "burning" else (lo, hi)
        target = flo + (fhi - flo) * torch.rand(size, device=dev, dtype=dt, generator=self.gen)
        target = torch.minimum(target, B.pos[:, 2] - 1.0)
        B.wind.reset(torch.arange(size, device=dev))
        B.hold[:] = 1.0
        B.lit[:] = False
        B.engines[:] = 1.0
        B.ign_req = torch.zeros(size, device=dev, dtype=dt)
        B.used = torch.zeros(size, dtype=torch.bool, device=dev)

        def trim():
            B.b_des = -unit(B.vel - B.wind.at(B.pos[:, 2]))
        for _ in range(8000):
            live = B.pos[:, 2] > target
            if not bool(live.any()):
                break
            trim()
            B.step(0.05, live)
        # where the passive fall would end (on a copy)
        C = Booster(size, dev, dt, self.gen)
        for k in Booster.STATE:
            setattr(C, k, getattr(B, k).clone())
        C.fins, C.hold, C.engines = B.fins.clone(), B.hold.clone(), B.engines.clone()
        C.wind.u0, C.wind.d0, C.wind.j, C.wind.dj, C.wind.g = B.wind.u0, B.wind.d0, B.wind.j, B.wind.dj, B.wind.g.clone()
        for _ in range(8000):
            live = C.pos[:, 2] > 0.0
            if not bool(live.any()):
                break
            C.b_des = -unit(C.vel - C.wind.at(C.pos[:, 2]))
            C.step(0.05, live)
        impact = C.pos[:, :2].clone()
        ang = 2 * math.pi * torch.rand(size, device=dev, dtype=dt, generator=self.gen)
        rad = offset_radius(B.pos[:, 2]) * torch.sqrt(torch.rand(size, device=dev, dtype=dt, generator=self.gen))
        deck = impact + torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
        B.pos[:, :2] -= deck            # the deck at the origin of each start
        B.deck.reset(torch.arange(size, device=dev), torch.zeros(size, 3, device=dev, dtype=dt), None)
        if kind == "burning":
            # fly the reference guidance down, taking each start where it passes its own height
            tgt = lo + (hi - lo) * torch.rand(size, device=dev, dtype=dt, generator=self.gen)
            for _ in range(6000):
                ring, _low = B.feet()
                live = (ring[:, 2] - DECK_Z) > tgt
                if not bool(live.any()):
                    break
                a = reference(B)
                old_lit = B.lit.clone()
                command(B, a)
                B.lit = torch.where(live, B.lit, old_lit)
                for _ in range(SUBSTEPS):
                    B.step(CTRL_DT / SUBSTEPS, live)
            # a little spread: the deck a few metres elsewhere, so the starts are not all on the reference line
            ang = 2 * math.pi * torch.rand(size, device=dev, dtype=dt, generator=self.gen)
            rad = (3.0 + 0.04 * tgt) * torch.sqrt(torch.rand(size, device=dev, dtype=dt, generator=self.gen))
            B.pos[:, 0] += rad * torch.cos(ang)
            B.pos[:, 1] += rad * torch.sin(ang)
        snap = {k: getattr(B, k).clone() for k in Booster.STATE}
        snap["wind"] = (B.wind.u0.clone(), B.wind.d0.clone(), B.wind.j.clone(), B.wind.dj.clone(), B.wind.g.clone())
        snap["fins"] = B.fins.clone()
        snap["thr_cmd"] = B.thr_cmd.clone()
        self.pools[stage] = snap
        if log:
            V = snap["vel"].norm(dim=-1)
            log("lesson %d (%s): %d starts from %.2f to %.2f km, %.0f to %.0f m/s, %d%% with the engine lit" % (
                stage + 1, kind, size, snap["pos"][:, 2].min() / 1e3, snap["pos"][:, 2].max() / 1e3, V.min(), V.max(), 100 * float(snap["lit"].float().mean())))
        return snap

    def reset(self, idx):
        import torch
        k = idx.numel()
        if k == 0:
            return
        stages = [s for s in range(len(STAGES)) if self.mix[s] > 0 and s in self.pools]
        w = torch.tensor([self.mix[s] for s in stages], device=self.dev, dtype=self.dtype)
        pick = torch.multinomial(w, k, replacement=True, generator=self.gen)
        B = self.B
        for j, s in enumerate(stages):
            sel = idx[pick == j]
            if sel.numel() == 0:
                continue
            P = self.pools[s]
            rows = torch.randint(0, P["pos"].shape[0], (sel.numel(),), device=self.dev, generator=self.gen)
            for key in Booster.STATE:
                getattr(B, key)[sel] = P[key][rows]
            B.fins[sel] = P["fins"][rows]
            B.thr_cmd[sel] = P["thr_cmd"][rows]
            u0, d0, jj, dj, g = P["wind"]
            B.wind.u0[sel], B.wind.d0[sel], B.wind.j[sel], B.wind.dj[sel], B.wind.g[sel] = u0[rows], d0[rows], jj[rows], dj[rows], g[rows]
            B.deck.reset(sel, torch.zeros(sel.numel(), 3, device=self.dev, dtype=self.dtype), None)
            self.stage_of[sel] = s
        B.used[idx] = B.lit[idx]
        B.ign_req[idx] = torch.where(B.lit[idx], torch.full_like(B.ign_req[idx], 3.0), torch.zeros_like(B.ign_req[idx]))
        B.engines[idx] = 1.0
        B.hold[idx] = 1.0
        B.legs_cmd[idx] = B.legs[idx] > 0.0
        B.t0[idx] = B.t[idx]
        self.ret[idx] = 0.0

    def start(self):
        import torch
        self.reset(torch.arange(self.n, device=self.dev))
        return observe(self.B)

    # ------------------------------------------------------------ one decision
    def step(self, a):
        """returns obs, reward, done, info (for the episodes that ended: landed, miss, vz, vh, tilt, stage)"""
        import torch
        B = self.B
        phi0 = potential(B)
        m0 = B.m.clone()
        command(B, a)
        done = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        landed = torch.zeros_like(done)
        on_deck = torch.zeros_like(done)
        met = None
        for _ in range(SUBSTEPS):
            B.step(CTRL_DT / SUBSTEPS, ~done)
            touched, ok, on, m = contact(B)
            new = touched & ~done
            if met is None:
                met = {k: torch.zeros_like(v) for k, v in m.items()}
            for k in met:
                met[k] = torch.where(new, m[k], met[k])
            landed = landed | (new & ok)
            on_deck = on_deck | (new & on)
            done = done | new
        crashed_air = (B.b[:, 2] < 0.0) & ~done          # tumbling over
        timeout = (B.t - B.t0) > T_LIMIT
        out_fuel = torch.zeros_like(done)          # out of propellant: it falls and hits, which says it all
        end = done | crashed_air | timeout | out_fuel
        phi1 = potential(B)
        # potential-based shaping (zero potential once it ends, so the best policy stays the same), propellant, and a
        # cost for going up: with discounting, climbing away would otherwise put off a crash
        r = (GAMMA * phi1 * (~end).to(phi1.dtype) - phi0) - 0.002 * (m0 - B.m) - 0.05 * B.vel[:, 2].clamp(min=0.0)
        bonus = torch.zeros_like(r)
        prec = (1.0 - met["miss"] / 20.0).clamp(min=0.0)
        soft = (1.0 - (-met["vz"]).clamp(min=0.0) / 2.5).clamp(min=0.0)
        bonus = torch.where(landed, 100.0 + 20.0 * prec + 20.0 * soft, bonus)
        hit = done & ~landed
        bonus = torch.where(hit, -50.0 - (met["vz"].abs() + met["vh"]).clamp(max=200.0) * 0.3 + 15.0 * on_deck.to(r.dtype), bonus)
        bonus = torch.where(~done & (crashed_air | timeout), torch.full_like(bonus, -120.0), bonus)
        r = r + bonus
        self.ret += r
        why = torch.zeros(self.n, dtype=torch.long, device=self.dev)       # 1 landed 2 deck crash 3 sea 4 tumbled 5 no fuel 6 time
        why = torch.where(landed, 1, torch.where(done & on_deck, 2, torch.where(done, 3, torch.where(crashed_air, 4, torch.where(out_fuel, 5, torch.where(timeout, 6, 0))))))
        info = {"end": end, "why": why, "landed": landed, "on_deck": on_deck, "touched": done, "miss": met["miss"], "vz": met["vz"], "vh": met["vh"],
                "tilt": met["tilt"], "stage": self.stage_of.clone(), "ret": self.ret.clone(), "prop": (B.m - DRY)}
        idx = end.nonzero(as_tuple=False).flatten()
        self.reset(idx)
        return observe(B), r, end, info


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


class RunningNorm:
    def __init__(self, torch, n, dev):
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-6.0, 6.0)


class PPO:
    def __init__(self, torch, dev, n_env, horizon=32, lr=3e-4, gamma=GAMMA, lam=0.95, clip=0.2, epochs=4, minibatches=4, rscale=0.1):
        self.torch, self.dev = torch, dev
        self.net = make_nets(torch).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.norm = RunningNorm(torch, OBS, dev)
        self.n, self.T = n_env, horizon
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.rscale = gamma, lam, clip, epochs, minibatches, rscale
        self.lr = lr

    def act(self, obs, deterministic=False):
        torch = self.torch
        with torch.no_grad():
            x = self.norm(obs)
            mu, v = self.net(x)
            if deterministic:
                return mu, None, v
            std = self.net.log_std.exp()
            a = mu + std * torch.randn_like(mu)
            lp = (-0.5 * ((a - mu) / std) ** 2 - self.net.log_std - 0.9189385).sum(-1)
            return a, lp, v

    def update(self, buf, last_v):
        torch = self.torch
        obs, act, lp, val, rew, done = buf
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        gae = torch.zeros_like(rew[0])
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t].float()
            delta = rew[t] * self.rscale + self.gamma * nv * nonterm - val[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val
        N = T * self.n
        obs, act, lp, adv, ret, val = obs.reshape(N, -1), act.reshape(N, -1), lp.reshape(N), adv.reshape(N), ret.reshape(N), val.reshape(N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        x = self.norm(obs)
        stats = {"kl": 0.0, "vloss": 0.0, "ent": 0.0}
        size = N // self.mb
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.dev)
            for i in range(self.mb):
                idx = perm[i * size:(i + 1) * size]
                mu, v = self.net(x[idx])
                ls = self.net.log_std
                std = ls.exp()
                nlp = (-0.5 * ((act[idx] - mu) / std) ** 2 - ls - 0.9189385).sum(-1)
                ratio = (nlp - lp[idx]).exp()
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                vc = val[idx] + (v - val[idx]).clamp(-1.0, 1.0)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                ent = (ls + 1.4189385).sum()
                loss = pl + 0.5 * vl - 0.004 * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(-2.2, 0.0)
                    stats["kl"] = float((lp[idx] - nlp).mean())
                    stats["vloss"] = float(vl)
        stats["std"] = float(self.net.log_std.exp().mean())
        return stats

    def state(self):
        return {"net": self.net.state_dict(), "mean": self.norm.mean.cpu(), "var": self.norm.var.cpu(), "count": self.norm.count}


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


def run(ctx):
    import collections
    import csv
    import time
    import torch
    p = ctx.params
    dev, label, kind = pick_device(p.get("accelerator") or "auto")
    n = int(p.get("boosters") or 4096)
    if kind == "cpu":
        n = min(n, 1024)
        torch.set_num_threads(max(1, torch.get_num_threads()))
    minutes = float(p.get("minutes") or 60)
    target = float(p.get("target") or 0.9)
    replay_every = float(p.get("replay_every") or 20)
    ch_rocket, ch_train, ch_ctl = p.get("rocket_channel") or "rocket", p.get("channel") or "train", p.get("control_channel") or "control"
    seed = int(p.get("seed") or 1)
    torch.manual_seed(seed)
    ctx.log("Learning to land on %s with %d boosters at once, for up to %.0f minutes (or until %.0f%% of full flights land)." % (label, n, minutes, target * 100))
    algo = PPO(torch, dev, n)
    start_stage = 0
    # every training goes on from the furthest one so far (the policies' dataset, wired to Policy so far): the lesson it
    # reached, then how often it lands full flights, then the newest; so an hour cut short is never lost
    import os
    import re
    found = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                st_ = torch.load(f.path, map_location="cpu", weights_only=False)
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                found.append((int(st_.get("stage", 0)), float(st_.get("landed_full") or 0), m_.group(1) if m_ else "", os.path.getmtime(f.path), f, st_))
            except Exception as e:
                ctx.log("Skipped %s: %s" % (f.name, e))
    if found and not bool(p.get("from_scratch")):
        found.sort(key=lambda x: x[:4], reverse=True)
        f, st = found[0][4], found[0][5]
        algo.net.load_state_dict(st["net"])
        algo.norm.mean, algo.norm.var, algo.norm.count = st["mean"].to(dev), st["var"].to(dev), st["count"]
        start_stage = int(st.get("stage", 0))
        seed += int(time.time()) % 100000          # new attempts, not the same ones again
        torch.manual_seed(seed)
        ctx.log("Continuing from %s: it had reached lesson %d of %d after %.0f minutes, %.0f%% of full flights landing." % (
            f.name, start_stage + 1, len(STAGES), float(st.get("minutes") or 0), 100 * float(st.get("landed_full") or 0)))
    elif found:
        ctx.log("Starting from scratch (asked): the %d policies so far are left as they are." % len(found))
    env = LandingEnv(n, dev, seed=seed)
    pool = max(4096, min(16384, 4 * n))
    ctx.progress(1, "computing the flights to start from")
    for s in range(start_stage + 1):
        env.make_pool(s, pool, log=ctx.log)
    stage = start_stage
    env.mix = [0.0] * len(STAGES)
    env.mix[stage] = 1.0
    if stage > 0:
        for s in range(stage):
            env.mix[s] = 0.4 / stage
        env.mix[stage] = 0.6
    obs = env.start()
    T = algo.T
    hist = {s: collections.deque(maxlen=max(400, min(3000, n))) for s in range(len(STAGES))}
    recent = collections.deque(maxlen=2000)
    curve = open(ctx.output("curve", "training.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "stage", "landed_stage", "landed_full", "miss_m", "touchdown_mps", "return", "steps_per_s"])
    t_start = time.time()
    last_replay, last_save, last_tel = 0.0, time.time(), 0.0
    rec, rec_ep = [], 0
    episodes, it, stop_reason = 0, 0, "time"
    best_full = -1.0
    whys = {}
    # checkpoints stay in the task's own folder; the Policy output is written once, at the end, with the best policy:
    # an output that is delivered as soon as it stops changing must not be an early checkpoint
    import os
    import shutil
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")

    def save(note):
        st = algo.state()
        st.update({"stage": stage, "landed_full": full_rate(), "fields": OBS, "act": ACT, "note": note, "boosters": n,
                   "device": label, "minutes": (time.time() - t_start) / 60.0, "format": "esasim.booster-policy/1"})
        torch.save(st, out_path)

    def full_rate():
        h = hist[len(STAGES) - 1]
        return sum(h) / len(h) if len(h) >= 100 else 0.0

    def send_replay(frames, res):
        nonlocal rec_ep
        rec_ep += 1
        for k, part, last in chunks(frames, 200):
            ctx.telemetry(ch_rocket, {"format": "esasim.rocket/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last,
                                       "fields": FIELDS, "frames": part, "ship": [0.0, 0.0], "result": res,
                                       "label": "Attempt %d · lesson %d of %d" % (episodes, stage + 1, len(STAGES))})

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            stop = False
            for m in ctx.messages(ch_ctl):
                d = m.get("data") if isinstance(m, dict) else None
                if isinstance(d, dict) and str(d.get("control", "")).lower() in ("finish now", "finish", "stop learning"):
                    stop = True
            if stop:
                stop_reason = "asked"
                break
            ob = torch.zeros(T, n, OBS, device=dev)
            ac = torch.zeros(T, n, ACT, device=dev)
            lps = torch.zeros(T, n, device=dev)
            vals = torch.zeros(T, n, device=dev)
            rews = torch.zeros(T, n, device=dev)
            dones = torch.zeros(T, n, dtype=torch.bool, device=dev)
            t0 = time.time()
            for t in range(T):
                a, lp, v = algo.act(obs)
                rec.append(frame_of(env.B, 0))
                nobs, r, d, info = env.step(a)
                ob[t], ac[t], lps[t], vals[t], rews[t], dones[t] = obs, a, lp, v, r, d
                if bool(d[0]):
                    res = "landed" if bool(info["landed"][0]) else "crashed on the deck" if bool(info["on_deck"][0]) else "in the sea" if bool(info["touched"][0]) else "lost"
                    if time.time() - last_replay > replay_every and len(rec) > 5:
                        last_replay = time.time()
                        send_replay(rec, {"text": res, "miss": round(float(info["miss"][0]), 1), "vz": round(float(info["vz"][0]), 2), "landed": bool(info["landed"][0])})
                    rec = []
                if bool(d.any()):
                    idx = d.nonzero(as_tuple=False).flatten()
                    L = info["landed"][idx].tolist()
                    S_ = info["stage"][idx].tolist()
                    M = info["miss"][idx].tolist()
                    V = info["vz"][idx].tolist()
                    Tc = info["touched"][idx].tolist()
                    R = info["ret"][idx].tolist()
                    for w_ in info["why"][idx].tolist():
                        whys[w_] = whys.get(w_, 0) + 1
                    for l, s, mm, vv, tc, rr in zip(L, S_, M, V, Tc, R):
                        hist[s].append(1.0 if l else 0.0)
                        recent.append((s, l, mm if tc else None, vv if tc else None, rr))
                    episodes += len(L)
                obs = nobs
            if len(rec) > 3000:
                rec = rec[-3000:]
            _, _, last_v = algo.act(obs)
            stats = algo.update((ob, ac, lps, vals, rews, dones), last_v)
            algo.norm.update(ob.reshape(-1, OBS))
            it += 1
            sps = T * n / max(1e-6, time.time() - t0)
            # curriculum: the next stage when this one lands often enough
            rate = sum(hist[stage]) / len(hist[stage]) if len(hist[stage]) >= hist[stage].maxlen // 2 else 0.0
            if stage < len(STAGES) - 1 and rate >= 0.5:
                stage += 1
                ctx.log("Lesson %d of %d after %d attempts: starts %s from %.1f to %.1f km." % (stage + 1, len(STAGES), episodes, "burning" if STAGES[stage][0] == "burning" else "falling", STAGES[stage][1] / 1e3, STAGES[stage][2] / 1e3))
                ctx.telemetry(p.get("events_channel") or "event", {"text": "Lesson %d of %d: starting %s from %.1f-%.1f km" % (stage + 1, len(STAGES), "mid-burn" if STAGES[stage][0] == "burning" else "falling", STAGES[stage][1] / 1e3, STAGES[stage][2] / 1e3)})
                env.make_pool(stage, pool, log=ctx.log)
                env.mix = [0.4 / stage if s < stage else 0.6 if s == stage else 0.0 for s in range(len(STAGES))]
            fr = full_rate()
            if stage == len(STAGES) - 1 and fr > best_full and len(hist[stage]) >= 200:
                best_full = fr
                save("best so far")
                last_save = time.time()
            if time.time() - last_save > 120 and best_full < 0:     # once there is a best, only a better one replaces it
                save("checkpoint")
                last_save = time.time()
            if time.time() - last_tel > 2.0:
                last_tel = time.time()
                rc = [x for x in recent if x[0] == stage]
                tch = [x for x in rc if x[2] is not None]
                miss = sorted(x[2] for x in tch)[len(tch) // 2] if tch else None
                vz = sorted(x[3] for x in tch)[len(tch) // 2] if tch else None
                ret = sum(x[4] for x in rc) / len(rc) if rc else 0.0
                row = {"iteration": it, "stage": stage + 1, "landed": round(100 * rate, 1), "landed_full": round(100 * fr, 1),
                       "miss_m": None if miss is None else round(miss, 1), "touchdown_mps": None if vz is None else round(-vz, 2),
                       "return": round(ret, 1), "attempts": episodes, "steps_per_s": round(sps), "minutes": round(elapsed / 60, 2), "std": round(stats["std"], 3)}
                tot = max(1, sum(whys.values()))
                row["outcomes"] = {k: round(100.0 * whys.get(i, 0) / tot) for i, k in ((1, "landed"), (2, "deck crash"), (3, "sea"), (4, "tumbled"), (5, "no fuel"), (6, "time"))}
                whys = {}
                ctx.telemetry(ch_train, row)
                cw.writerow([round(elapsed / 60, 2), it, stage + 1, round(rate, 4), round(fr, 4), row["miss_m"], row["touchdown_mps"], row["return"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "lesson %d of %d · %.0f%% land · %d attempts" % (stage + 1, len(STAGES), 100 * (fr if stage == len(STAGES) - 1 else rate), episodes))
            if stage == len(STAGES) - 1 and len(hist[stage]) >= hist[stage].maxlen and fr >= target:
                stop_reason = "target"
                break
    finally:
        curve.close()
    if best_full < 0 or full_rate() >= best_full:
        save("final")
    # each training adds its own file to the policies' dataset (when it was made and how often it lands), so every
    # policy ever trained stays there to choose from; Booster mission takes the best one by itself
    name = "booster_policy_%s_%02dpct.pt" % (time.strftime("%Y%m%d_%H%M"), int(round(100 * max(0.0, best_full, full_rate()))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s" % name)
    if stage < len(STAGES) - 1:
        ctx.log("Not every lesson done yet (lesson %d of %d): full flights are only counted in the last lesson, so 0%% of full flights "
                "is expected. Run this flow again: it goes on from here. The mission keeps flying the last policy that lands full flights." % (stage + 1, len(STAGES)))
    ctx.log("Done (%s): %d attempts in %.1f minutes; lesson %d of %d; %.0f%% of full flights land in the last %d." % (
        {"time": "time is up", "asked": "finished on request", "target": "target reached"}[stop_reason], episodes, (time.time() - t_start) / 60,
        stage + 1, len(STAGES), 100 * full_rate(), len(hist[len(STAGES) - 1])))
