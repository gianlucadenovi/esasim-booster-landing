<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures (a shot of the landing from the deck camera works well), then: ![Booster landing](pictures/banner.png) -->

# Booster landing

**A Falcon 9-class booster learns to land on a drone ship at sea by reinforcement learning, then flies a whole mission live in 3D.**

Thousands of boosters fly at once on your GPU and learn, lesson by lesson, to bring the first stage down onto a moving deck with one engine that cannot hover. Then a full mission: countdown, lift-off, max-Q, separation, the flip, the entry burn, and the landing flown by the policy it learned, with onboard cameras and a broadcast-style overlay.

![type](https://img.shields.io/badge/type-Aerospace-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

Landing an orbital booster is a hard control problem. The booster arrives at about 1.3 km/s, high in the atmosphere, steering with grid fins. Its single landing engine, even at its lowest throttle, pushes harder than the nearly empty booster weighs, so it cannot hover: the landing burn has to bring the speed to zero exactly at the deck, which is heaving, pitching and rolling with the swell.

Nobody writes the landing logic here. A neural network (the **policy**) decides 10 times a second the throttle, when to light the landing burn (it can light only once) and which way to lean; a flight computer turns the lean into engine gimbal, grid-fin and thruster commands. The policy learns by trial and error (PPO, proximal policy optimisation) over millions of attempts, flown thousands at a time on the GPU.

Everything you see is computed by the physics: the descent, the fins' forces at each speed and air density, the engine's thrust and the propellant it burns, the wind, and the touchdown itself, where the booster stands on four crushable legs on the moving deck, or topples if it came in too hard.

## Flows

- **1 · Learn to land**: thousands of boosters at once on the GPU learn to land, lesson by lesson. One attempt is replayed in 3D every 15 seconds, with charts of the landing rate, the miss from the centre, the touchdown speed and how attempts end. The policy goes to the **booster-policy** dataset when learning ends; each run goes on from the furthest policy so far.
- **2 · Launch and land**: press **Launch** for a whole mission at real time (about 8 minutes), flown by the newest policy that has finished its lessons. The main view follows the booster, the second stage, the ship or the pad; four more views are fixed cameras on the interstage, the engines, a grid fin and the ship's deck, plus a second-stage camera and a tracking camera.

## The physics

| Part | What is simulated |
|---|---|
| **Booster** | 25.6 t empty, 3.66 m wide, about 45 m long; centre of mass and inertia change as the propellant burns |
| **Engines** | nine Merlin-class engines for the ascent (845 kN each at sea level, 914 kN in vacuum); three relit for the entry burn, one for landing, throttling 40–100 %, ±5° gimbal, spool-up after ignition |
| **Aerodynamics** | standard atmosphere, drag (higher near Mach 1.25), body lift, four grid fins whose force grows with speed and air density |
| **Attitude** | the flight computer shares each turn between gimbal, grid fins and nitrogen thrusters by what each can give at that moment |
| **Weather** | wind growing with height, a jet stream near 11 km and gusts |
| **Drone ship** | a 90 m deck that heaves, pitches and rolls with the swell (about a metre of heave in a moderate sea) |
| **Touchdown** | a rigid body on four legs: each a stiff spring with a crushable core that yields and can fail, friction on the deck; the engine skirt and tank can hit the deck too |

Public, approximate figures only. The vehicle carries no logos, names or livery, and the drone ship is generic.

## The mission

| Event | Time | Height | Speed |
|---|---|---|---|
| Lift-off | T+0 | — | — |
| Max-Q (then throttle-down) | about T+1:02 | 11 km | Mach 1.5 |
| Main engine cut-off | about T+2:34 | 71 km | 2.4 km/s |
| Stage separation | T+2:37 | | |
| Second-stage ignition | T+2:41 | | |
| Booster's highest point | | about 119 km | |
| Entry burn (three engines) | | from about 56 km | down to 1.3 km/s |
| Landing policy takes over | | about 37 km | 1.3 km/s |
| Landing burn and touchdown | about T+8:00 | about 650 km downrange | under 2.5 m/s |
| Second-stage engine cut-off, in orbit | about T+8:30 | about 200 km | 7.8 km/s |

The countdown always takes 10 real seconds; the flight runs at real time unless you set a time warp.

## The lessons

Learning starts close to the deck and moves back, one lesson at a time. A lesson opens once half the attempts of the one before land.

1. In the middle of a landing burn, 30–400 m above the deck.
2. In the middle of a landing burn, 0.4–2.2 km up.
3. Falling from 2.5–5 km.
4. Falling from 5–12 km.
5. Falling from 12–22 km.
6. From the real hand-over after the entry burn: about 37 km up at 1.3 km/s.

Only lesson 6 counts **full flights**, so the full-flight landing rate stays at 0 % until then: that is expected. An attempt counts as landed when it touches the deck at under 2.5 m/s down and 1.5 m/s sideways, leaning under 8°.

## Where it runs

- **NVIDIA GPU**: the way to train. On an RTX 3070, a run from scratch reached lesson 6 after about 37 minutes and landed 77 % of full flights after an hour. Other work on the GPU (several open 3D views, games) slows it down.
- **CPU**: works, with at most 1,024 boosters at once; much slower, fine to try it out.
- Flow 2 runs on any machine: one booster, computed on the CPU.

## What you need

- A machine with the EsaSim worker (Project › Compute), with an NVIDIA GPU for training.
- Python packages, installed by the worker the first time: `torch` (with CUDA on NVIDIA machines). Nothing is downloaded from the internet and no account or key is needed.
- Somewhere for the policies: the **booster-policy** dataset, made by the first training run (see below).

## How to run it

**If you took this project from Demos**, first tell it where your policies live (a copy keeps no link to anyone else's storage):

1. Open **1 · Learn to land** under **Blueprint**, select the training block and set **Outputs go to** to a place in your workspace, with the dataset name `booster-policy`.
2. For the first run there is no policy yet: remove the **Policies so far** block (or its connection), and add it back after the run, set to the **booster-policy** dataset.
3. In **2 · Launch and land**, set the first block to the **booster-policy** dataset once it exists.

Then:

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn to land** (an hour by default; **Finish now** ends it early and still keeps the policy). Run it again to learn more: it goes on from the furthest policy so far.
3. Run **2 · Launch and land** and press **Launch** in its view. **Wind**, **Sea** and **Time warp** apply to the next launch. The run ends by itself 6 seconds after touchdown.

## Settings worth knowing

- **Learn to land**: **Boosters at once** (4,096 on a GPU), **Minutes at most** (60), **Stop at landing rate** (90 % of full flights), **Replay every** (15 s), **Start from scratch** (off: it goes on from the furthest policy).
- **Launch and land**: **Wind** at the sea (6 m/s) and **Jet stream** near 11 km (25 m/s), **Sea** from 0 calm to 1 rough (0.4), **Time warp** (1 = real time, up to 20), **Ship offset** (moves the drone ship from where the booster would fall unsteered), **Policy file** (empty: the newest that has finished its lessons), **Flights per run** (1).
- In the 3D view: the bar at the top (it shows when the mouse is over the view) switches cameras, turns the sound and the commentary on or off and picks the commentator's voice.

## What you get

- `booster_policy_<date>_<NN>pct.pt` in the **booster-policy** dataset: the policy (network, input scaling, the lesson it reached and its landing rate, NN % of full flights).
- **training.csv**: the learning curve, minute by minute (lesson, landing rates, miss, touchdown speed, steps per second).
- **flights.csv**: each mission's wind, sea, result, distance from the centre, touchdown speed and lean.
- Live in the views: the replayed attempts while learning; the whole mission with the telemetry overlay (speed, altitude, timeline, engines, G-force), sound and spoken commentary.

## Data and licences

- No dataset is read: every flight is simulated. The vehicle and engine figures are approximate public values.
- The Earth seen from orbit uses NASA's Blue Marble imagery (public domain).
- The code of the blocks is EsaSim's.

## Notes

- A policy is only as good as its lessons: one that never reached lesson 6 may still land, but the mission log warns you.
- The touchdown is decided by the contact physics, not by a rule: a landing at 3 m/s may stand, a hard one breaks a leg and topples.
- The second stage's ascent is computed in two dimensions for the view; the booster's flight from separation to touchdown is fully simulated.
- The 3D effects need EsaSim 0.12.69 or later.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (aerospace).

### Blueprints in this repository

- [1 · Learn to land](blueprints/1-learn-to-land.json) · 12 blocks · version 4
- [2 · Launch and land](blueprints/2-launch-and-land.json) · 11 blocks · version 3

### Workspace blocks it uses

- [Booster landing: train](blocks/ws-booster-landing-train) · version 3
- [Booster mission](blocks/ws-booster-mission) · version 6

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
