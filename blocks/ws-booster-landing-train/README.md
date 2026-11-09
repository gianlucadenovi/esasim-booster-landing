# Booster landing: train

A Falcon 9-class booster learns to land on a drone ship by reinforcement learning (PPO), thousands of boosters at once on the GPU.

Run it on the machine with the NVIDIA GPU (Accelerator: NVIDIA GPU). Each attempt flies a booster with real physics (gravity, standard atmosphere, drag and lift of body and grid fins, one engine throttling 40-100% that lights once, gimbal, nitrogen thrusters, wind with a jet stream and gusts, a deck that heaves and rolls) and the policy decides 10 times a second: throttle, when to light the landing burn, and which way to lean. Six lessons: two starting mid-burn near the deck, then falling from 2.5-5, 5-12, 12-22 km and from the real hand-over after the entry burn (about 33 km, 1.3 km/s); a lesson opens when half the attempts of the one before land. Landed means on the deck under 2.5 m/s down, 1.5 m/s sideways, 8 degrees of lean. Telemetry: channel train (lesson, landed %, landed_full % from the hand-over, miss_m, touchdown_mps, attempts, steps_per_s, minutes, outcomes.*), channel rocket (one attempt replayed in the 3D view every Replay every seconds, esasim.rocket/1) and event (new lessons). The policy is written when learning ends (the time limit, the target, or the button Finish now on the control channel), with the best one seen on full flights, under its own name: booster_policy_<date>_<NN>pct.pt (NN: the share of full flights that landed). Send it to a dataset: each training adds one policy, and Booster mission, wired to that dataset, takes the best by itself. Wire an earlier policy to Continue from to pick up where it stopped.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-booster-landing-train` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Continue from | Files (pt) | A booster_policy*.pt from an earlier run, to keep training it Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | booster_policy_<date>_<NN>pct.pt: the landing policy (network, input scaling, lesson reached, landing rate), written when learning ends |
| `curve` Learning curve | Table (csv) | training.csv: minutes, lesson, landing rates, miss, touchdown speed, steps per second |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `accelerator` Accelerator | select | `"auto"` | auto takes the NVIDIA GPU when this PyTorch can use it |
| `boosters` Boosters at once | number | `4096` | Parallel attempts on the GPU (the CPU uses at most 1024) |
| `minutes` Minutes at most | number | `60` |  |
| `target` Stop at landing rate | number | `0.9` | Share of full flights (from the hand-over) that land, to stop early |
| `replay_every` Replay every (s) | number | `15` | How often one attempt is sent to the 3D view |
| `seed` Seed | number | `1` |  |
| `channel` Training channel | text | `"train"` |  |
| `rocket_channel` Flight channel | text | `"rocket"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
