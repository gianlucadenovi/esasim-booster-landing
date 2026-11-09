# Booster mission

Launch and land a Falcon 9-class booster live: the precomputed launch, flip and entry burn, then the learned policy lands it on the drone ship.

Wire the policies of Booster landing: train (its dataset: each training adds a booster_policy_<date>_<NN>pct.pt) to Policies and the block's telemetry, through a Telemetry block, to one or more 3D views. The block reads every policy, lists them in the log with how often each landed in training, and flies the best (or the one Policy file names). Press Launch (control channel, from the 3D view's or a Controls view's buttons): a 10-second countdown in real seconds, lift-off on nine engines, max-Q, main engine cut-off and separation (the second stage lights its vacuum engine), the flip on nitrogen thrusters, grid fins out, a coast over about 160 km, the three-engine entry burn, then the learned policy flies the grid-fin glide and the single-engine landing burn to the drone ship about 700 km downrange, which waits where the booster would fall without steering (Ship offset moves it). Real time by default, Time warp up to 20x. Wind (m/s at the sea, growing with height), Jet stream (m/s near 11 km) and Sea (0 calm to 1 rough: the deck heaves, pitches and rolls) take effect at the next Launch. Channel rocket carries the flight (esasim.rocket/1: 20 frames a second, plus alt_km, speed_kmh, throttle, prop_t, downrange_km, q_kpa, to_ship_km, phase, clock for gauges); channel event the milestones; flights.csv the results. The run waits for the next Launch until it is stopped.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-booster-mission` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt) | The booster_policy*.pt files from Booster landing: train (wire its dataset: every training adds one); the best is flown unless Policy file names one |

## Outputs

| Port | Type | Description |
|---|---|---|
| `flights` Flights | Table (csv) | flights.csv: each flight's wind, sea, result, distance from the centre, touchdown speed and lean |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `wind` Wind (m/s) | number | `6` | At the sea, growing with height; with gusts |
| `jet` Jet stream (m/s) | number | `25` | Near 11 km |
| `sea` Sea (0-1) | number | `0.4` | 0 calm, 1 rough: how much the deck heaves, pitches and rolls |
| `warp` Time warp | number | `1` | 1 = real time (a flight takes about 9 minutes); the countdown always takes 10 real seconds |
| `offset` Ship offset (m) | number | `0` | Moves the drone ship downrange from where the booster would fall unsteered |
| `policy_file` Policy file | text | `""` | Empty: the policy that landed most often in training. Or part of a file name (such as 20261109_1203) to fly that one |
| `auto_launch` Launch at once | bool | `false` | Off: wait for the Launch button |
| `channel` Flight channel | text | `"rocket"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
