# Booster mission

Launch and land a Falcon 9-class booster live: the precomputed launch, flip and entry burn, then the learned policy lands it on the drone ship.

Wire the policies of Booster landing: train (its dataset: each training adds a booster_policy_<date>_<NN>pct.pt) to Policies and the block's telemetry, through a Telemetry block, to one or more 3D views. The block reads every policy, lists them in the log newest first with how often each landed in training, and flies the newest that has finished its lessons (or the one Policy file names; with none finished, the newest). Press Launch (control channel, from the 3D view's or a Controls view's buttons): a 10-second countdown in real seconds, lift-off on nine engines, max-Q (about T+1:02, 11 km), main engine cut-off (about T+2:34, 71 km, 2.4 km/s) and separation 3 s later; the second stage lights its vacuum engine and burns to orbit (about 7.8 km/s near 200 km, cut-off near T+8:30). The booster flips on nitrogen thrusters, grid fins out, coasts up to about 120 km, makes the three-engine entry burn, then the learned policy flies the grid-fin glide and the single-engine landing burn to the drone ship about 650 km downrange, which waits where the booster would fall without steering (Ship offset moves it). Touchdown is simulated as a collision: from the first foot on the deck, the booster is a rigid body on four legs (stiff springs with crushable cores that yield and can fail, friction on the deck; the engine skirt and the tank can hit it too) on the heaving, pitching, rolling deck, and that decides whether it stands or topples. The flight ends 6 s after touchdown; after Flights per run flights (1 by default) the run ends by itself. Real time by default, Time warp up to 20x. Wind (m/s at the sea, growing with height), Jet stream (m/s near 11 km) and Sea (0 calm to 1 rough: the deck heaves, pitches and rolls) take effect at the next Launch. Channel rocket carries the flight (esasim.rocket/1: 20 frames a second, plus alt_km, speed_kmh, throttle, prop_t, downrange_km, q_kpa, to_ship_km, phase, clock for gauges); channel event the milestones; flights.csv the results.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-booster-mission` · version 5 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt) | The booster_policy*.pt files from Booster landing: train (wire its dataset: every training adds one); the newest that has finished its lessons is flown unless Policy file names one |

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
| `warp` Time warp | number | `1` | 1 = real time (a flight takes about 8 minutes); the countdown always takes 10 real seconds |
| `offset` Ship offset (m) | number | `0` | Moves the drone ship downrange from where the booster would fall unsteered |
| `policy_file` Policy file | text | `""` | Empty: the newest policy that has finished its lessons (the date and time in its name). Or part of a file name (such as 20261109_1203) to fly that one |
| `flights_per_run` Flights per run | number | `1` | The run ends by itself after this many flights (6 s after the last touchdown); 0: it waits for Launch until stopped |
| `auto_launch` Launch at once | bool | `false` | Off: wait for the Launch button |
| `channel` Flight channel | text | `"rocket"` |  |
| `events_channel` Events channel | text | `"event"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
