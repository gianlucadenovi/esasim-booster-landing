# Booster landing

A Falcon 9-class first stage learns to land on a drone ship at sea by reinforcement learning, then flies a whole mission live in 3D: launch, separation, flip, entry burn, and the landing flown by the policy it learned.

The physics uses public, approximate figures: 25.6 t empty, nine Merlin-class engines, a single engine relit for landing that throttles down to 40 % (it cannot hover, so the landing burn has to reach zero speed exactly at the deck), ±5° gimbal, grid fins, nitrogen thrusters, a standard atmosphere, wind that grows with height with a jet stream and gusts, and a deck that heaves, pitches and rolls. The vehicle carries no logos, names or livery, and the drone ship is generic.

## 1 · Learn to land

Run it on **Gianluca De Novi PC** (RTX 3070) with the accelerator set to NVIDIA GPU. Thousands of boosters fly at once. The policy decides 10 times a second: the throttle, when to light the landing burn (it can light only once) and which way to lean. The flight computer turns the lean into gimbal, grid-fin and thruster commands.

Six lessons, each opening once half the attempts of the one before land:

1. In the middle of a landing burn, 30–400 m above the deck.
2. In the middle of a landing burn, 0.4–2.2 km up.
3. Falling from 2.5–5 km.
4. Falling from 5–12 km.
5. Falling from 12–22 km.
6. From the real hand-over after the entry burn: about 33 km up at 1.3 km/s.

An attempt counts as landed when it touches the deck at under 2.5 m/s down and 1.5 m/s sideways, leaning under 8°. One attempt is replayed in 3D every 15 seconds; the charts show the landing rate, the miss distance, the touchdown speed and how attempts end.

The policy is saved every 2 minutes to the **booster-policy** dataset. Once there is a best policy on full flights, only a better one replaces it. **Finish now** ends the run early.

## 2 · Launch and land

1. Choose the **booster-policy** dataset in the first block. It exists after flow 1 has run once.
2. Run the flow.
3. Press **Launch**. Wind, sea state and time warp apply to the next launch.

A flight takes about 9 minutes at real time. The main view's bar orbits the booster, second stage, ship or pad, or switches to a tracking or free camera. The other four views are fixed cameras: interstage, engines, grid fin and the ship's deck.

The 3D effects need EsaSim 0.12.58 or later.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (aerospace).

### Blueprints in this repository

- [1 · Learn to land](blueprints/1-learn-to-land.json) · 8 blocks · version 1
- [2 · Launch and land](blueprints/2-launch-and-land.json) · 9 blocks

### Workspace blocks it uses

- [Booster landing: train](blocks/ws-booster-landing-train) · version 1
- [Booster mission](blocks/ws-booster-mission) · version 1

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
