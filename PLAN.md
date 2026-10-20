# IRO 2026 Creative Category: Robot Build Plan

Working plan for the heritage storyteller robot. Compiled from the roadmap
(`IRO 2026 Creative Category Gold Medal Roadmap.docx`) and the planning
sessions of 19–20 Oct 2026. Each function lists the first step to take now and
a "done when" test that proves it works.

## Context

- **Competition:** IRO 2026 Creative Category, Austria. Theme: Cultural
  Heritage Service Robots. Sub-topic revealed on day 1.
- **Format:** 2 h production plan, 4 h build (on-site, over 2 days),
  3 min presentation + 2 min Q&A. Table about 180 x 60 cm.
- **Score:** Creativity 50, Technical 20, Completion 10, Presentation 20.
- **Concept:** a living storyteller for intangible heritage. It notices a
  visitor, turns to them, talks in BM, English or German, recognises heritage
  objects and performs a tradition (wayang kulit) with its arm.
- **Everything runs offline** (no Wi-Fi, no internet at the venue).

## Hardware

| Part | Job |
|---|---|
| Jetson Orin Nano, JetPack 6 (Ubuntu 22.04), ROS 2 Humble | Runs everything offline |
| ReSpeaker XVF3800 USB 4-Mic Array | Clean audio + direction of arrival (DoA) + echo cancellation |
| SO-ARM101, **follower arm only** (no leader) | Heritage performance, wrist camera close-ups |
| Global-shutter camera (on arm) | Object close-ups for few-shot recognition |
| Depth camera (low on body) | Sees children and wheelchair users |
| RPLidar | Visitor detection, artifact no-go zones |
| TETRIX + PRIZM + TorqueNADO + mecanum wheels | Drive base |
| SCD30 (optional) | CO2 / temperature / humidity for artifact care |
| Screen | RViz dashboard for judges |

## Build order

1. **Step 0** (blocks everything): Jetson + ROS 2 workspace.
2. **Tier 1** (must-have, in this order): base + E-stop, voice loop, turn to
   speaker, arm performance, few-shot recognition, backup mode, RViz dashboard.
3. **Gate:** the Tier 1 loop works 10 times in a row.
4. **Tier 2** (boosters, one at a time): LiDAR visitors, sensor fusion, no-go
   zones, visitor adaptation, YOLO-World, SCD30, sub-topic switcher.
5. **Feature freeze at end of November** (end of phase 3). After that, fixes only.

---

## Step 0: Jetson setup (start today)

- [ ] Flash JetPack 6 (Ubuntu 22.04)
- [ ] Install ROS 2 Humble
- [ ] Set max power mode (`sudo nvpmodel -m 0`, `sudo jetson_clocks`)
- [ ] Create the ROS 2 workspace in this repo (`ros2_ws/src/`)

**Done when:** `ros2 topic list` runs on the Jetson with Wi-Fi off.

---

## Tier 1 functions

### 1. Mecanum base + E-stop

- [ ] Wire a hardware E-stop that cuts motor power
- [ ] Drive the PRIZM from a laptop over USB serial with a simple text
      protocol, e.g. `vx vy wz\n`
- [ ] ROS 2 `base_node` on the Jetson: subscribes to `/cmd_vel`, talks to
      PRIZM over serial (the PRIZM cannot run ROS itself)
- [ ] Software deadman: stop if no `/cmd_vel` for 0.5 s

**Done when:** the base drives in all directions from the keyboard, and the
E-stop kills it instantly.

**Decide:** does the robot drive on the floor, or only rotate on the table?
Add an edge or cliff stop if it moves on the table.

### 2. Voice loop (STT → LLM → TTS)

Benchmark each piece on its own on the Jetson first.

- [ ] STT: whisper.cpp or faster-whisper (Whisper supports Malay)
- [ ] LLM: llama.cpp with a 1–3B model; test BM, English and German answers
- [ ] TTS: Piper for English and German
- [ ] TTS for Malay: Piper probably has no Malay voice; test Meta MMS-TTS
      (Malay) and espeak-ng as fallbacks
- [ ] Fill in the benchmark table below; have a native BM speaker rate quality

| Piece | Model | EN latency | DE latency | BM latency | BM quality (1–5) |
|---|---|---|---|---|---|
| STT | | | | | |
| LLM | | | | | |
| TTS | | | | | |

**Done when:** a full spoken question → spoken answer in under about 3 s in
all three languages.

**Watch:** Malay is the go/no-go check for model choice. This is the most
important result of week 1.

### 3. Turn to speaker (ReSpeaker XVF3800)

The XVF3800 does DoA, beamforming, noise suppression and echo cancellation on
the board. The Jetson gets cleaned-up audio plus an angle. The "5 m far-field"
range is a quiet-room number; expect less in a noisy hall.

- [ ] Plug into the Jetson; check it appears as a USB sound card
      (`arecord -l`), record 10 s and play back
- [ ] Check the firmware version and Seeed's update instructions (different
      firmware builds expose different channels and controls)
- [ ] Read the DoA angle with Seeed's host control tool / Python example.
      Clap or speak from 0°, 90°, 180°, 270° and record the reported angle
- [ ] Route the robot's speaker through the array's 3.5 mm output so echo
      cancellation works (robot does not hear itself; visitors can interrupt)
- [ ] Test Whisper in noise: play crowd noise, compare accuracy in all three
      languages at 1 m and 3 m
- [ ] ROS 2 `mic_node`: publishes DoA angle + "voice active" flag. Turn only
      when voice is active and the angle has been steady for about 0.5 s

**Done when:** the robot turns toward a speaker and ignores motor noise and
its own voice.

**Mounting:**
- Flat and level on top of the robot, array 0° marked and aligned with the
  robot's front
- Away from motors and arm servos, on soft mounting
- High, above the arm's swing, so nothing blocks it

**Watch:**
- The angle is only meaningful while someone speaks; gate it on voice
  activity and smooth with a median of the last few readings
- Table and walls reflect sound: mic gives the rough turn, the camera centres
  the person
- With several voices it follows the loudest; only one person talks at a time
  in the demo

### 4. Arm heritage performance (SO-ARM101 follower only)

No leader arm, so motions are taught by hand and replayed as keyframes. No
teleop recording and no learned policies (ACT etc.).

- [ ] Calibrate the follower (LeRobot calibration or own homing routine);
      save offsets and practise recalibrating, since this happens on-site
- [ ] Teach-by-hand script: torque off, move the arm to a pose, press a key,
      read and save the position of all 6 STS3215 servos
- [ ] Store routines as named keyframe lists in YAML, e.g.
      `wayang_greet: [pose1, pose2, ...]` with a duration per step
- [ ] Replay with smooth interpolation between keyframes, speed limit and
      joint limits
- [ ] Record one routine: wayang kulit puppet, about 15 s (greet, bow,
      "speaking" sway timed to TTS)

The arm's three jobs, all using the same keyframe player:

| Job | Approach | Done when |
|---|---|---|
| Heritage performance | Gripper holds the puppet's main rod; one ~15 s routine | Runs 10 times in a row, no drift, puppet never slips |
| Look at an object (wrist camera) | 2–3 fixed "inspect" poses over a known table spot; no inverse kinematics | Object centred in the camera at each pose |
| Idle / rest | Low rest pose, torque off or reduced | Servos stay cool through a 30-minute session |

**ROS 2 `arm_node`:** a `perform` action (routine name), a `go_to_pose`
service, and `/joint_states` with the SO-101 URDF so the arm shows in RViz.

**Watch:**
- Servo heat: STS3215s get hot holding poses, especially the shoulder with a
  puppet. Rest with torque off between performances
- Puppet grip: test the gripper on the real puppet early; a rod holder may be
  needed (and would also be built on-site)
- Table collisions: per-joint limits so a bad keyframe cannot hit the table or
  the robot body
- Time the calibration; it counts against the 4 h build

### 5. Few-shot object recognition

No training step: embeddings + nearest neighbour.

- [ ] Choose 3–4 heritage objects small enough for the table (e.g. wayang
      puppet, batik cloth, keris replica)
- [ ] Take frames from a 10 s video of each object; compute embeddings
      (DINOv2 or CLIP); store them
- [ ] Match new images to the nearest stored embedding; below a threshold,
      answer "I'm not sure"

**Done when:** shown 3 objects, it names each correctly from a new angle, and
says "not sure" for an unknown object.

**Watch:** film objects under the venue's lights on competition day.

### 6. Backup mode

- [ ] Record or generate audio for 10 ready-made answers
- [ ] Wire one physical button that plays them

**Done when:** a button press plays an answer even with the LLM process killed.

Cheapest Tier 1 item; do it early, it protects every demo after that.

### 7. RViz dashboard

- [ ] One RViz config: camera image, LiDAR scan, arm model, text panel for
      "heard / thinking / saying"

**Done when:** it opens with one command and updates live. Grows as each node
is added; no polishing yet.

---

## Tier 2 functions (research only for now, no building)

| Function | First step now |
|---|---|
| LiDAR visitor detection | Run `rplidar_ros`; record a bag of someone walking toward the robot. Later: learn the empty room, flag what moves |
| Sensor fusion | Write down the message flow (LiDAR → camera → mic → turn) so Tier 1 nodes publish the right topics from the start |
| No-go zones | Choose between a simple polygon check in an own node and the Nav2 keepout filter (polygon check is probably enough for a table) |
| Visitor adaptation | Record depth frames of a seated and a standing person; dataset for a simple height rule |
| YOLO-World | Check it runs at a usable FPS on the Orin Nano; no integration yet |
| SCD30 | Read values once over I2C |
| Sub-topic switcher | Write 5 practice sub-topics and a system prompt for each |

---

## ROS 2 packages (planned)

| Package | Publishes / provides | Subscribes / uses |
|---|---|---|
| `base_node` | — | `/cmd_vel` |
| `mic_node` | DoA angle, voice-active flag, audio | XVF3800 |
| `voice_node` | heard / thinking / saying text | audio from `mic_node` |
| `arm_node` | `/joint_states`, `perform` action, `go_to_pose` service | routines YAML |
| `vision_node` | recognised object + confidence | wrist camera |
| `behavior_node` | `/cmd_vel`, arm and voice triggers | all of the above |
| `bringup` | one launch file for everything | — |

---

## Team split (this week)

- **Software:** Jetson setup, then voice benchmarks (biggest unknown)
- **Hardware:** base + E-stop + serial protocol, then arm calibration and the
  first taught routine
- **Story and presentation:** choose the tradition and objects, write backup
  answers and the sub-topic prompts, find a BM native speaker to rate voice
  quality

## This week's key results

- [ ] Voice benchmark table filled in, especially Malay
- [ ] Base drives with a working E-stop
- [ ] Arm replays one taught routine
- [ ] XVF3800 angle measured at 4 directions
- [ ] Heritage tradition and 3–4 objects chosen

## Schedule

Weeks count from 20 Oct 2026 and assume the competition is around the end of
December.

| Phase | Dates | Goal | Gate to pass |
|---|---|---|---|
| 1. Foundation | 20 Oct – 2 Nov | Step 0, base, voice benchmarks, choices made | Voice reply time measured |
| 2. Core loop (Tier 1) | 3 – 16 Nov | All 7 Tier 1 functions | Tier 1 loop works 10 times in a row |
| 3. Gold boosters (Tier 2) | 17 – 30 Nov | Tier 2, one feature at a time | Feature freeze |
| 4. Mock rounds | 1 – 14 Dec | 5+ full rounds: 2 h plan, 4 h build, 3 min talk; Wi-Fi off | Full build from loose parts under 3.5 h |
| 5. Freeze and pack | 15 – 28 Dec | Fixes only, final mock with teachers as judges, packing list | — |

Also schedule one mock build in phase 2 to measure the real build time early.

## Risks and backups

| Risk | Backup |
|---|---|
| Build runs over 4 h | Mock builds until under 3.5 h; drop Tier 2 first |
| LLM slow or freezes | Smaller model, shorter replies, backup-mode button |
| Noisy hall confuses the mic | Mic for rough turn, camera centres the person; echo cancellation via XVF3800 output |
| Malay TTS poor or slow | MMS-TTS or espeak-ng fallback; pre-recorded BM lines for key moments |
| Venue lighting fools recognition | Film objects at the venue; "not sure" threshold |
| Arm calibration fails on-site | Practise calibration in every mock build |
| Servo overheating | Rest pose with torque off between performances |
| Robot falls off the table | Floor driving or rotate-only on table; edge stop |
| Battery runs low | Full charge before the slot; test a full-day run |
| Demo relies on a Tier 2 feature that was cut | Keep a Tier 1-only version of the presentation script |
