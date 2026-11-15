# SO-101 arm: measured limits, how picking works, and what to build next

Measured on the robot on 15 Nov 2026 (Jetson Orin NX, arm base 13.2 cm above the floor).
Every number here comes from a recorded run in `~/iro-runs` or a test in that session.
The Jetson was in **15W power mode** during all timing tests (GPU at half speed, 4 of 8 CPU
cores), so the times below are slower than the robot will be at 40W.

## 1. Limits of the arm

### Reach (fingertips pointing down)

Distance from the arm base to the fingertips, by fingertip height and by how far the gripper
leans outward. From the arm's model; the outer edge at 1.5 cm height was checked on the real arm
(28 to 40 cm: reached every point within 2 mm by its own encoders).

| Fingertip height | Straight down | Lean 15° | Lean 30° | Lean 45° |
|---|---|---|---|---|
| 1.2 cm (grasp height) | 3–30.8 cm | to 34.8 cm | to 38.0 cm | to 40.0 cm |
| 5 cm | 5–31.5 cm | to 35.8 cm | to 39.3 cm | to 41.9 cm |
| 10 cm | 8–31.9 cm | to 36.1 cm | to 39.9 cm | to 43.0 cm |
| 15 cm (look height) | 8.5–30.9 cm | to 35.4 cm | to 39.6 cm | to 43.2 cm |
| 20 cm | 12.5–27.5 cm | to 33.5 cm | to 38.0 cm | to 42.4 cm |
| 25 cm | not possible | 17–29.2 cm | to 35.7 cm | to 40.5 cm |

- Highest point with the gripper straight down: 22 cm above the floor.
- The base turns ±110°.
- The OAK-D sees the floor from 18 cm in front of the arm base, so the area where the scout
  can guide a pick is 18 to about 40 cm.
- The inner numbers (3–8 cm) ignore the arm hitting its own base; treat anything under about
  12 cm as untested.

### Gripper

- Widest usable opening: **71.4 mm** (at 46%). Beyond that the moving jaw swings up and away,
  so its tip no longer reaches down to a low object.
- With 8 mm of room on each side for the blind last centimetres, the widest object it takes
  from above is about **55 mm** across.
- The tape lying flat (70 mm across, 2.5 cm tall) cannot be gripped from above. Standing on
  its rim (25 mm across) it can.
- A hard object squeezes the gripper's own flex by about 0.2–0.3 mm per % of load.

### Speed and accuracy

| What | Value |
|---|---|
| Joint speed limit in open air | 1.5 rad/s (1.6 tested: no hotter, same accuracy) |
| An 8–10 cm move through open air | about 0.7 s |
| Settling after a move | 0.1–0.5 s (was 0.5–1 s of fixed waits) |
| Descent from 15 cm to the grasp height | 4–5 s (was 22.5 s) |
| Coming back to the same look pose from 8 directions | ±3 mm, worst 5.9 mm in 12 returns |
| Servo read of all six positions | 1 ms |
| Hottest servo after about an hour of tests | 46 °C |

### Things that limit accuracy

- **Soft servos.** At the standard stiffness (P 16) gravity leaves shoulder and elbow up to
  38 and 24 counts short. P 24 on those two cuts that to 24 and 15 with no shaking. P 32 made
  the wrist shake, so the wrist stays at 16.
- **Wobble the encoders cannot see.** After a quick move the plastic links and the wrist
  camera keep swinging for a moment while the encoders already read "still". A picture taken
  then was 28 px (about 1 cm) off. Pictures are now taken only when two frames in a row agree.
- **The wrist camera looks forward at a slant.** One metre of gripper movement moves the
  picture 4941 px sideways but only 2107 px along the view. A taller object therefore appears
  shifted, and a narrow object seen end-on looks wider or narrower than it is.

## 2. How a pick works now

1. **Scout (OAK-D, before the arm moves).** Depth points are turned into the arm's
   coordinates. Everything standing more than 1 cm above the floor is an object. For each one:
   its width across the camera's line of sight, its height, and where its front face is.
2. **Completing the hidden side.** The OAK-D sits low, so it only sees an object's near face.
   With the object's three known sizes (`objects/NAME/size.yaml`) it works out which way the
   object lies and where its true centre is.
3. **Straight to the look pose.** The arm goes directly to the pose that puts the object on
   the taught grasp mark in the wrist picture, with the jaws already turned across the true
   narrow side.
4. **Wrist camera confirms and fine-centres**, usually in one step, using a camera model
   measured once (`iro pick calibrate NAME`).
5. **Down in one smooth movement**: fast above the object, slower near it, stopping if a jaw
   lands on something.
6. **Close until first contact, lift, squeeze test.** The OAK-D then checks the floor is empty
   where the object stood.
7. **The pick teaches the scout.** Where the scout said the object was and where it really
   was are stored as a pair; later scout positions are corrected with them.

If the OAK-D is unavailable, or the wrist camera does not confirm what the scout found, the
old search with the wrist camera runs instead (`IRO_NO_SCOUT=1 iro pick NAME` forces it).

### Measured effect (same tape, standing)

| | Wrist camera only | With the scout |
|---|---|---|
| Whole pick including start-up | 61 s | 47–49 s |
| Search and jaw turning | 15–25 s | none |
| Centring steps | 7 | 1–3 (1 once the scout's correction was learnt) |
| First aim, distance from the grasp mark | not applicable | 84–114 px, then 12–33 px after 3 learnt pairs |
| A standing tape seen end-on | jaws turned the wrong way, tape knocked over | correct direction |
| A tape lying flat (too wide) | went down on it, let go afterwards | refused before moving |

Start-up (loading the models and starting the OAK-D) is 16–19 s of every pick in 15W mode.

## 3. The maths, briefly

- **Smooth moves (minimum jerk).** Position along a move follows
  `s(t) = 10t³ − 15t⁴ + 6t⁵` for t from 0 to 1. Speed and acceleration are zero at both ends,
  so the arm starts and lands softly. The top speed is 1.875 × distance / duration.
- **Inverse kinematics.** Find joint angles q that minimise
  `|target − tip(q)|² + (0.1 · |down − tool_axis(q)|)²` inside the joint limits. Solved with
  Levenberg–Marquardt: `step = (JᵀJ + λI)⁻¹ Jᵀe`, where λ shrinks after a good step and grows
  after a bad one. 6.5 ms for a reachable target (the previous fixed-step solver took about
  110 ms and stopped 0.5 mm and 0.5° short).
- **Gravity sag.** The goal offset that makes joint j arrive is modelled as
  `a_j · g_j(q) + b_j · s_j`, where g_j is how much the tip height changes per radian of that
  joint (gravity's lever) and s_j is the direction the joint came from. Learnt from every
  settled move, used after 8 moves.
- **Wrist picture model.** Picture shift = `Jg · (gripper movement in its own frame)`, with
  Jg a 2×2 matrix measured on the object. In the arm's frame `J = scale · Jg · R(−heading)`.
  A centring step is `−J⁻¹ · (object position − grasp mark)`. The scale is corrected from
  what each step really did, because a taller object is nearer the camera and moves more.
- **Scout correction.** A rigid shift and turn fitted to the stored pairs (2D least squares).
  With three pairs it found a 7.3° turn and a 3.2 cm shift, which brought the pairs from
  3–4.5 cm apart to 4–8 mm.
- **Blockage rule.** Blocked means: over the last 0.25 s the command went down at least 6 mm,
  the tip less than 1.5 mm, and the tip is more than 5 mm above its command. A rule based
  only on "the tip is X mm behind" gave false alarms at long reach, where the joints lag
  8–13 mm in free air.

## 4. New commands

| Command | What it does |
|---|---|
| `iro pick NAME` | Scout-guided pick (falls back to the wrist-camera search) |
| `iro pick NAME --drop` | Pick, then set it back down |
| `iro pick NAME --to=X,Y` | Pick and carry it to X, Y (cm from the arm base) |
| `iro pick scout` | What the OAK-D sees on the floor: positions and sizes |
| `iro pick calibrate NAME` | Measure the wrist camera model on NAME lying still (20 s) |
| `tools/pick_loop.sh NAME X,Y X,Y ...` | The arm moves the object from spot to spot and re-picks it; one summary line per pick |

## 5. Not yet verified

- The gentler put-down (all the way down, jaws fully open before rising, slow first 3 cm).
  The earlier, faster version let a tape standing on its rim fall over.
- The pick-and-place loop past its first pick: the tape was lying flat each time it was run.
- Everything at 40W power mode.
- Objects other than the tape.

## 6. Ideas to build next, in order of expected gain

1. **40W power mode** (`sudo nvpmodel -m 4`). Outlining the object takes 0.77 s per look at
   15W and start-up 16–19 s; both are GPU and CPU bound.
2. **A pick service that stays running**, keeping the models and cameras warm, so a pick
   starts in under a second instead of after 16 s. It must release the cameras when idle so
   other tools can use them.
3. **Track the object between looks by template shift** and outline it only at the first and
   last look. Cuts each extra centring step from about 1.1 s to about 0.35 s.
4. **Wide grasp by leaning.** With the jaws closing along the line from the base to the
   object, leaning the gripper lowers one jaw tip against the other by `gap · sin(lean)`.
   That could cancel the moving jaw's rise beyond 46% opening and allow objects wider than
   55 mm, such as the tape lying flat. Needs the jaw geometry from the URDF and careful tests.
5. **Grasp height and mark from the object's measured height**, so one teaching covers an
   object standing or lying. The camera model already has the vertical term (1093 px per
   metre of height).
6. **Re-run `iro efad calibrate`** when the floor is clear. The scout's learnt correction
   hides a 7° shift of the OAK-D mount that the calibration itself should fix.
7. **Turn the OAK-D's infrared dots off after scouting.** They show up in the wrist camera's
   picture.
