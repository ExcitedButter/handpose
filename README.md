# EgoForce hand keypoint algorithm

Offline 2D/3D hand keypoint estimation for the EgoForce head-camera video.
Both gloved hands are estimated from the left camera view and written as one
`.npz` per recording, aligned frame by frame with `head.mp4` and with the
`tactile.npz` of the same recording.

Part of the [EgoForce](https://huggingface.co/datasets/zhicao/EgoForce) release accompanying
[FACT: A Multimodal Force-Action-Track Foundation Model for Human-to-Robot Dexterity](https://arxiv.org/abs/2610.19155).

Zhi Cao<sup>1,2*</sup>, Yining Hong<sup>2*</sup>, Jiaolong Yang<sup>3</sup>, Yejin Choi<sup>2</sup>, Li Fei-Fei<sup>2</sup>, Jiajun Wu<sup>2&dagger;</sup>, Huang Huang<sup>2&dagger;</sup>

<sup>1</sup>University of Michigan &nbsp; <sup>2</sup>Stanford University &nbsp; <sup>3</sup>Microsoft Research &nbsp; <sup>*</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Equal advising

## How it works

1. **Detection.** YOLO-World (text prompts "white glove", "hand") runs on
   the full left view of every pose frame (`--pose-hz`). When it returns fewer
   than two confident boxes, the left 60 % and the right 60 % of the image are
   searched again at higher effective resolution. A box that covers two other
   boxes (both hands merged) is dropped. Each candidate also gets the share of
   white pixels inside it, since the EgoForce gloves are white, and a shoe
   score: the same YOLO-World weights are queried once more on the full view
   with the prompts "white glove", "hand", "shoe", "sneaker", and the
   candidate gets the shoe score minus the glove score of the overlapping
   prediction (white sneakers otherwise pass as gloves).
2. **Hand labels** (`labeling.py`, offline). All candidates of the recording
   are labeled at once by a Viterbi search over per-frame assignments
   (left box, right box), each possibly absent. The path cost rewards
   confident boxes; penalizes boxes with little white (white share below
   35%: sleeves, wristbands, knee pads), boxes that score clearly higher as a shoe than as a glove,
   left/right crossing, a hand far on the opposite side of the
   image, abrupt motion or size changes between frames, and every appearance
   or disappearance of a hand. Because the search sees the whole recording, a
   hand that re-enters the view is labeled from the past and the future,
   not only from the frame where it reappears.
3. **Pose** (`pose_select.py`). AnyHand-WiLoR regresses MANO parameters, 21
   3D joints and their 2D projections for each labeled hand box at
   `--pose-hz`. The network sometimes returns the mirror-image solution (palm
   and back swapped, thumb on the wrong side), mostly for right hands seen
   edge-on or as a fist. Each hand therefore gets four pose hypotheses: the
   labeled box, a box refit to the predicted keypoints, the refit box on an
   image rotated by 25 degrees, and the mirror-image alternative (opposite
   handedness flag, reflected across its own palm plane). A Viterbi search
   over the recording picks one hypothesis per sample: it rewards skeletons
   that lie on the white glove, meshes that cover the glove around the hand,
   and agreement with the other hypotheses; it prefers the back of the hand
   toward the head camera (graded, so edge-on hands are not forced either
   way), and penalizes changes of hand shape, wrist jumps and clear palm flips
   between samples.
4. **Stabilization** (`stabilize.py`, offline, zero phase). Drop detections
   with score below 0.05; reject poses whose wrist lies outside the hand's own
   box (more than 0.6 box diagonals from its center; a box that touches the
   image border is first extended past it by its own size, since a hand cut
   off by the frame has its wrist outside the image); split at holes
   longer than 1.2 s; repair spikes with a Hampel filter; smooth with a
   centered Savitzky-Golay filter (0.7 s window, order 2); interpolate onto
   every video frame, never across a hole longer than 0.3 s.

## 1. Install

Python 3.10 or newer.

```bash
cd handpose_algorithm
pip install -r requirements.txt
```

Use `opencv-contrib-python`, not `opencv-python`: the lower-level `run.py`
uses CSRT tracking from the contrib build. Install the PyTorch build that
matches your hardware first if you want GPU inference. On a GPU with `--fp16`
we measured about 15 video frames/s per process (four pose hypotheses per
hand); CPU works but is much slower.

## 2. Model weights

The weights are not stored in this GitHub repository. Download them from the
[EgoForce dataset on Hugging Face](https://huggingface.co/datasets/zhicao/EgoForce/tree/main/handpose_algorithm/weights)
into `weights/` (about 2.9 GB, everything except MANO):

```bash
pip install -U huggingface_hub
hf download zhicao/EgoForce --repo-type dataset \
    --include "handpose_algorithm/weights/*" --local-dir hf_egoforce
cp -r hf_egoforce/handpose_algorithm/weights/. weights/
```

The resulting layout:

```text
weights/
  wilor/anyhand_wilor_nomano.ckpt          AnyHand-WiLoR (MANO tensors removed)
  wilor/anyhand_model_config_wilor.yaml
  yolo_world/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth
  yolo_world/clip-vit-base-patch32/vocab.json
  yolo_world/clip-vit-base-patch32/merges.txt
  mano/                                    you add MANO_RIGHT.pt here (step 3)
```

`ASSETS.json` lists the SHA-256 of every file; `python check_weights.py`
verifies them. To keep the weights elsewhere,
pass `--weights /path/to/weights` or set `HANDPOSE_WEIGHTS` to a directory
with the same layout.

## 3. MANO setup (required, one time)

The MANO hand model is licensed by the Max Planck Institute and cannot be
redistributed, so each user downloads it.

1. Register at <https://mano.is.tue.mpg.de> and accept the license.
2. Download **Models & Code** (`mano_v1_2.zip`) and unzip it. The file needed
   is `mano_v1_2/models/MANO_RIGHT.pkl`.
3. Convert it to the tensor file used here (needs only numpy, scipy, torch;
   chumpy is not required):

   ```bash
   python mano_pkl_to_pt.py --pkl /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
       --out weights/mano/MANO_RIGHT.pt
   ```

4. Run `python check_weights.py`; every line should read `OK`. A MANO
   conversion made with a different MANO release may report `MISMATCH` for
   `MANO_RIGHT.pt` and still work. `MANO_LEFT` is not needed: left hands are
   estimated as mirrored right hands.

**`mano_mean_params.npz` is not required.** The mean MANO pose, shape and
camera parameters are stored in the WiLoR checkpoint, which overrides any
initial values. If `weights/mano/mano_mean_params.npz` is present it is
loaded as the initial value and then overwritten by the checkpoint, so the
results are identical. It is the mean-parameter file distributed with HaMeR
and WiLoR (their demo data downloads) for anyone who wants it, but this
package does not use it.

## 4. Quick start: one recording

An EgoForce recording directory contains `head.mp4` and `timestamps.json`:

```bash
python extract_handpose.py /data/EgoForce/20260902/Pour_water/BNN_2828_E_01/recording_001 \
    --output recording_001_handpose.npz --device cuda --fp16
```

A one-minute recording takes about 2 minutes on one GPU. For a quick check, add
`--max-frames 300`. The command prints per-hand coverage and stabilizer
statistics.

## 5. Batch: a date folder (or any folder)

Give a folder instead of a recording; every recording below it is processed
with one model load:

```bash
python extract_handpose.py /data/EgoForce/20260902 --output handpose_out/20260902 \
    --device cuda --fp16
```

Outputs mirror the input tree as
`handpose_out/20260902/<task>/<collector>/recording_NNN/handpose.npz`.
Recordings that already have an output are skipped, so an interrupted batch
resumes where it stopped (`--overwrite` recomputes). One line per recording,
with coverage, is appended to `handpose_out/20260902/summary.jsonl`; a failed
recording is reported and the batch continues. To use several GPUs, run one
process per GPU on different folders (`CUDA_VISIBLE_DEVICES=k`).

## 6. Output

| key | dtype, shape | meaning |
|---|---|---|
| `frame_t` | float32 (N,) | frame time in seconds, equal to `timestamps.json["head"]` |
| `L_xy`, `R_xy` | float16 (N,21,2) | 2D joints in `head.mp4` pixels |
| `L_mm`, `R_mm` | float16 (N,21,3) | 3D joints in millimeters, relative to the wrist |
| `L_ok`, `R_ok` | bool (N,) | hand estimated on that frame |
| `scale` | float32 | factor from inference pixels to `*_xy` (0.5 for MP4 input) |
| `resized_wh` | int32 (2,) | width, height of the frame `*_xy` refers to (1920, 600) |

Row `i` is frame `i` of `head.mp4` (N = number of video frames). Frames where a
hand was not estimated are zero with `*_ok == False`.

```python
import numpy as np
z = np.load("recording_001_handpose.npz")
i = 300
if z["R_ok"][i]:
    wrist_xy = z["R_xy"][i, 0]        # pixels in head.mp4 frame i
    fingertip_mm = z["R_mm"][i, 8]    # index fingertip, mm from the wrist
```

## 7. Coordinates and conventions

* `head.mp4` is 1920x600 side-by-side stereo. Inference runs on the left eye
  upscaled to its native 1920x1200, and joints are scaled by 0.5 back into the
  video frame, so all `*_xy` lie in the left half: x in [0, 960), y in
  [0, 600), x to the right, y down.
* Joint order (21): 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring;
  17-20 little finger (each from base to tip).
* `*_mm` are camera-oriented 3D joints (x right, y down, z forward) relative
  to the wrist, at MANO metric scale.
* `L`/`R` are the wearer's left and right hands, decided as described in
  "How it works" (geometry, appearance and motion continuity over the whole
  recording, not an anatomical classifier).
* A raw `.mcap` recording is also accepted as input; then coordinates are in
  native left-image pixels (1920x1200) and `scale` is 1.

## 8. Parameters

| flag | default | |
|---|---|---|
| `--pose-hz` | 10 | detection and pose rate before interpolation; 0 = every frame |
| `--upscale` | 2.0 | MP4 left-view upscale before inference |
| `--device` | auto | `cpu`, `cuda` or `auto` |
| `--fp16` | off | half precision pose model (CUDA) |
| `--weights` | `weights/` | model asset directory |
| `--max-frames` | all | process only the first N frames |
| `--save-predictions` | off | single mode: also write raw per-pose predictions (`.jsonl.gz`) |
| `--overwrite` | off | batch mode: recompute existing outputs |

Labeling costs and detection thresholds are constants at the top of
`labeling.py`; stabilizer thresholds are constants at the top of `stabilize.py`.
`python stabilize.py --preds predictions.jsonl.gz --timestamps timestamps.json`
re-stabilizes saved predictions and prints statistics.

## 9. Known limitations

* Monocular estimate: depth uses the model's focal-length heuristic, not the
  stereo pair. 3D joints are wrist-relative; absolute 3D hand position is
  not provided.
* Hands are missed when heavily occluded, motion-blurred, at the image border
  or out of view; those frames are `*_ok == False`. Holes up to 0.3 s are
  interpolated, longer ones are left empty.
* Labels rely on image geometry, glove appearance and motion continuity.
  Long stretches with crossed hands can be labeled the wrong way round.
* White objects that look like a glove can be taken for a hand, especially
  while the real hand is occluded or out of view. White sneakers are
  recognized by the shoe score and rejected; when a glove holds a shoe and one
  box covers both, that hand may be dropped for a few frames rather than
  mislabeled.
* The detector is prompted for white gloves; bare hands or other gloves are
  detected less reliably.
* Detector scores are not joint confidences.

## NW_bench: bare hands, fisheye and stereo (`run_nwbench.py`)

`extract_handpose.py` is tuned for white gloves and the EgoForce head camera.
NW_bench is different: bare hands, a
head-mounted Double Sphere fisheye camera (mono or stereo), and some plain
MP4s without calibration. `run_nwbench.py` reuses the labeling, pose
hypotheses and stabilizer of this package and replaces the rest:

* **Detection** (`hand_detector.py`): the YOLO hand detector shipped with
  WiLoR instead of the glove-prompted YOLO-World, which scores these bare
  hands below 0.1. The detector finds a hand only when the hand has a
  suitable size in its input: a close-up hand at a small input size, a
  distant one at a large size. It is therefore run at four sizes (320 to
  640 px) and the boxes are merged; a box seen at one size only scores low.
  Dropped are person-sized boxes; on the fisheye views, boxes touching the
  bottom border (the wearer's own face); on videos without calibration, boxes
  less than a quarter as large as the nearest hands of that video. The
  wearer's hands are the hands nearest to the camera, so the reference is
  the largest confident box of each frame; a crowd of bystanders cannot
  outvote it. These boxes go before the labeling, where they would otherwise
  take the place of a real hand, and before the list of a frame is cut to
  its five best boxes: in front of a crowd the bystanders' hands outscore the
  wearer's own and would use up the list. Among the remaining boxes the labeling
  prefers the larger ones: a box less than half as large as the reference
  pays a cost that grows as the box gets smaller, so a coworker's hand in the
  background does not take the label from the wearer's hand in front of it.
* **Two hands on one object**: the labeling of `labeling.py` refuses two boxes
  that share more than 60 % of the smaller one, which for gloves means two
  boxes on one hand. Bare hands that shred or knead something lie against
  each other, and their boxes overlap more than that; the limit is 80 % here.
  Two poses that do end up on one hand are removed after the pose step (see
  hand verification).
* **Left and right**: `labeling.py` assigns the labels by position. When both
  arms reach in from one side this is the wrong way round for the whole
  video, and the pose network, given the wrong handedness, returns unstable
  poses. Two independent checks must agree before the labels are swapped:
  the detector, which also classifies each box as left or right, must lean
  against the labels over the whole video; then 40 pose samples are computed
  with both handedness flags, and the flipped flags must lower the
  disagreement of the pose hypotheses by at least 20 %. On 19 NW_bench
  videos the flipped flags changed the disagreement by a factor of 0.60 on
  the one video with both arms from one side and by 1.27 to 4.7 on the
  others. `summary.json` reports `handedness_trial` and `left_right_swapped`.
  Position also fails for a part of a video: one hand alone in the picture
  on the other hand's side, or a hand next to an object taken for one. How
  far the detector's vote on a box can be trusted differs between videos:
  with WiLoR's weights it agrees with the labels on nearly every box of the
  fisheye samples and is at chance on the gloved hands of `B06`; a detector
  adapted to the videos (see below) has learned what their left and right
  hands look like. Where the votes separate the two labels of a video (the
  share of "right" votes is at least 0.5 higher on the boxes labeled right
  than on those labeled left), the labeling is repeated with every vote
  against a label as a cost; elsewhere nothing changes
  (`detector_side_votes_used` in `summary.json`).
* **Per-hand perspective view** (`fisheye.py`): each labeled hand is
  re-rendered through a virtual pinhole camera aimed at it, so the pose
  network sees an undistorted, centered hand. Keypoints are mapped back
  through the Double Sphere model of `camera.json`.
* **Pose hypotheses** are scored against skin color instead of glove white.
* **Hand verification**: a box is kept only when the box, refit and rotated
  pose hypotheses agree. On a box the detector is sure of (score 0.45 or
  more) they may differ more: a hand seen edge-on or half hidden is read in
  more than one way by the pose network. The hand must be within arm's
  reach: 1 m with a calibrated camera,
  1.5 m at the assumed focal length without one. Finally the detector must
  confirm the hand: poses that can be the same hand form a run (it may pause
  for one sample),
  and a pose counts only if within 1 s of it the run holds a pose that the
  detector scores 0.45 or more, in the full frame or in the hand's own
  perspective view, where every hand has the size the detector handles best.
  A hand may thus be unsure for a moment (motion blur, a hand under a
  plastic bag), while a plate or a jar that the pose network takes for a hand
  never gets such a score and is dropped as a whole. A left and a right pose
  that coincide joint by joint lie on one hand; the one with less detector
  support is dropped.
* **Tracking through detector misses**: a hand the detector loses is followed
  from the neighboring pose sample, forward and backward, for up to 1 s, as
  long as the result still passes the verification; a sample reached from
  both sides takes the pose that is fewer steps from its detection. A hand
  is output while at least 8 of its 21 keypoints lie inside the picture. It
  is followed further, down to 2 keypoints, so that a hand that dips out of
  the picture and comes back does not have to be detected again.
* **Stabilization** uses a 0.4 s smoothing window and repairs only isolated
  outliers, so the start of a fast motion is not flattened; poses are sampled
  at 15 Hz (`--pose-hz`). A run ends where the label jumps to another hand
  (more than one hand size per 0.1 s, or an abrupt change of size), smoothing
  may not move a sample more than 0.3 hand sizes from its measured pose, and
  only a single missing sample is interpolated; otherwise a skeleton appears
  between two hands. A frame between two pose samples that is not
  interpolated (a run ends there, or the next sample is missing) takes the
  pose of the nearest sample, at most half a sampling period away. These
  rules are the `max_step` and `hold` options of `stabilize.py`;
  `extract_handpose.py` does not use them.
* **Metric 3D**: the wrist-relative joints are placed in the camera frame by
  fitting them to the viewing rays of their 2D keypoints. Stereo cases use the
  rays of cam2 and cam3 together, which triangulates depth and calibrates the
  hand size per video; mono cases rely on the MANO hand size. Videos without
  `camera.json` use an assumed pinhole focal length (0.6 x the longer image
  side), so their absolute depth is only approximate.

Extra requirements: `pip install ultralytics dill`, and the detector weights
at `weights/hand_detector/detector.pt`
([download](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt)).
The file is a pickle, so load only a copy you trust. `--detector` (or
`HANDPOSE_HAND_DETECTOR`) selects another file.

```bash
python run_nwbench.py /data/NW_bench --out results                    # all samples
python run_nwbench.py /data/NW_bench --out results --only stereo/case0 mono_video_only/H02
CUDA_VISIBLE_DEVICES=1 python run_nwbench.py /data/NW_bench --out results --shard 1/4   # one process per GPU
```

### Adapting the detector to the videos (`train_detector.py`)

WiLoR's detector scores hands in thin plastic gloves, close to the camera or
smeared by motion, so low that many of them never become a candidate box, at
any input size. The pipeline still finds a good part of them, because it
follows a hand through detector misses and checks every pose. Those verified
hands are training data the detector has not seen. `train_detector.py` takes
evenly spaced frames in which both hands were found, with the box and the 21
keypoints of each hand as labels (the detector is a YOLOv8 pose model with
exactly these keypoints), and fine-tunes the detector on them for a few
epochs. Frames with one hand are left out, because the other hand may be
visible but missed and would be learned as background. Bystanders' hands carry
no label, so the detector also learns to leave them alone. No ground truth is
used, and for the 19 samples a round takes two minutes. The results below come
from two rounds:

```bash
python run_nwbench.py data/NW_bench --out pass0                                  # WiLoR's detector
python train_detector.py data/NW_bench pass0 --out weights/hand_detector/pass1.pt
python run_nwbench.py data/NW_bench --out pass1 --detector weights/hand_detector/pass1.pt
python train_detector.py data/NW_bench pass1 --out weights/hand_detector/detector_nwbench.pt
python run_nwbench.py data/NW_bench --out results --detector weights/hand_detector/detector_nwbench.pt
```

Every round starts from WiLoR's weights. Training needs `pip install polars`.
What the adapted detector learns is the look of the hands in the videos it was
trained on: it has to be trained on the videos that are to be processed. A
detector trained without a video does not help on that video (see the
leave-one-out row below).

Each sample gets `results/<group>_<name>/` with `handpose.pkl`, `handpose.npz`,
`summary.json` and `overlay.mp4`. `handpose.pkl` holds, for `left` and
`right`: `kp3d` (N,21,3) meters in the reference camera frame (cam3; +X right,
+Y down, +Z forward), `kp3d_wrist_rel_mm`, `kp2d` (pixels of the reference
video), `valid` (N,) and `depth_source` (0 none, 1 mono, 2 stereo). For stereo
cases `summary.json` reports the reprojection error of the fused hand in the
second view and how far the mono depth is from the triangulated one.
`python eval/demo_video.py results results/demo.mp4` strings six seconds of
every overlay into one video.

Left/right is decided from image position and motion continuity, as in the
EgoForce pipeline, followed by the whole-video check described above. Labels
that are wrong for only a part of a video are corrected where the detector's
votes can be trusted; with WiLoR's weights that leaves out the videos with
gloved hands, after adaptation it covers them.

`python eval/score.py results` reports recall and precision on frames
labeled by eye. For eight of the nine videos without calibration,
`eval/hand_points.json` holds the position and the side (left or right) of
every hand on 59 to 95 evenly spaced frames: the wearer's clearly visible
hands, which must be found; the wearer's hands that are cut by the image
border, hidden or blurred ("partly visible"), which count neither way; and
other people's hands, where a skeleton is an error. For the fisheye samples
and `H02`, `eval/visible_hands.json` holds the number of visible hands on 41
to 47 frames each; this does not check where a skeleton lies. Frames without
a visible hand do not count against recall.

The run stored in `results/` (detector adapted in two rounds) scores as
follows; the second column is the same pipeline with WiLoR's detector as
shipped (`pass0` above).

| Sample | Recall, WiLoR's detector | Recall, adapted (found / labeled) | Precision, adapted (false skeletons) | Left/right correct | Partly visible hands with a skeleton |
|---|---|---|---|---|---|
| `mono_video_only/B02` | 93.3 % | 96.6 % (115 / 119) | 97.5 % (3) | 115 / 115 | 39 / 59 |
| `mono_video_only/B06` | 83.7 % | 96.9 % (95 / 98) | 97.9 % (2) | 95 / 95 | 14 / 17 |
| `mono_video_only/B21` | 91.2 % | 99.1 % (112 / 113) | 100 % (0) | 112 / 112 | 7 / 7 |
| `mono_video_only/B47` | 100 % | 100 % (82 / 82) | 97.6 % (2) | 82 / 82 | 16 / 18 |
| `mono_video_only/H07` | 100 % | 100 % (118 / 118) | 100 % (0) | 118 / 118 | 2 / 2 |
| `mono_video_only/H18` | 100 % | 100 % (106 / 106) | 100 % (0) | 106 / 106 | 9 / 10 |
| `mono_video_only/L04` | 100 % | 100 % (43 / 43) | 97.7 % (1) | 42 / 42 | 19 / 31 |
| `mono_video_only/L07` | 97.8 % | 100 % (46 / 46) | 97.9 % (1) | 45 / 46 | 19 / 26 |
| all eight | 95.2 % (690 / 725); precision 98.3 % (12), left/right 676 / 689, partly visible 100 / 170 | 98.9 % (717 / 725) | 98.8 % (9) | 715 / 716 | 125 / 170 |
| `H02`, `mono/case0`, `stereo/case0`, `mono/case10`, `mono/case30`, `stereo/case40` (counts) | 100 % | 100 % (488 / 488) | 100 % (0) | not labeled | |

On the 10 fisheye samples the adapted detector changes the coverage of a hand
by at most 1.9 %. The stereo samples, fused from both cameras:

| Sample | Reprojection error in the second view, left / right | Mono depth against triangulated depth, left / right |
|---|---|---|
| `stereo/case0` | 2.6 / 3.3 px | 1.4 / 3.4 % |
| `stereo/case22` | 3.2 / 2.9 px | 4.0 / 3.9 % |
| `stereo/case40` | 2.7 / 6.2 px | 2.7 / 7.7 % |
| `stereo/case55` | 4.3 / 5.5 px | 2.2 / 2.5 % |

Coverage, the share of all frames of a video with a skeleton, also counts the
frames in which the hand is not in the picture, and in several of these videos
that is most of the time. The labels say how often a hand is in view; it
cannot be covered more often than that. `score.py` prints both:

| Sample | Frames with a skeleton, left / right | Hand in view at least partly, left / right | Hand clearly visible, left / right |
|---|---|---|---|
| `mono/case0` | 67.4 % / 68.3 % | both hands together: 69.6 % (counts) | |
| `mono/case5` | 100.0 % / 100.0 % | not labeled | |
| `mono/case10` | 100.0 % / 100.0 % | both hands together: 100.0 % (counts) | |
| `mono/case22` | 100.0 % / 100.0 % | not labeled | |
| `mono/case30` | 100.0 % / 100.0 % | both hands together: 100.0 % (counts) | |
| `mono/case47` | 100.0 % / 100.0 % | not labeled | |
| `stereo/case0` | 76.1 % / 72.7 % | both hands together: 74.4 % (counts) | |
| `stereo/case22` | 100.0 % / 100.0 % | not labeled | |
| `stereo/case40` | 99.6 % / 100.0 % | both hands together: 100.0 % (counts) | |
| `stereo/case55` | 100.0 % / 100.0 % | not labeled | |
| `mono_video_only/B02` | 69.4 % / 92.9 % | 83.0 % / 97.9 % | 40.4 % / 86.2 % |
| `mono_video_only/B06` | 93.7 % / 92.9 % | 95.0 % / 96.7 % | 86.7 % / 76.7 % |
| `mono_video_only/B21` | 99.2 % / 97.8 % | 100.0 % / 100.0 % | 96.7 % / 91.7 % |
| `mono_video_only/B47` | 84.8 % / 83.2 % | 83.3 % / 83.3 % | 76.7 % / 60.0 % |
| `mono_video_only/H02` | 100.0 % / 100.0 % | both hands together: 100.0 % (counts) | |
| `mono_video_only/H07` | 100.0 % / 100.0 % | 100.0 % / 100.0 % | 98.3 % / 98.3 % |
| `mono_video_only/H18` | 99.4 % / 93.0 % | 100.0 % / 93.3 % | 96.7 % / 80.0 % |
| `mono_video_only/L04` | 35.3 % / 34.7 % | 35.8 % / 40.0 % | 22.1 % / 22.1 % |
| `mono_video_only/L07` | 30.7 % / 79.1 % | 33.9 % / 86.4 % | 20.3 % / 57.6 % |

Where coverage exceeds the share in view, the difference is sampling (the
labels cover every 8th to 15th frame) and the few false skeletons.

How far these numbers carry:

* The adapted detector was trained on these same videos, on the pipeline's
  own output. Leave-one-out, measured with the previous version of the
  pipeline and the detector trained on the other 18 samples only, `B02`,
  `B06` and `B21` together reached 295 of 330 hands with 12 false skeletons,
  against 296 and 8 with WiLoR's detector: no gain. New videos need their own
  adaptation rounds.
* After one round the eight videos reach 714 of 725 hands with 9 false
  skeletons, after two 717 with 9; `B06` goes from 83.7 % to 92.9 % to
  96.9 %. A third round, tried on the detector of the previous version,
  changed single videos by a few hands but not the total, and stronger shift
  augmentation in the training cost three hands and added three false
  skeletons. A few hands per video move with the training set.
* After adaptation the left/right votes come from a detector that learned
  the names from the pipeline's own labels. They correct stretches that
  deviate from the rest of a video; a video labeled the wrong way round as a
  whole stays that way unless the handedness trial catches it.
* The remaining misses are gloved hands under a plastic bag or small behind
  the other arm (`B02`) and hands very close to the camera (`B06`, `B21`);
  the false skeletons lie on a plastic bag and next to a cleaver whose hand
  is outside the picture (`B02`), at the image border (`B47`) and on a glass
  on the bar top (`L07`). Raising the number of keypoints that must lie
  inside the picture removes some of the border cases and as many partly
  visible hands with them.

## Lower-level tools

* `run.py`: the earlier frame-by-frame detector/CSRT-tracker/pose runner over raw MCAP files or numbered
  sessions, writing full per-frame predictions (MANO parameters, vertices,
  tracking diagnostics) and optional preview images or videos
  (`python run.py --help`).
* `process_poses.py`: optional causal MANO EKF over existing `run.py` exports.
* `mano_pkl_to_pt.py`, `check_weights.py`: MANO conversion and asset check.

## Citation

```bibtex
@article{cao2026fact,
  title   = {FACT: A Multimodal Force-Action-Track Foundation Model for Human-to-Robot Dexterity},
  author  = {Cao, Zhi and Hong, Yining and Yang, Jiaolong and Choi, Yejin and Fei-Fei, Li and Wu, Jiajun and Huang, Huang},
  journal = {arXiv preprint arXiv:2610.19155},
  year    = {2026}
}
```

This package builds on WiLoR, AnyHand, YOLO-World and MANO; please also cite
those works when you use it.

## Licenses

Code derived from AnyHand (CC0 1.0), WiLoR (CC BY-NC-ND 4.0) and YOLO-World
(GPLv3) keeps the notices in `licenses/`. The bundled weights keep the terms of
their original releases. MANO is distributed only by the Max Planck Institute
(<https://mano.is.tue.mpg.de/license.html>). For non-commercial research use.
