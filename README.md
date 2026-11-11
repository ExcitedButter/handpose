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
   white pixels inside it, since the EgoForce gloves are white.
2. **Hand labels** (`labeling.py`, offline). All candidates of the recording
   are labeled at once by a Viterbi search over per-frame assignments
   (left box, right box), each possibly absent. The path cost rewards
   confident boxes; penalizes boxes with almost no white pixels (sleeves,
   wristbands), left/right crossing, a hand far on the opposite side of the
   image, abrupt motion or size changes between frames, and every appearance
   or disappearance of a hand. Because the search sees the whole recording, a
   hand that re-enters the view is labeled from the past and the future,
   not only from the frame where it reappears.
3. **Pose.** AnyHand-WiLoR regresses MANO parameters, 21 3D joints and their 2D
   projections for each labeled hand box at `--pose-hz`.
4. **Stabilization** (`stabilize.py`, offline, zero phase). Drop detections
   with score below 0.05; reject poses whose wrist lies outside the hand's own
   box (more than 0.6 box diagonals from its center); split at holes
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
we measured about 35 video frames/s per process; CPU works but is much slower.

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

A one-minute recording takes about 100 s on one GPU. For a quick check, add
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
* White objects that look like a glove (for example a white sneaker held next
  to the hands) can be taken for a hand, especially while the real hand is
  occluded or out of view.
* The detector is prompted for white gloves; bare hands or other gloves are
  detected less reliably.
* Detector scores are not joint confidences.

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
