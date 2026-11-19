"""One demo video from the overlays of all samples of a run_nwbench.py result folder.

    python eval/demo_video.py results results/demo.mp4 [seconds per sample, default 6]

Takes the middle seconds of every <sample>/overlay.mp4, fits them into a 1280x720 picture and
writes the sample name and the share of frames with a left / right skeleton on top, for the labeled
samples also how often that hand is in view at all (see score.py).
"""
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import in_view  # noqa: E402

W, H, FPS = 1280, 720, 30
GROUPS = ["mono", "stereo", "mono_video_only"]


def order(name):
    group, sample = name.split("/")
    return GROUPS.index(group), int("".join(c for c in sample if c.isdigit())), sample


def main():
    res, out = Path(sys.argv[1]), sys.argv[2]
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    names = {json.loads(line)["sample"] for line in (res / "summary.jsonl").read_text().splitlines()}
    here = Path(__file__).parent
    points = json.loads((here / "hand_points.json").read_text())["samples"]
    counts = json.loads((here / "visible_hands.json").read_text())["labels"]
    with av.open(out, "w") as dst:
        stream = dst.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}
        for name in sorted(names, key=order):
            folder = res / name.replace("/", "_")
            s = json.loads((folder / "summary.json").read_text())
            text = f"{name}   skeleton on L {s['coverage_L']:.0%}  R {s['coverage_R']:.0%} of frames"
            v = in_view(points[name]) if name in points else None
            if v:
                text += f"   (hand in view: L {v['L'][0]:.0%}  R {v['R'][0]:.0%})"
            elif name in counts:
                text += f"   (hands in view: {sum(counts[name].values()) / len(counts[name]) / 2:.0%})"
            with av.open(str(folder / "overlay.mp4")) as src:
                vs = src.streams.video[0]
                fps = float(vs.average_rate)
                step, take = max(1, round(fps / FPS)), int(secs * fps)
                start = max(0, (vs.frames - take) // 2)
                for i, frame in enumerate(src.decode(vs)):
                    if i >= start + take:
                        break
                    if i < start or (i - start) % step:
                        continue
                    img = frame.to_ndarray(format="bgr24")
                    k = min(W / img.shape[1], H / img.shape[0])
                    img = cv2.resize(img, (int(img.shape[1] * k) // 2 * 2, int(img.shape[0] * k) // 2 * 2),
                                     interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((H, W, 3), np.uint8)
                    y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
                    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
                    cv2.putText(canvas, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(canvas, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                        dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    print(out)


if __name__ == "__main__":
    main()
