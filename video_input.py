"""Frame source for EgoForce head-camera MP4 files.

An EgoForce ``head.mp4`` is side-by-side stereo at reduced resolution
(1920x600: two 960x600 eyes). The models were run on the native left eye at
1920x1200, so the left half is upscaled by ``upscale`` (default 2) before
inference. Frame times come from ``timestamps.json`` (key ``head``, seconds),
which share the time base of the tactile and video streams of the recording.
"""
import json
from pathlib import Path

import av
import cv2
import numpy as np

from mcap_video import VideoFrame, left_view


def load_frame_times(recording_dir, key="head"):
    with open(Path(recording_dir) / "timestamps.json") as f:
        times = json.load(f)[key]
    if not times:
        raise ValueError(f"No '{key}' timestamps in {recording_dir}")
    return np.asarray(times, np.float64)


def iter_mp4_frames(mp4_path, frame_times, *, layout="side-by-side", upscale=2.0,
                    max_frames=None, threads=2, keep=None):
    """Yield one VideoFrame per decoded picture, in presentation order.

    ``image`` is the left view resized by ``upscale``; ``offset_s`` is the
    matching entry of ``frame_times``. Decoding stops at the shorter of the
    video and the timestamp list. With ``keep`` (a set of frame indices), only
    those frames are converted and yielded.
    """
    frame_times = np.asarray(frame_times, np.float64)
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        for index, decoded in enumerate(container.decode(stream)):
            if index >= len(frame_times) or (max_frames is not None and index >= max_frames):
                break
            if keep is not None and index not in keep:
                continue
            stereo = decoded.to_ndarray(format="bgr24")
            h, w = stereo.shape[:2]
            image = left_view(stereo, layout)
            if upscale != 1.0:
                image = cv2.resize(image, (int(round(image.shape[1] * upscale)), int(round(image.shape[0] * upscale))),
                                   interpolation=cv2.INTER_LINEAR)
            t = float(frame_times[index])
            ns = int(round(t * 1e9))
            yield VideoFrame(np.ascontiguousarray(image), index, ns, ns, None, t, w, h)
