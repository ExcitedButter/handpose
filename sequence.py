"""Ordered MCAP inputs and tracker continuity across genuine file rollovers."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from tracking import HandTracker


@dataclass(frozen=True)
class Recording:
    path: Path
    session: str | None = None
    number: int | None = None


def recording_inputs(paths, *, sequence=False, first_recording=1):
    if not sequence:
        return [Recording(Path(path)) for path in paths]
    groups = {}
    seen = set()
    for source in paths:
        source = Path(source)
        candidates = sorted(source.rglob("recording_*.mcap")) if source.is_dir() else [source]
        for path in candidates:
            match = re.fullmatch(r"recording_(\d+)\.mcap", path.name)
            if not match or not path.is_file():
                raise ValueError(f"Expected an existing recording_NNN.mcap: {path}")
            path = path.resolve()
            number = int(match[1])
            if number < first_recording or path in seen:
                continue
            seen.add(path)
            session = str(path.parent)
            files = groups.setdefault(session, {})
            if number in files:
                raise ValueError(f"Duplicate recording number {number} in {session}")
            files[number] = path
    if not groups:
        raise ValueError("No recordings at or after the requested first recording")
    result = []
    for session, files in sorted(groups.items()):
        if first_recording not in files:
            raise ValueError(f"Missing recording_{first_recording:03d}.mcap in {session}; cannot initialize this sequence")
        result.extend(Recording(path, session, number) for number, path in sorted(files.items()))
    return result


def camera_signature(info, *, topic, layout, rectify):
    calibration = info.get("left_calibration") or {}
    keys = ("width", "height", "K", "D", "R", "P", "distortionModel", "frameId")
    # Calibration message timestamps vary by file; camera geometry is what matters.
    return (topic, layout, rectify, {key: calibration.get(key) for key in keys})


class TrackingSequence:
    """Reuse the existing tracker on a shared clock, resetting on discontinuity."""

    def __init__(self, detector, *, conf=.12, detect_every=5, max_gap_s=.6,
                 tracker_factory=None):
        self.detector = detector
        self.options = dict(conf=conf, detect_every=detect_every, max_gap_s=max_gap_s)
        self.factory = tracker_factory or HandTracker
        self.max_gap_ns = round(max_gap_s * 1e9)
        self.tracker = None
        self.last_recording = None
        self.last_timestamp_ns = None
        self.last_shape = None
        self.last_signature = None
        self.session_origin_ns = None
        self.segment_origin_ns = None
        self.segment_number = 0
        self.native_index = 0

    def begin_recording(self, recording, signature):
        self.recording = recording
        self.signature = signature
        self.pending_boundary = True
        self.boundary = None

    def _begin_frame(self, image, timestamp_ns):
        current, previous = self.recording, self.last_recording
        gap_ns = None if self.last_timestamp_ns is None else timestamp_ns-self.last_timestamp_ns
        new_session = previous is None or current.session != previous.session
        reason = None
        if new_session:
            reason = "sequence_start" if previous is None else "session_change"
        elif current.number != previous.number+1:
            reason = "recording_number_gap"
        elif image.shape != self.last_shape:
            reason = "image_geometry_change"
        elif self.signature != self.last_signature:
            reason = "camera_configuration_change"
        elif gap_ns <= 0:
            reason = "non_monotonic_timestamp"
        elif gap_ns > self.max_gap_ns:
            reason = "timestamp_gap"
        if new_session:
            self.session_origin_ns = timestamp_ns
            self.segment_number = 0
            self.native_index = 0
        if reason is not None:
            self.tracker = self.factory(self.detector, **self.options)
            self.segment_origin_ns = timestamp_ns
            self.segment_number += 1
        sequence_id = hashlib.sha256(current.session.encode()).hexdigest()[:12]
        self.segment_id = f"{sequence_id}:segment-{self.segment_number:04d}"
        self.sequence_id = sequence_id
        self.boundary = {
            "sequence_id": sequence_id,
            "tracking_segment_id": self.segment_id,
            "recording_number": current.number,
            "previous_source_mcap": str(previous.path) if previous is not None and not new_session else None,
            "continued_from_previous_file": reason is None,
            "reset_reason": reason,
            "video_gap_from_previous_file_s": gap_ns/1e9 if not new_session else None,
            "sequence_start_video_log_time_ns": self.session_origin_ns,
            "first_video_log_time_ns": timestamp_ns,
        }
        self.pending_boundary = False

    def update(self, image, timestamp_ns):
        if self.pending_boundary:
            self._begin_frame(image, timestamp_ns)
        elif image.shape != self.last_shape:
            raise ValueError("Camera image dimensions changed within one recording")
        hands, diagnostics = self.tracker.update(image, (timestamp_ns-self.segment_origin_ns)/1e9)
        for hand in hands:
            hand["track_id"] = f"{self.segment_id}:{hand['track_id']}"
        diagnostics["track_ids"] = [hand["track_id"] for hand in hands]
        context = {
            "sequence_id": self.sequence_id,
            "tracking_segment_id": self.segment_id,
            "sequence_frame_index": self.native_index,
            "sequence_offset_s": (timestamp_ns-self.session_origin_ns)/1e9,
            "recording_number": self.recording.number,
        }
        self.native_index += 1
        self.last_timestamp_ns = timestamp_ns
        self.last_recording = self.recording
        self.last_shape = image.shape
        self.last_signature = self.signature
        return hands, diagnostics, context
