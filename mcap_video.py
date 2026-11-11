"""Timestamp-preserving decoding of Foxglove H.264 MCAP video.

Decode from the beginning so reference pictures are available. Packet PTS is
the unique packet index, so delayed/reordered pictures retain their metadata.
Only selected pictures are converted to BGR. Source recordings are read-only.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

HEAD_TOPIC = "/camera/head/stereo/h264"
CALIBRATION_TOPIC = "/camera/head/left/calibration"


@dataclass
class VideoFrame:
    image: np.ndarray
    packet_index: int
    log_time_ns: int
    publish_time_ns: int
    capture_time_ns: int | None
    offset_s: float
    stereo_width: int
    stereo_height: int


def inspect_recording(path: Path, topic=HEAD_TOPIC):
    with Path(path).open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError("MCAP summary statistics are required for a stable recording origin")
        channels = [c for c in summary.channels.values() if c.topic == topic]
        if not channels:
            raise ValueError(f"No {topic} in {path}")
        calibration = None
        decoder = DecoderFactory()
        for schema, channel, message in reader.iter_messages(topics=[CALIBRATION_TOPIC]):
            fn = decoder.decoder_for(channel.message_encoding, schema)
            if fn is None:
                raise ValueError("Unsupported calibration encoding")
            calibration = MessageToDict(fn(message.data))
            break
        stats = summary.statistics
        return {
            "source_mcap": str(Path(path).resolve()),
            "topic": topic,
            "recording_start_ns": stats.message_start_time,
            "recording_end_ns": stats.message_end_time,
            "duration_s": (stats.message_end_time - stats.message_start_time) / 1e9,
            "video_packet_count": sum(stats.channel_message_counts.get(c.id, 0) for c in channels),
            "left_calibration": calibration,
        }


def left_view(stereo, layout="side-by-side"):
    if layout == "left-only":
        return np.ascontiguousarray(stereo)
    height, width = stereo.shape[:2]
    if width % 2:
        raise ValueError(f"Stereo width {width} is not even")
    return np.ascontiguousarray(stereo[:, :width // 2])


def iter_left_frames(path, *, start_s=0., duration_s=None, sample_hz=1.,
                     times=None, max_frames=None, topic=HEAD_TOPIC,
                     layout="side-by-side", all_frames=False):
    if start_s < 0 or sample_hz <= 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("Require start >= 0, sample_hz > 0, duration > 0")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    info = inspect_recording(Path(path), topic)
    origin = info["recording_start_ns"]
    requested = sorted(set(float(t) for t in times)) if times is not None else None
    if requested is not None and (not requested or requested[0] < 0):
        raise ValueError("Sample times must be nonempty and nonnegative")
    cursor = 0
    target = requested[0] if requested else start_s
    stop_s = start_s + duration_s if duration_s is not None else float("inf")
    emitted = 0
    codec = av.CodecContext.create("h264", "r")
    codec.thread_count = 2
    decoder = DecoderFactory()
    decoders = {}
    pending = {}

    def selected(decoded):
        nonlocal emitted, cursor, target
        if decoded.pts is None or decoded.pts not in pending:
            raise ValueError("H.264 decoder did not preserve the packet PTS; refusing guessed timestamps")
        packet_index, message, capture_ns = pending.pop(decoded.pts)
        offset = (message.log_time - origin) / 1e9
        if offset + 1e-9 < target or offset >= stop_s:
            return None
        if max_frames is not None and emitted >= max_frames:
            return None
        emitted += 1
        if all_frames:
            # Native-frame consumers (tracking) must see every picture in the
            # window, independently of the downstream pose-output schedule.
            pass
        elif requested is not None:
            # A picture may satisfy multiple closely spaced target times; emit
            # it once, then advance beyond its timestamp.
            while cursor < len(requested) and requested[cursor] <= offset + 1e-9:
                cursor += 1
            target = requested[cursor] if cursor < len(requested) else float("inf")
        else:
            steps = max(1, int(np.floor((offset - target) * sample_hz + 1e-7)) + 1)
            target += steps / sample_hz
        stereo = decoded.to_ndarray(format="bgr24")
        h, w = stereo.shape[:2]
        return VideoFrame(left_view(stereo, layout), packet_index, message.log_time,
                          message.publish_time, capture_ns, offset, w, h)

    with Path(path).open("rb") as stream:
        reader = make_reader(stream)
        for index, (schema, channel, message) in enumerate(reader.iter_messages(topics=[topic])):
            if channel.id not in decoders:
                fn = decoder.decoder_for(channel.message_encoding, schema)
                if fn is None:
                    raise ValueError(f"Unsupported encoding for {topic}")
                decoders[channel.id] = fn
            value = decoders[channel.id](message.data)
            if value.format.lower() not in {"h264", "h.264"}:
                raise ValueError(f"Expected H.264, got {value.format}")
            capture_ns = None
            if value.HasField("timestamp"):
                capture_ns = value.timestamp.seconds * 1_000_000_000 + value.timestamp.nanos
            pending[index] = (index, message, capture_ns)
            packet = av.Packet(bytes(value.data))
            packet.pts = packet.dts = index
            packet.time_base = Fraction(1, 1_000_000)
            for decoded in codec.decode(packet):
                result = selected(decoded)
                if result is not None:
                    yield result
            if (max_frames is not None and emitted >= max_frames) or target == float("inf"):
                return
            # H.264 can buffer pictures. Drain below before stopping a window.
            if (message.log_time - origin) / 1e9 > stop_s + 2:
                break
        for decoded in codec.decode(None):
            result = selected(decoded)
            if result is not None:
                yield result


class FrameSampler:
    """Select exported pose pictures independently from native tracking updates."""

    def __init__(self, *, start_s=0., sample_hz=1., times=None, max_frames=None):
        self.times = sorted(set(float(t) for t in times)) if times is not None else None
        self.cursor = 0
        self.target = self.times[0] if self.times else float(start_s)
        self.sample_hz = float(sample_hz)
        self.max_frames = max_frames
        self.emitted = 0

    @property
    def done(self):
        return ((self.max_frames is not None and self.emitted >= self.max_frames)
                or (self.times is not None and self.cursor >= len(self.times)))

    def select(self, timestamp_s):
        if self.done or timestamp_s + 1e-9 < self.target:
            return False
        self.emitted += 1
        if self.times is not None:
            while self.cursor < len(self.times) and self.times[self.cursor] <= timestamp_s + 1e-9:
                self.cursor += 1
            self.target = self.times[self.cursor] if self.cursor < len(self.times) else float("inf")
        else:
            steps = max(1, int(np.floor((timestamp_s-self.target)*self.sample_hz + 1e-7))+1)
            self.target += steps / self.sample_hz
        return True
