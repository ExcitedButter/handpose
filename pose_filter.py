"""Causal MANO error-state EKF; no learned weights or image-model changes.

Each rotation has a right-multiplicative SO(3) error and body angular velocity.
Independent 6x6 covariance blocks keep memory constant per tracked hand.
Noise settings are tuning parameters, not calibrated model uncertainties.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_rotation(vector):
    return cv2.Rodrigues(np.asarray(vector, np.float64).reshape(3))[0]


def log_rotation(matrix):
    matrix = np.asarray(matrix, np.float64).reshape(3, 3)
    sine_axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                          matrix[1, 0] - matrix[0, 1]]) / 2
    sine = np.linalg.norm(sine_axis)
    cosine = np.clip((np.trace(matrix) - 1) / 2, -1., 1.)
    if sine < 1e-6 and cosine > 0:
        # OpenCV's matrix-to-vector conversion rounds very small angles to
        # zero; retain them for EKF corrections and covariance Jacobians.
        return sine_axis * (1 + sine**2 / 6)
    angle = np.arctan2(sine, cosine)
    if np.pi - angle < 1e-4:
        return cv2.Rodrigues(matrix)[0].ravel()
    return sine_axis * angle / sine


def right_jacobian(vector):
    vector = np.asarray(vector, np.float64)
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    if angle < 1e-5:
        return np.eye(3) - .5 * cross + (cross @ cross) / 6
    return (np.eye(3) - (1 - np.cos(angle)) / angle**2 * cross
            + (angle - np.sin(angle)) / angle**3 * (cross @ cross))


def white_acceleration_covariance(dt, variance):
    # Integrated continuous white acceleration, in local tangent coordinates.
    return np.kron(np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), np.eye(3)) * variance


def damped_angular_covariance(dt, tau, variance):
    if dt / tau < 1e-4:
        return white_acceleration_covariance(dt, variance)
    decay = np.exp(-dt / tau)
    angle = tau**2 * (dt - 2 * tau * (1 - decay) + tau / 2 * (1 - decay**2))
    cross = tau**2 / 2 * (1 - decay)**2
    velocity = tau / 2 * (1 - decay**2)
    return np.kron([[angle, cross], [cross, velocity]], np.eye(3)) * variance


@dataclass(frozen=True)
class EKFConfig:
    observation_noise_deg: float = 12.
    angular_accel_noise_deg: float = 300.
    angular_velocity_decay_s: float = .25
    max_innovation_deg: float = 75.
    innovation_gate: float = 16.27  # 3-D chi-square 99.9%; heuristic covariance.
    max_gap_s: float = .6
    reinitialize_after_s: float = .3
    recovery_speed_deg_s: float = 360.
    translation_noise_m: float = .015
    translation_accel_noise: float = 1.5
    shape_noise: float = .7
    shape_process_noise: float = .03

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("EKF settings must be finite and positive")
        if self.max_innovation_deg >= 180:
            raise ValueError("EKF maximum rotation innovation must be below 180 degrees")

    def metadata(self):
        return {"method": "causal_right_error_state_ekf", "version": 1,
                "config": asdict(self), "learned_prior": False,
                "covariance_calibrated": False, "backward_smoothing": False}


class RotationEKF:
    def __init__(self, rotation, config):
        self.config = config
        self.rotation = np.array(rotation, np.float64, copy=True)
        self.velocity = np.zeros(3)
        noise = np.deg2rad(config.observation_noise_deg)
        self.covariance = np.diag([noise**2] * 3 + [np.deg2rad(90.)**2] * 3)
        self.pending_rotation = None
        self.pending_time = 0.
        self.pending_count = 0

    def update(self, observation, dt):
        c = self.config
        decay = np.exp(-dt / c.angular_velocity_decay_s)
        integrated_dt = c.angular_velocity_decay_s * (1 - decay)
        step = self.velocity * integrated_dt
        increment = exp_rotation(step)
        transition = np.eye(6)
        transition[:3, :3] = increment.T
        transition[:3, 3:] = right_jacobian(step) * integrated_dt
        transition[3:, 3:] *= decay
        predicted = self.rotation @ increment
        covariance = transition @ self.covariance @ transition.T
        covariance += damped_angular_covariance(dt, c.angular_velocity_decay_s, np.deg2rad(c.angular_accel_noise_deg)**2)
        self.velocity *= decay
        residual = log_rotation(predicted.T @ observation)
        noise = np.eye(3) * np.deg2rad(c.observation_noise_deg)**2
        innovation = covariance[:3, :3] + noise
        nis = float(residual @ np.linalg.solve(innovation, residual))
        rejected = nis > c.innovation_gate or np.linalg.norm(residual) > np.deg2rad(c.max_innovation_deg)
        reinitialized = False
        if rejected:
            # Recover from a sustained new pose; a one-frame flip cannot reset
            # the filter. Require temporally consistent rejected observations.
            consistent = (self.pending_rotation is not None and
                          np.linalg.norm(log_rotation(self.pending_rotation.T @ observation))
                          < np.deg2rad(30.) + np.linalg.norm(self.velocity) * dt)
            self.pending_time = self.pending_time + dt if consistent else 0.
            self.pending_count = self.pending_count + 1 if consistent else 1
            self.pending_rotation = observation.copy()
            self.rotation, self.covariance = predicted, covariance
            # Unreliable observations should not leave a stale angular rate
            # extrapolating indefinitely into the occluded interval.
            self.velocity *= np.exp(-dt / .05)
            if self.pending_count >= 3 and self.pending_time >= c.reinitialize_after_s:
                limit = np.deg2rad(min(c.max_innovation_deg / 2, c.recovery_speed_deg_s * dt))
                self.rotation = predicted @ exp_rotation(residual * min(1., limit / max(np.linalg.norm(residual), 1e-12)))
                self.velocity[:] = 0
                noise_variance = np.deg2rad(c.observation_noise_deg)**2
                self.covariance = np.diag([noise_variance] * 3 + [np.deg2rad(90.)**2] * 3)
                reinitialized = True
        else:
            gain = np.linalg.solve(innovation, covariance[:3, :]).T
            correction = gain @ residual
            self.rotation = predicted @ exp_rotation(correction[:3])
            self.velocity += correction[3:]
            # Joseph update followed by the right-error coordinate reset.
            remainder = np.eye(6)
            remainder[:, :3] -= gain
            posterior = remainder @ covariance @ remainder.T + gain @ noise @ gain.T
            reset = np.eye(6)
            reset[:3, :3] = right_jacobian(correction[:3])
            self.covariance = reset @ posterior @ reset.T
            self.covariance = (self.covariance + self.covariance.T) / 2
            self.pending_rotation, self.pending_time, self.pending_count = None, 0., 0
        return {"innovation_deg": float(np.rad2deg(np.linalg.norm(residual))),
                "rejected": bool(rejected), "reinitialized": reinitialized}


class HandState:
    def __init__(self, rotations, translation, betas, timestamp_ns, right, config):
        self.rotations = [RotationEKF(r, config) for r in rotations]
        self.translation = np.r_[translation, np.zeros(3)].astype(np.float64)
        self.translation_covariance = np.diag([config.translation_noise_m**2] * 3 + [.5**2] * 3)
        self.betas = np.array(betas, np.float64, copy=True)
        self.shape_covariance = np.full(10, config.shape_noise**2)
        self.timestamp_ns, self.right = timestamp_ns, right

    def update(self, rotations, translation, betas, timestamp_ns, config):
        dt = (timestamp_ns - self.timestamp_ns) / 1e9
        diagnostics = [f.update(r, dt) for f, r in zip(self.rotations, rotations)]
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        prediction = transition @ self.translation
        covariance = transition @ self.translation_covariance @ transition.T
        covariance += white_acceleration_covariance(dt, config.translation_accel_noise**2)
        noise = np.eye(3) * config.translation_noise_m**2
        gain = np.linalg.solve(covariance[:3, :3] + noise, covariance[:3, :]).T
        residual = translation - prediction[:3]
        # Bound individual translation outliers; rotations have a separate gate.
        radius = 4 * np.sqrt(np.diag(covariance[:3, :3] + noise))
        self.translation = prediction + gain @ np.clip(residual, -radius, radius)
        remainder = np.eye(6)
        remainder[:, :3] -= gain
        self.translation_covariance = remainder @ covariance @ remainder.T + gain @ noise @ gain.T
        if self.translation[2] <= 0:
            self.translation = np.r_[translation, np.zeros(3)]
            self.translation_covariance = np.diag([config.translation_noise_m**2] * 3 + [.5**2] * 3)
        self.shape_covariance += config.shape_process_noise**2 * dt
        shape_gain = self.shape_covariance / (self.shape_covariance + config.shape_noise**2)
        self.betas += shape_gain * np.clip(betas - self.betas, -4 * config.shape_noise, 4 * config.shape_noise)
        self.shape_covariance *= 1 - shape_gain
        self.timestamp_ns = timestamp_ns
        return {"dt_s": dt, "rejected_rotation_indices": [i for i, d in enumerate(diagnostics) if d["rejected"]],
                "reinitialized_rotation_indices": [i for i, d in enumerate(diagnostics) if d["reinitialized"]],
                "rotation_innovation_deg": [d["innovation_deg"] for d in diagnostics]}


class MANOPoseEKF:
    def __init__(self, config=None):
        self.config = config or EKFConfig()
        self.states = {}
        self.last_timestamp_ns = None

    def clear(self):
        self.states.clear()
        self.last_timestamp_ns = None

    def filter_hands(self, hands, timestamp_ns):
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise ValueError("Pose EKF requires exact integer nanosecond timestamps")
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            self.clear()
        self.last_timestamp_ns = timestamp_ns
        gap_ns = round(self.config.max_gap_s * 1e9)
        self.states = {k: s for k, s in self.states.items() if 0 < timestamp_ns - s.timestamp_ns <= gap_ns}
        output, seen = [], set()
        for hand in hands:
            identity = hand.get("track_id")
            if not identity or identity in seen:
                raise ValueError("Pose EKF requires a unique persistent track_id for each hand")
            seen.add(identity)
            if hand["right"] not in (0, 1):
                raise ValueError("MANO handedness must be 0 or 1")
            right = int(hand["right"])
            rotations = np.concatenate([np.asarray(hand["mano_global_orient"]).reshape(1, 3, 3),
                                        np.asarray(hand["mano_hand_pose"]).reshape(15, 3, 3)]).astype(np.float64)
            translation = np.asarray(hand["cam_t"], np.float64).reshape(3)
            betas = np.asarray(hand["mano_betas"], np.float64).reshape(10)
            if (right not in (0, 1) or not np.isfinite(rotations).all() or not np.isfinite(translation).all()
                    or not np.isfinite(betas).all() or translation[2] <= 0
                    or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
                    or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
                raise ValueError("Invalid MANO rotations, parameters or positive-depth translation")
            state = self.states.get(identity)
            initialized = state is None or state.right != right
            if initialized:
                state = HandState(rotations, translation, betas, timestamp_ns, right, self.config)
                self.states[identity] = state
                diagnostics = {"dt_s": None, "rejected_rotation_indices": [], "reinitialized_rotation_indices": []}
            else:
                diagnostics = state.update(rotations, translation, betas, timestamp_ns, self.config)
            filtered = dict(hand)
            filtered["mano_global_orient"] = state.rotations[0].rotation[None].astype(np.float32).copy()
            filtered["mano_hand_pose"] = np.stack([f.rotation for f in state.rotations[1:]]).astype(np.float32)
            filtered["mano_betas"] = state.betas.astype(np.float32).copy()
            filtered["cam_t"] = state.translation[:3].astype(np.float32).copy()
            if "pred_cam" in hand:
                old_camera = np.asarray(hand["pred_cam"], np.float64).reshape(3)
                # Invert cam_crop_to_full using the original crop's scale and
                # offset. Preserve consistency with the filtered root position.
                camera = old_camera.copy()
                old_z, new_z = translation[2], float(filtered["cam_t"][2])
                camera[0] *= old_z / new_z
                camera[1:] = filtered["cam_t"][:2] - (translation[:2] - old_camera[1:]) * new_z / old_z
                filtered["pred_cam"] = camera.astype(np.float32)
            filtered["pose_filter"] = {"method": "ekf", "initialized": initialized, **diagnostics}
            output.append(filtered)
        return output


def regenerate_geometry(hands, mano, image_width, image_height):
    """Update every geometric field from canonical-right filtered parameters."""
    if not hands:
        return hands
    import torch
    device, dtype = mano.v_template.device, mano.v_template.dtype
    params = {target: torch.as_tensor(np.stack([h[source] for h in hands]), device=device, dtype=dtype)
              for target, source in [("global_orient", "mano_global_orient"),
                                     ("hand_pose", "mano_hand_pose"), ("betas", "mano_betas")]}
    with torch.inference_mode():
        geometry = mano(**params, pose2rot=False)
    vertices = geometry.vertices.float().cpu().numpy()
    joints = geometry.joints.float().cpu().numpy()
    for i, hand in enumerate(hands):
        sign = 2 * int(hand["right"]) - 1
        vertices[i, :, 0] *= sign
        joints[i, :, 0] *= sign
        hand["vertices"], hand["joints_3d"] = vertices[i], joints[i]
        for local, camera, projected in [("vertices", "vertices_camera_3d", "vertices_2d"),
                                         ("joints_3d", "joints_camera_3d", "keypoints_2d")]:
            points = hand[local] + np.asarray(hand["cam_t"])[None]
            if np.any(points[:, 2] <= 0):
                raise ValueError("Filtered MANO geometry projects behind the camera")
            hand[camera] = points
            hand[projected] = (float(hand["focal_length"]) * points[:, :2] / points[:, 2:]
                               + np.array([image_width / 2, image_height / 2]))
    return hands
