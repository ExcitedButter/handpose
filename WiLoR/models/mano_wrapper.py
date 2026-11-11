from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from WiLoR.models.utils.geometry import aa_to_rotmat


@dataclass
class MANOOutput:
    vertices: torch.Tensor
    joints: torch.Tensor
    betas: Optional[torch.Tensor] = None
    global_orient: Optional[torch.Tensor] = None
    hand_pose: Optional[torch.Tensor] = None


_MANO_TIP_VERTEX_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky
_MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def _tensor_from_asset(value, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().clone().to(dtype=dtype)


def _make_transform(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    batch = rot.shape[0]
    out = torch.zeros(batch, 4, 4, dtype=rot.dtype, device=rot.device)
    out[:, :3, :3] = rot
    out[:, :3, 3:4] = trans
    out[:, 3, 3] = 1.0
    return out


def _batch_rigid_transform(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_joints = joints.shape[:2]
    joints_h = joints.unsqueeze(-1)
    rel_joints = joints_h.clone()
    rel_joints[:, 1:] -= joints_h[:, parents[1:]]

    transforms_mat = _make_transform(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(batch_size, num_joints, 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        transform_chain.append(torch.matmul(transform_chain[parents[i]], transforms_mat[:, i]))
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = torch.cat([joints, torch.zeros(batch_size, num_joints, 1, dtype=joints.dtype, device=joints.device)], dim=2).unsqueeze(-1)
    init_bone = torch.matmul(transforms, joints_homogen)
    zeros = torch.zeros(batch_size, num_joints, 4, 3, dtype=joints.dtype, device=joints.device)
    init_bone = torch.cat([zeros, init_bone], dim=3)
    rel_transforms = transforms - init_bone
    return posed_joints, rel_transforms


class MANO(nn.Module):
    """Minimal MANO layer for WiLoR inference, with no smplx dependency.

    Supports the subset used by this project: betas, global_orient, hand_pose,
    and pose2rot=False rotation-matrix inference. Axis-angle input is also
    handled for completeness.
    """

    def __init__(
        self,
        model_path: str,
        gender: str = "neutral",
        num_hand_joints: int = 15,
        create_body_pose: bool = False,
        joint_regressor_extra: Optional[str] = None,
        hand_side: str = "right",
        **kwargs,
    ):
        super().__init__()
        del gender, create_body_pose, joint_regressor_extra, kwargs
        if num_hand_joints != 15:
            raise ValueError("This MANO port expects 15 hand joints")
        side = str(hand_side).lower()
        if side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
        model_dir = Path(model_path)
        mano_path = model_dir / f"MANO_{side.upper()}.pt"
        if not mano_path.exists():
            raise FileNotFoundError(
                f"{mano_path} not found. Download MANO from https://mano.is.tue.mpg.de and run "
                f"'python mano_pkl_to_pt.py --pkl .../MANO_{side.upper()}.pkl --out {mano_path}' (see README.md, MANO setup)."
            )
        data = torch.load(mano_path, map_location="cpu", weights_only=True)

        self.register_buffer("faces_tensor", _tensor_from_asset(data["f"], torch.long), persistent=False)
        self.faces = self.faces_tensor.detach().cpu()
        self.register_buffer("v_template", _tensor_from_asset(data["v_template"], torch.float32), persistent=False)
        self.register_buffer("shapedirs", _tensor_from_asset(data["shapedirs"], torch.float32), persistent=False)
        self.register_buffer("posedirs", _tensor_from_asset(data["posedirs"], torch.float32), persistent=False)
        self.register_buffer("J_regressor", _tensor_from_asset(data["J_regressor"], torch.float32), persistent=False)
        self.register_buffer("lbs_weights", _tensor_from_asset(data["weights"], torch.float32), persistent=False)
        parents = _tensor_from_asset(data["parents"], torch.long)
        parents[0] = -1
        self.register_buffer("parents", parents, persistent=False)
        self.register_buffer("extra_joints_idxs", torch.tensor(_MANO_TIP_VERTEX_IDS, dtype=torch.long), persistent=False)
        self.register_buffer("joint_map", torch.tensor(_MANO_TO_OPENPOSE, dtype=torch.long), persistent=False)

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        hand_pose: Optional[torch.Tensor] = None,
        pose2rot: bool = True,
        **kwargs,
    ) -> MANOOutput:
        del kwargs
        batch_size = 1
        device = self.v_template.device
        dtype = self.v_template.dtype
        for x in (betas, global_orient, hand_pose):
            if x is not None:
                batch_size = x.shape[0]
                device = x.device
                dtype = x.dtype
                break
        if betas is None:
            betas = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        if global_orient is None:
            global_orient = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3).expand(batch_size, -1, -1, -1)
        if hand_pose is None:
            hand_pose = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3).expand(batch_size, 15, -1, -1)

        if pose2rot:
            if global_orient.shape[-1] == 3 and global_orient.ndim == 2:
                global_orient = aa_to_rotmat(global_orient.reshape(-1, 3)).reshape(batch_size, 1, 3, 3)
            if hand_pose.shape[-1] == 3 and hand_pose.ndim in (2, 3):
                hand_pose = aa_to_rotmat(hand_pose.reshape(-1, 3)).reshape(batch_size, 15, 3, 3)
        rot_mats = torch.cat([global_orient.reshape(batch_size, 1, 3, 3), hand_pose.reshape(batch_size, 15, 3, 3)], dim=1)

        v_template = self.v_template.to(device=device, dtype=dtype)
        shapedirs = self.shapedirs.to(device=device, dtype=dtype)
        posedirs = self.posedirs.to(device=device, dtype=dtype)
        j_regressor = self.J_regressor.to(device=device, dtype=dtype)
        lbs_weights = self.lbs_weights.to(device=device, dtype=dtype)
        parents = self.parents.to(device=device)

        v_shaped = v_template.unsqueeze(0) + torch.einsum("bl,vcl->bvc", betas, shapedirs)
        joints = torch.einsum("jv,bvc->bjc", j_regressor, v_shaped)
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = torch.einsum("bp,vcp->bvc", pose_feature, posedirs)
        v_posed = v_shaped + pose_offsets

        joints_transformed, rel_transforms = _batch_rigid_transform(rot_mats, joints, parents)
        transform = torch.einsum("vj,bjkl->bvkl", lbs_weights, rel_transforms)
        v_homo = torch.cat([v_posed, torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device)], dim=2).unsqueeze(-1)
        vertices = torch.matmul(transform, v_homo)[:, :, :3, 0]

        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(device=device))
        all_joints = torch.cat([joints_transformed, extra_joints], dim=1)
        all_joints = all_joints[:, self.joint_map.to(device=device), :]
        return MANOOutput(vertices=vertices, joints=all_joints, betas=betas, global_orient=global_orient, hand_pose=hand_pose)
