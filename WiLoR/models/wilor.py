from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from WiLoR.models.utils.geometry import perspective_projection
from .backbones import create_backbone
from .heads import RefineNet
from .mano_wrapper import MANO


class WiLoRModel(nn.Module):
    """Inference-only WiLoR port as a plain torch.nn.Module.

    The implementation intentionally drops PyTorch Lightning, training losses,
    optimizers, and renderers from the upstream implementation. The state_dict
    key names match WiLoR-compatible checkpoints, including AnyHand-WiLoR.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        self.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)
        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        # WiLoR's detector/dataset follows the upstream flipped-crop convention:
        # left-hand crops are flipped into a canonical right-hand crop before the
        # network. The model therefore predicts right-hand MANO geometry, and
        # callers mirror final vertices/joints for left hands after inference.
        self.mano = MANO(**mano_cfg, hand_side="right")
        self.register_buffer("initialized", torch.tensor(False), persistent=True)

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["img"]
        batch_size = x.shape[0]

        # Upstream WiLoR crops the square 256x256 hand patch to the ViT's
        # 192x256 aspect by dropping 32px on each side.
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:, :, :, 32:-32])

        device = temp_mano_params["hand_pose"].device
        dtype = temp_mano_params["hand_pose"].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)

        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)
        temp_mano_output = self.mano(**{k: v for k, v in temp_mano_params.items()}, pose2rot=False)
        temp_vertices = temp_mano_output.vertices

        pred_mano_params, pred_cam = self.refine_net(vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length)

        output: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}
        output["pred_cam"] = pred_cam
        output["pred_mano_params"] = {k: v.clone() for k, v in pred_mano_params.items()}

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length

        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v for k, v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices

        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)
        output["pred_keypoints_2d"] = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
        ).reshape(batch_size, -1, 2)
        return output  # type: ignore[return-value]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_step(batch)


# Backward-compatible alias for code that expects `WiLoR`.
WiLoR = WiLoRModel
