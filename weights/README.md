# Model weights

Everything except MANO is downloaded from
https://huggingface.co/datasets/zhicao/EgoForce/tree/main/handpose_algorithm/weights
(see the main README, section 2). Expected layout:

```text
weights/
  wilor/
    anyhand_wilor_nomano.ckpt              AnyHand-WiLoR, MANO tensors removed
    anyhand_model_config_wilor.yaml
  yolo_world/
    yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth
    clip-vit-base-patch32/
      vocab.json
      merges.txt
  mano/
    MANO_RIGHT.pt                          YOU create this (see below)
```

SHA-256 of every file is in `../ASSETS.json`.

| file | origin |
|---|---|
| `wilor/anyhand_wilor_nomano.ckpt` | AnyHand-WiLoR checkpoint with the 13 MANO buffers (`mano.*`) and the training-only discriminator removed. The loader ignores only those two prefixes and requires every other key; results are identical to the original checkpoint. |
| `wilor/anyhand_model_config_wilor.yaml` | Model configuration of that checkpoint. |
| `yolo_world/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth` | YOLO-World v2-M (Objects365 + GoldG pretrain), official AILab-CVC release. |
| `yolo_world/clip-vit-base-patch32/{vocab.json,merges.txt}` | Tokenizer files of `openai/clip-vit-base-patch32`. |

## MANO (required, not included)

MANO may only be obtained from the Max Planck Institute:

1. Register at <https://mano.is.tue.mpg.de> and accept the license.
2. Download **Models & Code** (`mano_v1_2.zip`), unzip, and locate
   `mano_v1_2/models/MANO_RIGHT.pkl`.
3. From the `handpose_algorithm` folder run:

   ```bash
   python mano_pkl_to_pt.py --pkl /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
       --out weights/mano/MANO_RIGHT.pt
   ```

`MANO_RIGHT.pt` is a converted tensor file, not a renamed `.pkl`. `MANO_LEFT`
is not needed (left hands are mirrored right hands).

`mano_mean_params.npz` is not needed: the mean MANO parameters are stored in
the WiLoR checkpoint. If you place the HaMeR/WiLoR `mano_mean_params.npz` in
`weights/mano/`, it is read and then overwritten by the checkpoint values.

To keep weights elsewhere, pass `--weights DIR` or set `HANDPOSE_WEIGHTS=DIR`
(same layout). The runtime loads only local files and makes no network
requests.
