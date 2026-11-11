from __future__ import annotations

from pathlib import Path
from typing import Any


from WiLoR.assets import weights_dir

_DEFAULT_CFG = weights_dir() / "wilor" / "anyhand_model_config_wilor.yaml"
_DEFAULT_MANO_DIR = weights_dir() / "mano"


class Cfg(dict):
    """Tiny attribute-access config for WiLoR inference.

    Supports the subset of yacs used by this port: nested attribute access,
    mapping operations (`in`, `get`, `pop`, `dict(cfg.SECTION)`), and no-op
    `freeze`/`defrost` methods for compatibility.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Cfg(value)
        if isinstance(value, list):
            return [Cfg._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def defrost(self) -> None:
        return None

    def freeze(self) -> None:
        return None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"\'')


def _next_content(lines: list[str], start: int) -> tuple[int, str] | None:
    for j in range(start, len(lines)):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#"):
            return len(lines[j]) - len(lines[j].lstrip(" ")), stripped
    return None


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple WiLoR YAML subset without PyYAML/yacs."""
    lines = path.read_text().splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        is_list_item = stripped.startswith("- ")
        while stack and (indent < stack[-1][0] if is_list_item else indent <= stack[-1][0]):
            stack.pop()
        parent = stack[-1][1]

        if is_list_item:
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected list item in {path}:{i + 1}: {line}")
            parent.append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {path}:{i + 1}: {line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            value: Any = _parse_scalar(raw_value)
        else:
            nxt = _next_content(lines, i + 1)
            value = [] if nxt is not None and nxt[1].startswith("- ") else {}
        if not isinstance(parent, dict):
            raise ValueError(f"Unexpected mapping item in {path}:{i + 1}: {line}")
        parent[key] = value
        if isinstance(value, (dict, list)):
            stack.append((indent, value))
    return root


def load_config(cfg_path: str | Path | None = None, mano_dir: str | Path | None = None) -> Cfg:
    """Load WiLoR's YAML config and make it inference-safe for this repo."""
    cfg_path = Path(cfg_path or _DEFAULT_CFG).expanduser().resolve()
    cfg = Cfg(_load_simple_yaml(cfg_path))

    if "vit" in cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in cfg.MODEL:
        assert cfg.MODEL.IMAGE_SIZE == 256, "WiLoR ViT checkpoints expect IMAGE_SIZE=256"
        cfg.MODEL.BBOX_SHAPE = [192, 256]

    # Inference loads the whole WiLoR checkpoint, so don't try to separately
    # load the training-time ViT pretrain path from the YAML.
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")

    mano_dir = Path(mano_dir).expanduser().resolve() if mano_dir is not None else _DEFAULT_MANO_DIR
    cfg.MANO.DATA_DIR = str(mano_dir)
    cfg.MANO.MODEL_PATH = str(mano_dir)
    cfg.MANO.MEAN_PARAMS = str((mano_dir / "mano_mean_params.npz").resolve())
    return cfg


def recursive_to(obj: Any, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to(v, device) for v in obj)
    return obj
