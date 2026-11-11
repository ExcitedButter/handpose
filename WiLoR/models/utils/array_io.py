from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import torch


_DTYPE_MAP = {
    "<f4": torch.float32,
    "|f4": torch.float32,
    "<f8": torch.float64,
    "|f8": torch.float64,
    "<i8": torch.int64,
    "|i8": torch.int64,
    "<i4": torch.int32,
    "|i4": torch.int32,
    "<u4": torch.int64,
    "|u4": torch.int64,
}


_ITEMSIZE = {
    "<f4": 4,
    "|f4": 4,
    "<f8": 8,
    "|f8": 8,
    "<i8": 8,
    "|i8": 8,
    "<i4": 4,
    "|i4": 4,
    "<u4": 4,
    "|u4": 4,
}


def _read_npy_tensor(blob: bytes) -> torch.Tensor:
    if not blob.startswith(b"\x93NUMPY"):
        raise ValueError("not an npy file")
    major = blob[6]
    if major == 1:
        header_len = int.from_bytes(blob[8:10], "little")
        offset = 10
    elif major in (2, 3):
        header_len = int.from_bytes(blob[8:12], "little")
        offset = 12
    else:
        raise ValueError(f"unsupported npy version: {major}")
    header = ast.literal_eval(blob[offset:offset + header_len].decode("latin1").strip())
    if header.get("fortran_order"):
        raise ValueError("Fortran-order npy arrays are not supported")
    descr = header["descr"]
    shape = tuple(int(x) for x in header["shape"])
    if descr not in _DTYPE_MAP:
        raise ValueError(f"unsupported npy dtype: {descr}")
    start = offset + header_len
    n_items = 1
    for dim in shape:
        n_items *= dim
    n_bytes = n_items * _ITEMSIZE[descr]
    raw = bytearray(blob[start:start + n_bytes])
    tensor = torch.frombuffer(raw, dtype=_DTYPE_MAP[descr]).clone().reshape(shape)
    if descr == "<u4" or descr == "|u4":
        tensor = tensor.to(torch.int64)
    return tensor


def load_npz_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load simple .npz tensor assets without importing numpy."""
    out: dict[str, torch.Tensor] = {}
    with zipfile.ZipFile(str(path), "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = Path(name).stem
            out[key] = _read_npy_tensor(zf.read(name))
    return out
