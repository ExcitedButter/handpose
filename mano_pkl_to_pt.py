#!/usr/bin/env python
"""Convert an authorized MANO ``MANO_RIGHT.pkl`` into the ``MANO_RIGHT.pt`` tensor asset.

The runtime loads the hand model with ``torch.load(..., weights_only=True)``
and reads exactly these keys: ``f, v_template, shapedirs, posedirs, J_regressor,
weights, parents``.

The 2018 MPI release is a Python-2 pickle whose arrays are ``chumpy`` objects, and
chumpy itself no longer installs against NumPy >= 1.24.  Rather than resurrect it,
the unpickler below maps every ``chumpy.*`` class to a stub that just carries the
underlying ndarray.

    python mano_pkl_to_pt.py --pkl /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
        --out weights/mano/MANO_RIGHT.pt

Obtain MANO yourself from https://mano.is.tue.mpg.de (registration and license
acceptance; MANO may not be redistributed).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

KEYS = ("f", "v_template", "shapedirs", "posedirs", "J_regressor", "weights")


class ChumpyStub:
    """Stands in for ``chumpy.Ch``: keeps the pickled state, exposes ``.r``."""

    def __init__(self, *args, **kwargs) -> None:
        self.x = args[0] if args else None

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.x = state

    @property
    def r(self):
        state = self.__dict__
        # plain chumpy.Ch: the array lives in `x`
        for key in ("x", "_x"):
            if state.get(key) is not None:
                return np.asarray(state[key])
        # chumpy reorder expression (MANO stores shapedirs this way):
        #   r = a.r.ravel()[idxs].reshape(preferred_shape)
        if state.get("a") is not None and state.get("idxs") is not None:
            base = np.asarray(unwrap(state["a"])).ravel()
            out = base[np.asarray(state["idxs"]).ravel()]
            shape = state.get("preferred_shape")
            return out.reshape(shape) if shape is not None else out
        raise AttributeError(f"cannot recover an array from chumpy state {sorted(state)}")


class ManoUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".")[0] == "chumpy":
            return ChumpyStub
        return super().find_class(module, name)


def unwrap(value):
    """chumpy stub / scipy sparse / ndarray -> dense ndarray."""
    if hasattr(value, "toarray"):           # scipy sparse (J_regressor)
        value = value.toarray()
    if isinstance(value, ChumpyStub):
        value = value.r
    elif hasattr(value, "r") and not isinstance(value, np.ndarray):
        value = value.r
    array = np.asarray(value)
    if array.dtype == object:               # nested chumpy elements
        flat = [np.asarray(unwrap(x)) for x in array.ravel()]
        array = np.stack(flat).reshape(array.shape + flat[0].shape)
    return array


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with args.pkl.open("rb") as handle:
        raw = ManoUnpickler(handle, encoding="latin1").load()
    print(f"keys in {args.pkl.name}: {sorted(raw)}")

    data = {}
    for key in KEYS:
        if key not in raw:
            raise KeyError(f"{key} missing; present: {sorted(raw)}")
        array = unwrap(raw[key])
        data[key] = torch.as_tensor(array.astype(np.int64 if key == "f" else np.float32))
    parents = unwrap(raw["kintree_table"])[0].astype(np.int64).copy()
    parents[0] = -1                          # the root has no parent
    data["parents"] = torch.as_tensor(parents)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size/2**20:.1f} MiB)")
    for key, tensor in data.items():
        finite = bool(torch.isfinite(tensor.float()).all())
        print(f"   {key:14s} {tuple(tensor.shape)} {tensor.dtype} finite={finite}")


if __name__ == "__main__":
    main()
