# disk_latent_nodes.py
# ComfyUI Nodes: Disk-cached LATENTs (save / stitch / load) without blowing RAM/VRAM
#
# What you get:
#   1) LatentToDiskCache  : Save a LATENT (tensor or dict with tensors) to disk in chunks, free memory, return a handle.
#   2) DiskLatentConcat   : Stitch multiple disk-cached latents (metadata-only) along a chosen dimension.
#   3) DiskLatentLoad     : Load a (possibly stitched) disk-cached latent back to LATENT, streaming in chunks.
#
# Custom type: "DISK_LATENT" — a tiny Python handle that points to a manifest.json on disk.
#
# Supported inputs:
#   - A raw torch.Tensor (any shape)
#   - A ComfyUI LATENT dict (commonly {"samples": (B,C,H,W), ...}), with any number of tensor fields
#
# Notes:
#   - Chunking is along dim=0 by default (configurable).
#   - Non-tensor fields (e.g., floats/ints/strings) are copied into manifest metadata.
#   - All saves occur on CPU to free GPU memory quickly.
#   - Loading can stream with max_chunk_elems to keep memory pressure low.

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# ======================================================================================
# Utilities
# ======================================================================================

DISK_ROOT_ENV = "MAGIX_DISK_LATENT_ROOT"  # optional env override

def _root_dir() -> Path:
    base = os.environ.get(DISK_ROOT_ENV)
    if base:
        root = Path(base)
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path(tempfile.gettempdir()) / "magix_disk_latents"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _new_dir(prefix: str = "disk_latent_") -> Path:
    d = _root_dir() / f"{prefix}{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _cleanup_mem():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass
    for _ in range(3):
        gc.collect()

def _is_tensorlike(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def _to_cpu_detach(x: torch.Tensor) -> torch.Tensor:
    try:
        return x.detach().to("cpu", copy=False)
    except Exception:
        return x.detach().cpu()

def _elem_count(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ======================================================================================
# Handle + manifest
# ======================================================================================

@dataclass
class DiskLatentHandle:
    manifest_path: str

    @property
    def manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def root(self) -> Path:
        return Path(self.manifest_path).parent

    def __repr__(self) -> str:
        m = self.manifest
        return f"<DiskLatentHandle kind={m.get('kind')} fields={list(m.get('fields', {}).keys())} path={self.manifest_path}>"


def _write_manifest(root: Path, data: Dict[str, Any]) -> DiskLatentHandle:
    man = root / "manifest.json"
    with open(man, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return DiskLatentHandle(str(man))


# ======================================================================================
# Saving (chunked)
# ======================================================================================

def _chunk_slices(length: int, dim: int, max_chunk_elems: int, shape: Tuple[int, ...]) -> List[slice]:
    """
    Build slice objects along `dim` so that each chunk has <= max_chunk_elems elements.
    """
    if max_chunk_elems <= 0:
        # single chunk
        return [slice(0, length)]
    # elems per unit step along dim
    elems_per_step = _elem_count(shape) // int(shape[dim])
    steps = max(1, max_chunk_elems // max(1, elems_per_step))
    steps = max(1, steps)
    slices = []
    start = 0
    while start < length:
        end = min(length, start + steps)
        slices.append(slice(start, end))
        start = end
    return slices


def _save_tensor_chunks(t: torch.Tensor, base_dir: Path, field: str, dim: int, max_chunk_elems: int) -> List[Dict[str, Any]]:
    """
    Save tensor `t` in chunks along dimension `dim`. Returns a list of chunk metadata dicts.
    """
    t = _to_cpu_detach(t).contiguous()
    base = base_dir / field
    base.mkdir(parents=True, exist_ok=True)

    shape = tuple(int(s) for s in t.shape)
    length = shape[dim]
    parts: List[Dict[str, Any]] = []

    for idx, sl in enumerate(_chunk_slices(length, dim, max_chunk_elems, shape)):
        # Build full slicing tuple
        st = [slice(None)] * t.ndim
        st[dim] = sl
        chunk = t[tuple(st)].contiguous()  # (chunk_len, ...)
        fname = base / f"{idx:06d}.pt"
        torch.save(chunk, fname)
        parts.append({
            "file": os.path.relpath(fname, start=base_dir),
            "slice": [sl.start, sl.stop],
            "shape": list(chunk.shape),
        })
        del chunk
    return parts


def _save_latent_any(latent: Union[torch.Tensor, Dict[str, Any]], out_dir: Path, chunk_dim: int, max_chunk_elems: int, tag: str) -> DiskLatentHandle:
    """
    Save either a raw tensor or a dict of tensors.
    """
    fields: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    if _is_tensorlike(latent):
        t: torch.Tensor = latent
        shape = tuple(int(s) for s in t.shape)
        parts = _save_tensor_chunks(t, out_dir, field="__tensor__", dim=chunk_dim, max_chunk_elems=max_chunk_elems)
        fields["__tensor__"] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(shape),
            "dim": int(chunk_dim),
            "parts": parts,
        }
        meta["type"] = "tensor"

    elif isinstance(latent, dict):
        # Save each tensor field; copy non-tensor metadata as-is.
        meta["type"] = "dict"
        for k, v in latent.items():
            if _is_tensorlike(v):
                t: torch.Tensor = v
                shape = tuple(int(s) for s in t.shape)
                parts = _save_tensor_chunks(t, out_dir, field=k, dim=chunk_dim, max_chunk_elems=max_chunk_elems)
                fields[k] = {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(shape),
                    "dim": int(chunk_dim),
                    "parts": parts,
                }
            else:
                # Lightweight, JSON-serializable non-tensor metadata
                try:
                    json.dumps(v)
                    meta.setdefault("extra", {})[k] = v
                except Exception:
                    meta.setdefault("extra", {})[k] = f"<unsupported:{type(v).__name__}>"
    else:
        raise TypeError("LATENT must be a torch.Tensor or dict with tensor fields")

    manifest = {
        "kind": "latent_sequence",
        "version": 1,
        "created": float(torch.cuda.Event(enable_timing=False).elapsed_time if torch.cuda.is_available() else 0.0),
        "tag": tag,
        "root": str(out_dir),
        "fields": fields,   # { field_name: { dtype, shape, dim, parts:[{file, slice, shape}] } }
        "meta": meta,       # { type: "tensor"/"dict", extra?: {} }
    }
    return _write_manifest(out_dir, manifest)


# ======================================================================================
# Stitching (metadata only)
# ======================================================================================

def _merge_manifests(handles: List[DiskLatentHandle], tag: str) -> DiskLatentHandle:
    """
    Create a stitched manifest that references the given handles without loading tensors.
    Requires that the set of tensor field names match across inputs and shapes are compatible
    to concatenate along each field's chunk dim.
    """
    if not handles:
        raise ValueError("No handles to merge.")

    # Use the first as schema
    base_man = handles[0].manifest
    base_fields = base_man.get("fields", {})
    field_names = list(base_fields.keys())

    # Quick validation
    for h in handles[1:]:
        m = h.manifest
        flds = m.get("fields", {})
        if set(flds.keys()) != set(field_names):
            raise RuntimeError("All latents must have the same tensor fields to be stitched.")

        for name in field_names:
            s0 = tuple(base_fields[name]["shape"])
            s1 = tuple(flds[name]["shape"])
            dim0 = int(base_fields[name]["dim"])
            dim1 = int(flds[name]["dim"])
            if dim0 != dim1:
                raise RuntimeError(f"Chunk dim mismatch for field '{name}': {dim0} vs {dim1}")
            # Check other dims equal
            for i, (a, b) in enumerate(zip(s0, s1)):
                if i == dim0:
                    continue
                if int(a) != int(b):
                    raise RuntimeError(f"Field '{name}' incompatible shapes at dim {i}: {a} vs {b}")

    out_dir = _new_dir(prefix="disk_latent_stitched_")
    stitched_fields: Dict[str, Any] = {}

    for name in field_names:
        dim = int(base_fields[name]["dim"])
        dtype = base_fields[name]["dtype"]
        # compute stitched shape
        base_shape = list(base_fields[name]["shape"])
        concat_len = 0
        seg_refs: List[Dict[str, Any]] = []
        for h in handles:
            man = h.manifest
            fld = man["fields"][name]
            concat_len += int(fld["shape"][dim])
            seg_refs.append({
                "ref_manifest": os.path.relpath(h.manifest_path, start=_root_dir()),
                "field": name,
            })
        stitched_shape = list(base_shape)
        stitched_shape[dim] = concat_len
        stitched_fields[name] = {
            "dtype": dtype,
            "shape": stitched_shape,
            "dim": dim,
            "segments": seg_refs,  # segments reference other manifests
        }

    out_manifest = {
        "kind": "latent_stitched",
        "version": 1,
        "created": 0.0,
        "tag": tag,
        "fields": stitched_fields,
        "base_root": str(_root_dir()),
        "meta": {"type": base_man.get("meta", {}).get("type", "dict")},
    }
    return _write_manifest(out_dir, out_manifest)


# ======================================================================================
# Loading (streamed)
# ======================================================================================

def _load_field_sequence(manifest: Dict[str, Any], field: str, max_chunk_elems: int, dtype_override: Optional[str], device: str) -> torch.Tensor:
    """
    Load a single field either from a 'latent_sequence' manifest or by flattening
    a 'latent_stitched' manifest, streaming chunks to keep memory manageable.
    """
    kind = manifest.get("kind")
    if kind not in ("latent_sequence", "latent_stitched"):
        raise RuntimeError("Unsupported manifest kind for loading.")

    dim: int
    target_shape: List[int]
    dtype_str: str

    if kind == "latent_sequence":
        fld = manifest["fields"][field]
        dim = int(fld["dim"])
        target_shape = list(fld["shape"])
        dtype_str = fld["dtype"]
        root = Path(manifest["root"])
        parts = fld["parts"]

        # Allocate final tensor on CPU
        # (We build list of chunks then cat to avoid gigantic preallocation)
        chunks: List[torch.Tensor] = []
        for p in parts:
            f = root / p["file"]
            chunk = torch.load(f, map_location="cpu")
            if dtype_override:
                chunk = chunk.to(getattr(torch, dtype_override))
            chunks.append(chunk)
        out = torch.cat(chunks, dim=dim)
        # Safety reshape to target in case any off-by-one
        out = out.reshape(target_shape)
        if device != "cpu":
            out = out.to(device)
        return out

    else:  # stitched
        fld = manifest["fields"][field]
        dim = int(fld["dim"])
        target_shape = list(fld["shape"])
        dtype_str = fld["dtype"]
        base_root = Path(manifest.get("base_root", _root_dir()))
        segs = fld["segments"]

        chunks: List[torch.Tensor] = []
        for seg in segs:
            man_path = base_root / seg["ref_manifest"]
            with open(man_path, "r", encoding="utf-8") as f:
                child = json.load(f)
            # recurse to load child field (sequence only)
            if child.get("kind") == "latent_sequence":
                cfld = child["fields"][field]
                root = Path(child["root"])
                for p in cfld["parts"]:
                    f = root / p["file"]
                    chunk = torch.load(f, map_location="cpu")
                    if dtype_override:
                        chunk = chunk.to(getattr(torch, dtype_override))
                    chunks.append(chunk)
            else:
                # nested stitched (rare): recurse
                chunk = _load_field_sequence(child, field, max_chunk_elems, dtype_override, device="cpu")
                chunks.append(chunk)
        out = torch.cat(chunks, dim=dim)
        out = out.reshape(target_shape)
        if device != "cpu":
            out = out.to(device)
        return out


# ======================================================================================
# Nodes
# ======================================================================================

class LatentToDiskCache:
    """
    Save a LATENT (tensor or dict with tensors) to disk in chunks, free memory, and return a DISK_LATENT handle.

    Inputs:
      - latent (LATENT) or (TENSOR)     : Any tensor shape or dict with tensor fields (e.g., {"samples": (B,C,H,W)}).
      - chunk_dim (INT)                 : Axis to chunk along (default 0). Must be < rank for all tensor fields.
      - max_chunk_elems (INT)           : Upper bound of elements per chunk (approx). 0 disables (single file).
      - tag (STRING)                    : Optional label.

    Outputs:
      - disk_latent (DISK_LATENT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),  # Comfy will pass dicts here; raw tensors can be wired via TENSOR if needed
                "chunk_dim": ("INT", {"default": 0, "min": 0, "max": 7}),
                "max_chunk_elems": ("INT", {"default": 16_000_000, "min": 0, "max": 1_000_000_000}),
                "tag": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("DISK_LATENT",)
    RETURN_NAMES = ("disk_latent",)
    FUNCTION = "save"
    CATEGORY = "latent/disk"

    def save(self, latent: Union[torch.Tensor, Dict[str, Any]], chunk_dim: int, max_chunk_elems: int, tag: str):
        out_dir = _new_dir(prefix="disk_latent_seq_")
        handle = _save_latent_any(latent, out_dir, chunk_dim=chunk_dim, max_chunk_elems=max_chunk_elems, tag=tag)
        # free source
        try:
            del latent
        except Exception:
            pass
        _cleanup_mem()
        return (handle,)


class DiskLatentConcat:
    """
    Stitch multiple disk-cached latents along their recorded chunk dims (metadata-only).
    All inputs must have the same tensor field names and compatible shapes on non-chunk dims.

    Inputs:
      - a (DISK_LATENT)
      - b (DISK_LATENT)
      - c..f (optional DISK_LATENT)
      - tag (STRING)

    Outputs:
      - disk_latent (DISK_LATENT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("DISK_LATENT",),
                "b": ("DISK_LATENT",),
                "tag": ("STRING", {"default": "stitched"}),
            },
            "optional": {
                "c": ("DISK_LATENT",),
                "d": ("DISK_LATENT",),
                "e": ("DISK_LATENT",),
                "f": ("DISK_LATENT",),
            },
        }

    RETURN_TYPES = ("DISK_LATENT",)
    RETURN_NAMES = ("disk_latent",)
    FUNCTION = "merge"
    CATEGORY = "latent/disk"

    def merge(
        self,
        a: DiskLatentHandle,
        b: DiskLatentHandle,
        tag: str,
        c: Optional[DiskLatentHandle] = None,
        d: Optional[DiskLatentHandle] = None,
        e: Optional[DiskLatentHandle] = None,
        f: Optional[DiskLatentHandle] = None,
    ):
        hs = [a, b] + [x for x in (c, d, e, f) if x is not None]
        out = _merge_manifests(hs, tag=tag)
        _cleanup_mem()
        return (out,)


class DiskLatentLoad:
    """
    Load a disk-cached (or stitched) latent back into a ComfyUI LATENT.

    Inputs:
      - disk_latent (DISK_LATENT)
      - device (CHOICE)              : "cpu" | "cuda"
      - dtype (CHOICE)               : "keep" | "float16" | "bfloat16" | "float32"
      - fields_to_load (STRING)      : Comma-separated tensor fields to load; empty = all.
      - as_dict (BOOL)               : If true, return dict with original field names; otherwise try to return {"samples": tensor} or the single tensor.
      - max_chunk_elems (INT)        : Stream size cap when loading; lower to reduce peak RAM.

    Outputs:
      - latent (LATENT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disk_latent": ("DISK_LATENT",),
                "device": (["cpu", "cuda"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "dtype": (["keep", "float16", "bfloat16", "float32"], {"default": "keep"}),
                "fields_to_load": ("STRING", {"default": ""}),
                "as_dict": ("BOOL", {"default": True}),
                "max_chunk_elems": ("INT", {"default": 16_000_000, "min": 0, "max": 1_000_000_000}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"
    CATEGORY = "latent/disk"

    def load(
        self,
        disk_latent: DiskLatentHandle,
        device: str,
        dtype: str,
        fields_to_load: str,
        as_dict: bool,
        max_chunk_elems: int,
    ):
        man = disk_latent.manifest
        fields = man.get("fields", {})
        if not fields:
            raise RuntimeError("No tensor fields in manifest.")

        # Which fields?
        if fields_to_load.strip():
            wanted = [x.strip() for x in fields_to_load.split(",") if x.strip() in fields]
        else:
            wanted = list(fields.keys())

        dtype_override = None if dtype == "keep" else dtype

        loaded: Dict[str, torch.Tensor] = {}
        for name in wanted:
            loaded[name] = _load_field_sequence(man, name, max_chunk_elems=max_chunk_elems, dtype_override=dtype_override, device=device)

        _cleanup_mem()

        # Shape output
        if as_dict:
            # Recreate a LATENT-like dict; prioritize "samples" if present
            return ({**loaded},)

        # Non-dict mode:
        if "samples" in loaded and len(loaded) == 1:
            return ({"samples": loaded["samples"]},)
        if len(loaded) == 1:
            # Wrap the only field as "samples" to keep downstream happy
            only_name = next(iter(loaded))
            return ({"samples": loaded[only_name]},)
        # Multiple fields but asked not to return dict => still return dict (safest)
        return ({**loaded},)
