# disk_video_nodes.py
# ComfyUI Nodes: Disk-backed pseudo-tensors for videos (save/merge/export without RAM/VRAM blowups)
#
# Update:
#  - DiskTensorToVideo now auto-numbers outputs to avoid overwriting (e.g., out.mp4, out_001.mp4, …),
#    with a toggle to force overwrite if you really want it.
#  - New node DiskTensorToImages converts a disk-backed tensor back into a regular IMAGES tensor,
#    with stride/resize/limit options to avoid OOM.
#
# Provides:
#  1) ImagesToDiskTensor        : Save an IMAGE tensor (B,H,W,C) to disk, free memory; returns handle.
#  2) DiskTensorMerge           : Metadata-only stitch of multiple disk-backed tensors.
#  3) DiskTensorToVideo         : Export disk-backed tensor to video via ffmpeg (auto-numbering).
#  4) DiskTensorToImages        : Load disk-backed tensor back into (B,H,W,C) in [0,1] RGB.
#
# Custom type: "DISK_TENSOR"
#
# Requirements: pillow, torch, ffmpeg executable in PATH.

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
import numpy as np  # noqa: E402

# =================================================================================================
# Utilities
# =================================================================================================

DISK_ROOT_ENV = "MAGIX_DISK_TENSOR_ROOT"  # optional env override for base temp directory
DEFAULT_IMG_FORMAT = "png"  # "png" or "jpg"


def _now_ts() -> float:
    return time.time()


def _root_dir() -> Path:
    base = os.environ.get(DISK_ROOT_ENV)
    if base:
        root = Path(base)
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path(tempfile.gettempdir()) / "magix_disk_tensors"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_tensor_dir(prefix: str = "disk_tensor_") -> Path:
    root = _root_dir()
    d = root / f"{prefix}{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _aggressive_cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass
    for _ in range(3):
        gc.collect()


def _next_numbered_path(path: Path) -> Path:
    """
    Return the same path if it doesn't exist; otherwise append _001/_002/... before extension.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        cand = parent / f"{stem}_{idx:03d}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


# =================================================================================================
# Disk-backed tensor format (manifest)
# =================================================================================================

@dataclass
class DiskTensorHandle:
    """Lightweight handle that just points to the manifest.json on disk."""
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
        kind = m.get("kind", "?")
        total = m.get("total_frames", "?")
        return f"<DiskTensorHandle kind={kind} frames={total} path={self.manifest_path}>"


def _write_manifest(root: Path, data: Dict[str, Any]) -> DiskTensorHandle:
    man = root / "manifest.json"
    with open(man, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return DiskTensorHandle(str(man))


def _save_images_sequence(
    images: torch.Tensor,
    out_dir: Path,
    fmt: str = DEFAULT_IMG_FORMAT,
    quality: int = 90,
) -> Tuple[int, Tuple[int, int]]:
    """
    Save IMAGE tensor (B,H,W,C) in [0,1] RGB to disk as numbered frames starting at 00000001.
    Returns (num_frames, (height, width)).
    """
    assert images.ndim == 4 and images.shape[-1] >= 3, "images must be (B,H,W,C>=3)"
    B, H, W, C = images.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(B):
        frame = images[i, :, :, :3].detach().to("cpu").clamp(0.0, 1.0)
        arr = (frame * 255.0 + 0.5).to(torch.uint8).numpy()  # (H,W,3)
        img = Image.fromarray(arr, mode="RGB")
        fname = out_dir / f"{i + 1:08d}.{fmt}"
        if fmt.lower() in ("jpg", "jpeg"):
            img.save(fname, "JPEG", quality=int(quality), subsampling=1, optimize=False)
        else:
            img.save(fname, "PNG", compress_level=3)
        del img
    _aggressive_cleanup()
    return B, (H, W)


# =================================================================================================
# Node 1: ImagesToDiskTensor
# =================================================================================================

class ImagesToDiskTensor:
    """
    Save an IMAGE tensor (B,H,W,C in [0,1] RGB) to a temp location on disk,
    freeing RAM/VRAM, and return a pseudo-tensor handle ("DISK_TENSOR").

    Frames are stored as numbered images (00000001.png, ...). A manifest.json
    describes shape, dtype, format, and frame pattern.

    Inputs:
      - images (IMAGE)           : (B,H,W,C) float in [0,1], RGB.
      - image_format (CHOICE)    : "png" | "jpg"
      - jpeg_quality (INT)       : 1..100 (only used if image_format="jpg")
      - tag (STRING)             : Optional label stored in manifest.

    Outputs:
      - disk_tensor (DISK_TENSOR): Handle referencing the on-disk frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "image_format": (["png", "jpg"], {"default": DEFAULT_IMG_FORMAT}),
                "jpeg_quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "tag": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("DISK_TENSOR",)
    RETURN_NAMES = ("disk_tensor",)
    FUNCTION = "save"
    CATEGORY = "video/disk"

    def save(self, images: torch.Tensor, image_format: str, jpeg_quality: int, tag: str):
        assert images.ndim == 4 and images.shape[-1] >= 3, "images must be (B,H,W,C>=3)"
        out_dir = _new_tensor_dir()
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        n, (H, W) = _save_images_sequence(images, frames_dir, fmt=image_format, quality=jpeg_quality)

        manifest = {
            "kind": "sequence",
            "version": 1,
            "created": _now_ts(),
            "tag": tag,
            "frame_dir": "frames",
            "frame_ext": image_format,
            "start_number": 1,
            "digits": 8,
            "height": H,
            "width": W,
            "channels": 3,
            "total_frames": n,
            "segments": [
                {"type": "sequence", "relative_dir": "frames", "count": n, "start": 1}
            ],
        }
        handle = _write_manifest(out_dir, manifest)

        # Free source tensor aggressively
        del images
        _aggressive_cleanup()

        return (handle,)


# =================================================================================================
# Node 2: DiskTensorMerge
# =================================================================================================

class DiskTensorMerge:
    """
    Merge multiple disk-backed tensors by stitching metadata only (no frame loading/copying).
    The output manifest references all input segments in order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_a": ("DISK_TENSOR",),
                "tensor_b": ("DISK_TENSOR",),
                "tag": ("STRING", {"default": "stitched"}),
            },
            "optional": {
                "tensor_c": ("DISK_TENSOR",),
                "tensor_d": ("DISK_TENSOR",),
                "tensor_e": ("DISK_TENSOR",),
                "tensor_f": ("DISK_TENSOR",),
            },
        }

    RETURN_TYPES = ("DISK_TENSOR",)
    RETURN_NAMES = ("disk_tensor",)
    FUNCTION = "merge"
    CATEGORY = "video/disk"

    def merge(
        self,
        tensor_a: DiskTensorHandle,
        tensor_b: DiskTensorHandle,
        tag: str,
        tensor_c: Optional[DiskTensorHandle] = None,
        tensor_d: Optional[DiskTensorHandle] = None,
        tensor_e: Optional[DiskTensorHandle] = None,
        tensor_f: Optional[DiskTensorHandle] = None,
    ):
        inputs = [tensor_a, tensor_b] + [t for t in [tensor_c, tensor_d, tensor_e, tensor_f] if t is not None]

        segs: List[Dict[str, Any]] = []
        total = 0
        H = W = None

        for h in inputs:
            man = h.manifest
            if H is None or W is None:
                H, W = man.get("height"), man.get("width")

            if man.get("kind") == "sequence":
                segs.append({
                    "type": "ref_sequence",
                    "ref_manifest": os.path.relpath(h.manifest_path, start=_root_dir()),
                    "height": man.get("height"),
                    "width": man.get("width"),
                    "count": man.get("total_frames"),
                })
                total += int(man.get("total_frames", 0))
            else:
                child_segs = man.get("segments", [])
                segs.append({
                    "type": "ref_stitched",
                    "ref_manifest": os.path.relpath(h.manifest_path, start=_root_dir()),
                    "segments": child_segs,
                    "height": man.get("height"),
                    "width": man.get("width"),
                    "count": man.get("total_frames"),
                })
                total += int(man.get("total_frames", 0))

        out_dir = _new_tensor_dir(prefix="disk_tensor_merged_")
        manifest = {
            "kind": "stitched",
            "version": 1,
            "created": _now_ts(),
            "tag": tag,
            "height": H,
            "width": W,
            "channels": 3,
            "total_frames": total,
            "segments": segs,
            "base_root": str(_root_dir()),
        }
        handle = _write_manifest(out_dir, manifest)
        _aggressive_cleanup()
        return (handle,)


# =================================================================================================
# Helpers for exporting
# =================================================================================================

def _list_segment_frames(segment_manifest: Dict[str, Any], base_root: Path) -> List[Path]:
    seg_type = segment_manifest.get("type")

    if seg_type in ("sequence", "ref_sequence"):
        if seg_type == "sequence":
            raise RuntimeError("Internal error: local 'sequence' segments are expanded from top-level only.")
        else:
            ref_manifest_rel = segment_manifest["ref_manifest"]
            man_path = base_root / ref_manifest_rel
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if man.get("kind") != "sequence":
                raise RuntimeError("ref_sequence points to non-sequence manifest")
            frame_dir = (man_path.parent / man["frame_dir"]).resolve()
            ext = man.get("frame_ext", "png")
            count = int(man["total_frames"])
            digits = int(man.get("digits", 8))
            start = int(man.get("start_number", 1))
            return [frame_dir / f"{i:0{digits}d}.{ext}" for i in range(start, start + count)]

    elif seg_type == "ref_stitched":
        ref_manifest_rel = segment_manifest["ref_manifest"]
        man_path = base_root / ref_manifest_rel
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        base = Path(man_path).parent
        child_segs = man.get("segments", [])
        flat: List[Path] = []
        for s in child_segs:
            s_type = s.get("type")
            if s_type == "sequence":
                frame_dir = (base / s["relative_dir"]).resolve()
                ext = man.get("frame_ext", "png")
                count = int(s["count"])
                digits = int(man.get("digits", 8))
                start = int(s.get("start", 1))
                flat.extend([frame_dir / f"{i:0{digits}d}.{ext}" for i in range(start, start + count)])
            else:
                flat.extend(_list_segment_frames(s, base_root))
        return flat

    else:
        raise RuntimeError(f"Unknown segment type: {seg_type}")


def _collect_all_frames(handle: DiskTensorHandle) -> List[Path]:
    man = handle.manifest
    kind = man.get("kind")
    if kind == "sequence":
        frame_dir = (Path(handle.manifest_path).parent / man["frame_dir"]).resolve()
        ext = man.get("frame_ext", "png")
        digits = int(man.get("digits", 8))
        start = int(man.get("start_number", 1))
        count = int(man.get("total_frames", 0))
        return [frame_dir / f"{i:0{digits}d}.{ext}" for i in range(start, start + count)]
    elif kind == "stitched":
        base_root = Path(man.get("base_root", _root_dir()))
        flat: List[Path] = []
        for seg in man.get("segments", []):
            flat.extend(_list_segment_frames(seg, base_root))
        return flat
    else:
        raise RuntimeError(f"Unsupported manifest kind: {kind}")


def _ensure_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH.")
    return ff


# =================================================================================================
# Node 3: DiskTensorToVideo (auto-numbering)
# =================================================================================================

class DiskTensorToVideo:
    """
    Export a disk-backed (possibly merged) pseudo-tensor to a video using ffmpeg.

    Auto-numbering:
      - If overwrite=False (default) and the output file exists, a suffix _001, _002, ...
        is appended automatically to avoid overwriting (e.g., out.mp4 → out_001.mp4).

    Inputs:
      - disk_tensor (DISK_TENSOR)
      - output_path (STRING)         : e.g., /path/to/out.mp4 (container inferred from extension)
      - fps (FLOAT)                  : output framerate
      - codec (CHOICE)               : h264 | hevc | prores | vp9 | av1
      - crf (INT)                    : quality factor (ignored for some codecs like prores)
      - pix_fmt (CHOICE)             : yuv420p | yuv422p10le | yuv444p | auto
      - overwrite (BOOL)             : if True, do not auto-number; force overwrite

    Outputs:
      - out_path (STRING)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disk_tensor": ("DISK_TENSOR",),
                "output_path": ("STRING", {"default": "output.mp4"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "codec": (["h264", "hevc", "prores", "vp9", "av1"], {"default": "h264"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "pix_fmt": (["auto", "yuv420p", "yuv422p10le", "yuv444p"], {"default": "yuv420p"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_path",)
    FUNCTION = "export"
    CATEGORY = "video/disk"

    def _codec_args(self, codec: str, crf: int) -> List[str]:
        if codec == "h264":
            return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
        if codec == "hevc":
            return ["-c:v", "libx265", "-preset", "medium", "-crf", str(crf)]
        if codec == "prores":
            profile = 2 if crf >= 28 else 3 if crf >= 22 else 4
            return ["-c:v", "prores_ks", "-profile:v", str(profile)]
        if codec == "vp9":
            return ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf)]
        if codec == "av1":
            return ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0"]
        return []

    def export(
        self,
        disk_tensor: DiskTensorHandle,
        output_path: str,
        fps: float,
        codec: str,
        crf: int,
        pix_fmt: str,
        overwrite: bool,
    ):
        ffmpeg = _ensure_ffmpeg()

        # Resolve frames
        frames = _collect_all_frames(disk_tensor)
        if not frames:
            raise RuntimeError("No frames found to export.")

        # Staging with continuous numbering via links
        staging = _new_tensor_dir(prefix="staging_")
        seq_dir = staging / "seq"
        seq_dir.mkdir(parents=True, exist_ok=True)

        use_symlink = hasattr(os, "symlink")
        for idx, src in enumerate(frames, start=1):
            dst = seq_dir / f"{idx:08d}{src.suffix.lower()}"
            try:
                os.link(src, dst)
            except Exception:
                if use_symlink:
                    try:
                        os.symlink(src, dst)
                    except Exception:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)

        # Decide final output path (auto-number unless overwrite=True)
        out_path = Path(output_path)
        if not overwrite:
            out_path = _next_numbered_path(out_path)

        # Build ffmpeg command
        first_ext = frames[0].suffix.lower().lstrip(".")
        pattern = str(seq_dir / f"%08d.{first_ext}")
        pix_fmt_args = [] if pix_fmt == "auto" else ["-pix_fmt", pix_fmt]
        overwrite_flag = ["-y"] if overwrite else []  # we already auto-numbered when overwrite=False
        vcodec_args = self._codec_args(codec, crf)

        movflags = []
        if out_path.suffix.lower() in (".mp4", ".mov"):
            movflags = ["-movflags", "+faststart"]

        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-r", f"{fps}",
            "-pattern_type", "sequence",
            "-start_number", "1",
            "-i", pattern,
            *vcodec_args,
            *pix_fmt_args,
            *movflags,
            str(out_path),
            *overwrite_flag,
        ]

        try:
            subprocess.run([c for c in cmd if c != ""], check=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            _aggressive_cleanup()

        return (str(out_path.resolve()),)


# =================================================================================================
# Node 4: DiskTensorToImages
# =================================================================================================

class DiskTensorToImages:
    """
    Convert a disk-backed tensor back into a regular IMAGES tensor (B,H,W,C) in [0,1] RGB.

    To avoid OOM, you can stride/limit/resize:
      - frame_stride: take every N-th frame (default 1 = all)
      - max_frames: if >0, uniformly sample down to this count
      - resize_shorter_to: if >0, resize frames so the shorter side matches this value

    Inputs:
      - disk_tensor (DISK_TENSOR)
      - frame_stride (INT ≥1)
      - max_frames (INT ≥0)
      - resize_shorter_to (INT ≥0)

    Outputs:
      - images (IMAGE)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disk_tensor": ("DISK_TENSOR",),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 256}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "resize_shorter_to": ("INT", {"default": 0, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load"
    CATEGORY = "video/disk"

    def _sample_indices(self, total: int, stride: int, max_frames: int) -> List[int]:
        idxs = list(range(0, total, max(1, stride)))
        if max_frames and len(idxs) > max_frames:
            # uniform sampling
            import numpy as np
            pos = np.linspace(0, len(idxs) - 1, num=max_frames)
            idxs = [idxs[int(round(p))] for p in pos]
        return idxs

    def _resize_keep_aspect_pil(self, img: Image.Image, shorter_to: int) -> Image.Image:
        if shorter_to <= 0:
            return img
        w, h = img.size
        short = min(w, h)
        if short == 0 or short == shorter_to:
            return img
        scale = shorter_to / float(short)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.BICUBIC)

    def load(
        self,
        disk_tensor: DiskTensorHandle,
        frame_stride: int,
        max_frames: int,
        resize_shorter_to: int,
    ):
        # Resolve frame file paths
        frames = _collect_all_frames(disk_tensor)
        if not frames:
            # Return empty IMAGE tensor
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32),)

        # Choose indices
        idxs = self._sample_indices(len(frames), frame_stride, max_frames)
        sel_paths = [frames[i] for i in idxs]

        # Load to tensor batch
        batch: List[torch.Tensor] = []
        for p in sel_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = self._resize_keep_aspect_pil(im, resize_shorter_to)
                t = torch.from_numpy(np.asarray(im)).to(torch.float32) / 255.0  # (H,W,3)
                batch.append(t.unsqueeze(0))
        if not batch:
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32),)

        images = torch.cat(batch, dim=0).contiguous()  # (B,H,W,3)
        _aggressive_cleanup()
        return (images,)


# =================================================================================================
# ComfyUI registry
# =================================================================================================

CUSTOM_TYPE_NAME = "DISK_TENSOR"
