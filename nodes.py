from __future__ import annotations
"""
ComfyUI Nodes Pack — Video Tools & Qwen2-VL

Overview
--------
A unified pack of utility and AI nodes for video workflows:
- Fast scene/shot detection with optional seek-only extraction
- Post-process jitter/boil effect for anime-style motion
- Qwen2-VL model loader + captioning for stills or batched video frames

Tensor Conventions
------------------
- IMAGE tensors use shape (B, H, W, C), RGB channels, float in [0, 1].
- Unless stated otherwise, dtype is preserved across nodes.

Nodes
-----

1) VideoSceneExtractor
   Detect scenes and return the N-th scene’s frames as an IMAGE tensor.
   If no detections are provided, scans once to compute them; then streams again to extract.

   Inputs:
     - video_path (STRING)             : Path to a readable video file.
     - scene_index (INT, 0-based)      : Which scene to extract.
     - threshold (FLOAT)               : Cut sensitivity (higher ⇒ fewer cuts).
     - min_scene_len (INT)             : Minimum frames per scene.
     - downscale_factor (INT ≥1)       : Detection runs on downscaled luma for speed.
     - resize_shorter_to (INT ≥0)      : If >0, resize extracted frames so the shorter side matches.
     - max_frames (INT ≥0)             : If >0, cap number of frames returned.
     - detections (STRING, optional)   : JSON from a prior run to skip re-detection.

   Outputs:
     - images (IMAGE)                  : Selected scene as (B,H,W,C) in [0,1].
     - detections (STRING)             : JSON with metadata and scene list.

   Notes:
     - Uses adaptive luma diffs with rolling MAD to find cuts.
     - Streams from frame 0 during extraction (simple & robust).

   Category: video/io


2) VideoSceneExtractorSeek
   Seek-optimized variant that decodes ONLY the requested scene range.
   If valid detections are provided, it seeks directly to the scene start and reads [start, end).

   Inputs:
     - video_path (STRING)             : Path to a readable video file.
     - scene_index (INT, 0-based)      : Which scene to extract.
     - threshold (FLOAT)               : Cut sensitivity (higher ⇒ fewer cuts).
     - min_scene_len (INT)             : Minimum frames per scene.
     - downscale_factor (INT ≥1)       : Detection runs on downscaled luma for speed.
     - resize_shorter_to (INT ≥0)      : If >0, resize extracted frames so the shorter side matches.
     - max_frames (INT ≥0)             : If >0, cap number of frames returned.
     - detections (STRING, optional)   : JSON from a prior run; if absent, one detection pass is performed.

   Outputs:
     - images (IMAGE)                  : Selected scene as (B,H,W,C) in [0,1].
     - detections (STRING)             : JSON with metadata and scene list.

   Notes:
     - Seeking accuracy depends on codec/indexes; node guards by skipping up to start after seek.

   Category: video/io


3) AnimeJitter
   Adds subtle anime-style boil/jitter: tiny per-frame translate/rotate/scale,
   optional per-channel micro-shift (chromatic aberration), and light grain.
   Intensity scales with a single “strength” control.

   Inputs:
     - image (IMAGE)                   : (B,H,W,C) in [0,1].
     - translate_px (FLOAT)            : Max absolute pixel translation.
     - rotate_deg (FLOAT)              : Max absolute rotation in degrees.
     - scale_jitter (FLOAT)            : Max absolute per-axis scale jitter around 1.0.
     - chroma_px (FLOAT)               : Per-channel micro-shift in pixels (0 disables).
     - grain (FLOAT)                   : Film-grain std-dev in 0..1.
     - strength (FLOAT 0..1)           : Global intensity scaler.
     - seed (INT)                      : Deterministic randomization (-1 = random).

   Optional:
     - sample_mode (CHOICE)            : "bilinear" | "nearest".
     - pad_mode (CHOICE)               : "border" | "reflection" | "zeros".

   Outputs:
     - image (IMAGE)                   : Jittered image(s), dtype preserved.

   Category: image/animation


4) Qwen2VLLoader
   Loads Qwen/Qwen2-VL-2B-Instruct and its processor; returns a reusable handle.

   Inputs:
     - model_id (STRING)               : HF ID (default: "Qwen/Qwen2-VL-2B-Instruct").
     - device_map (CHOICE)             : "auto" | "cuda" | "cpu".
     - dtype (CHOICE)                  : "auto" | "float16" | "bfloat16" | "float32".

   Outputs:
     - qwen2vl (QWEN2VL)               : Handle containing model & processor.

   Notes:
     - Performs light VRAM cleanup before load.
     - Model is set to eval().

   Category: ai/qwen2vl


5) Qwen2VLCaption
   Captions a still image (B=1) or a batched frame sequence (B>1) from an IMAGE tensor.

   Inputs:
     - qwen2vl (QWEN2VL)               : Handle from Qwen2VLLoader.
     - images (IMAGE)                  : (B,H,W,C) RGB in [0,1]; B>1 treated as video frames.
     - system_prompt (STRING)          : Instruction to the model.
     - max_new_tokens (INT)            : Generation length cap.
     - max_frames (INT)                : If >0 and B>max_frames, uniformly sample to this many.
     - frame_stride (INT ≥1)           : Take every N-th frame before max_frames sampling.
     - use_chat_template (BOOL)        : If true, wraps prompt via chat template.

   Outputs:
     - caption (STRING)                : Generated description.

   Notes:
     - Converts tensors to PIL for the processor (video as list of frames).
     - Best-effort fallback for processor kwargs across versions.

   Category: ai/qwen2vl


Requirements
------------
- Video nodes: OpenCV (cv2), ffmpeg recommended for external preprocessing.
- Qwen nodes: transformers ≥ 4.41, pillow, torch (CUDA optional but recommended).
- All nodes assume RGB images and preserve value range [0,1].
"""

import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import cv2
except Exception:
    cv2 = None

import torch
import torch.nn.functional as F

import gc
from PIL import Image

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from .video import ImagesToDiskTensor, DiskTensorMerge, DiskTensorToVideo, DiskTensorToImages


# ---- helpers ----

def _get_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    if seed is None or seed < 0:
        # make a reasonably random seed
        seed = torch.seed() % (2**63 - 1)
    g.manual_seed(int(seed))
    return g

def _to_nchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    # (B, H, W, C) -> (B, C, H, W)
    return img_bhwc.permute(0, 3, 1, 2).contiguous()

def _to_bhwc(img_bchw: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) -> (B, H, W, C)
    return img_bchw.permute(0, 2, 3, 1).contiguous()

def _build_affine_batch(
    B: int,
    H: int,
    W: int,
    translate_px: torch.Tensor,
    rotate_deg: torch.Tensor,
    scale_jit: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns theta of shape (B, 2, 3) for grid_sample.
    translate_px: (B,2) pixels
    rotate_deg: (B,) degrees
    scale_jit: (B,2) multiplicative jitter around 1.0 (already applied)
    """
    # Convert to normalized translation in [-1,1]
    # tx_norm maps pixels to normalized coords along X (width), ty along Y (height).
    tx = (2.0 * translate_px[:, 0] / max(W - 1, 1)).to(dtype)
    ty = (2.0 * translate_px[:, 1] / max(H - 1, 1)).to(dtype)

    # Rotation
    theta_rad = (rotate_deg * 3.141592653589793 / 180.0).to(dtype)
    cos_t = torch.cos(theta_rad)
    sin_t = torch.sin(theta_rad)

    # Per-axis scale
    sx = scale_jit[:, 0].to(dtype)
    sy = scale_jit[:, 1].to(dtype)

    # Compose scale * rotation
    a11 = sx * cos_t
    a12 = -sx * sin_t
    a21 = sy * sin_t
    a22 =  sy * cos_t

    theta = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = a11
    theta[:, 0, 1] = a12
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = a21
    theta[:, 1, 1] = a22
    theta[:, 1, 2] = ty
    return theta

def _grid_sample(img_bchw: torch.Tensor, theta: torch.Tensor, mode: str, pad_mode: str) -> torch.Tensor:
    B, C, H, W = img_bchw.shape
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
    return F.grid_sample(
        img_bchw, grid, mode=mode, padding_mode=pad_mode, align_corners=False
    )

# --------------------------- Utilities ---------------------------

def _tensor_to_pil_list(images: torch.Tensor, indices: Optional[List[int]] = None) -> List[Image.Image]:
    """
    Convert IMAGE tensor (B,H,W,C) float in [0,1] RGB to a list of PIL Images.
    Optionally subset by indices.
    """
    assert images.ndim == 4, "IMAGE must be (B,H,W,C)"
    B, H, W, C = images.shape
    assert C >= 3, "IMAGE must have at least 3 channels (RGB)"
    imgs = images[..., :3].detach().cpu().clamp(0.0, 1.0)
    if indices is None:
        batch = imgs
    else:
        sel = torch.tensor(indices, dtype=torch.long)
        batch = imgs.index_select(0, sel)
    # (B,H,W,3) -> PIL list
    batch = (batch * 255.0 + 0.5).to(torch.uint8).numpy()  # (B,H,W,3) uint8
    pil_list = [Image.fromarray(arr, mode="RGB") for arr in batch]
    return pil_list


def _aggressive_cleanup():
    """Best-effort VRAM/RAM cleanup after generation."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass
    for _ in range(2):
        gc.collect()


# --------------------------- Model handle ---------------------------

class _Qwen2VLHandle:
    """
    Simple container to hold the model & processor.
    Stored as a Python object flowing between ComfyUI nodes.
    """
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device


# ---- Node ----

class AnimeJitter:
    """
    Add subtle anime-like jitter/boil to a batched IMAGE.
    - Tiny per-frame translate/rotate/scale
    - Optional chromatic aberration (per-channel micro-shift)
    - Optional film grain
    All effects scale with 'strength'.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Max absolute pixel translation for the main shake.
                "translate_px": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                # Max absolute rotation in degrees.
                "rotate_deg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 15.0, "step": 0.1}),
                # Max absolute scale jitter (e.g., 0.01 => up to ±1%).
                "scale_jitter": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001}),
                # Per-channel micro-shift (pixels). 0 disables chroma effect.
                "chroma_px": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 8.0, "step": 0.1}),
                # Film grain standard deviation in 0..1 units.
                "grain": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
                # Overall intensity scaler 0..1.
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Deterministic randomization across the batch.
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                # Interp & padding for resampling
                "sample_mode": (["bilinear", "nearest"], {"default": "bilinear"}),
                "pad_mode": (["border", "reflection", "zeros"], {"default": "border"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/animation"

    def apply(
        self,
        image: torch.Tensor,
        translate_px: float,
        rotate_deg: float,
        scale_jitter: float,
        chroma_px: float,
        grain: float,
        strength: float,
        seed: int,
        sample_mode: str = "bilinear",
        pad_mode: str = "border",
    ) -> Tuple[torch.Tensor]:

        if strength <= 0.0:
            return (image,)

        # Grab shape and device
        assert image.ndim == 4, "IMAGE must be a 4D tensor (B, H, W, C)"
        B, H, W, C = image.shape
        device = image.device
        in_dtype = image.dtype

        # Work in float32 internally for transforms/noise, then cast back
        img = image.to(torch.float32)

        g = _get_generator(seed)

        # Scale parameters by strength
        t_px = float(translate_px) * float(strength)
        r_deg = float(rotate_deg) * float(strength)
        s_jit = float(scale_jitter) * float(strength)
        c_px = float(chroma_px) * float(strength)
        g_std = float(grain) * float(strength)

        # ---- Main jitter: translate / rotate / scale ----
        if t_px > 0.0 or r_deg > 0.0 or s_jit > 0.0:
            # random in [-1,1]
            r2 = torch.rand((B, 2), generator=g, device=device) * 2.0 - 1.0
            r1 = torch.rand((B,), generator=g, device=device) * 2.0 - 1.0
            r2_scale = torch.rand((B, 2), generator=g, device=device) * 2.0 - 1.0

            translate = r2 * t_px  # pixels
            rotate = r1 * r_deg    # degrees
            # scale jitter around 1.0 independently per axis
            scale_xy = 1.0 + (r2_scale * s_jit)

            theta = _build_affine_batch(
                B, H, W, translate, rotate, scale_xy, device, torch.float32
            )

            img_bchw = _to_nchw(img)
            img_bchw = _grid_sample(img_bchw, theta, mode=sample_mode, pad_mode=pad_mode)
            img = _to_bhwc(img_bchw)

        # ---- Chromatic aberration: tiny per-channel shift ----
        if c_px > 0.0 and C >= 3:
            # Create independent tiny translations for R,G,B; keep rotation/scale = identity
            # Shift only, no rotate/scale, for crisp micro-fringe
            # (You can still get a slight "boil" because main jitter already did R/G/B together.)
            rgb = []
            for ch in range(3):
                shift = (torch.rand((B, 2), generator=g, device=device) * 2.0 - 1.0) * c_px
                theta = _build_affine_batch(
                    B, H, W,
                    translate_px=shift,
                    rotate_deg=torch.zeros((B,), device=device),
                    scale_jit=torch.ones((B, 2), device=device),
                    device=device,
                    dtype=torch.float32,
                )
                ch_tensor = img[..., ch:ch+1]  # (B,H,W,1)
                ch_bchw = _to_nchw(ch_tensor)
                ch_bchw = _grid_sample(ch_bchw, theta, mode=sample_mode, pad_mode=pad_mode)
                rgb.append(_to_bhwc(ch_bchw))
            # Stack back RGB and keep any extra channels as-is
            img = torch.cat([rgb[0], rgb[1], rgb[2], img[..., 3:]], dim=-1)

        # ---- Grain ----
        if g_std > 0.0:
            noise = torch.randn_like(img) * g_std
            img = img + noise

        # Clamp and cast back
        img = img.clamp(0.0, 1.0).to(in_dtype)

        return (img,)
PRESETS = {
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "red":    (255, 0, 0),
    "green":  (0, 255, 0),
    "blue":   (0, 0, 255),
}

class AlphaToSolidBackground:
    """
    Composite images with alpha over a solid color background.

    Modes:
      - white, black, red, green, blue: use preset color
      - custom: use provided R/G/B INTs (0–255)
      - transparent: pass-through (no compositing)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["white", "black", "transparent", "red", "green", "blue", "custom"], {"default": "white"}),
                "r": ("INT", {"default": 255, "min": 0, "max": 255}),
                "g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "b": ("INT", {"default": 255, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/alpha"

    def apply(
        self,
        image: torch.Tensor,
        mode: str,
        r: int,
        g: int,
        b: int,
    ) -> Tuple[torch.Tensor]:

        # Validate input
        assert image.ndim == 4, "IMAGE must be shape (B, H, W, C)"
        B, H, W, C = image.shape
        assert C >= 3, "IMAGE must have at least 3 channels (RGB)"
        device = image.device
        in_dtype = image.dtype

        # Transparent mode => no-op
        if mode == "transparent":
            return (image,)

        # Determine background color (0..1 float)
        if mode == "custom":
            bg_rgb_255 = (int(r), int(g), int(b))
        else:
            bg_rgb_255 = PRESETS.get(mode, (255, 255, 255))

        bg_rgb = torch.tensor([bg_rgb_255[0] / 255.0,
                               bg_rgb_255[1] / 255.0,
                               bg_rgb_255[2] / 255.0],
                              device=device, dtype=torch.float32)

        # If no alpha channel, pass through unchanged
        if C < 4:
            return (image,)

        # Work in float32 for math, preserve dtype on output
        img = image.to(torch.float32)

        src_rgb = img[..., :3]                  # (B,H,W,3)
        alpha = img[..., 3:4].clamp(0.0, 1.0)   # (B,H,W,1)

        # Expand bg to image shape
        bg = bg_rgb.view(1, 1, 1, 3).expand(B, H, W, 3)

        out_rgb = src_rgb * alpha + bg * (1.0 - alpha)

        # If there were extra channels beyond A (rare), drop them; output RGB result
        out = out_rgb.clamp(0.0, 1.0).to(in_dtype)

        return (out,)



class Qwen2VLLoader:
    """
    Load Qwen/Qwen2-VL-2B-Instruct and return a reusable handle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Qwen/Qwen2-VL-2B-Instruct"}),
                "device_map": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN2VL",)
    RETURN_NAMES = ("qwen2vl",)
    FUNCTION = "load"
    CATEGORY = "ai/qwen2vl"

    def load(self, model_id: str, device_map: str, dtype: str):
        # Translate dtype
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Device map
        if device_map == "auto":
            dm = "auto"
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_map == "cuda":
            dm = {"": 0} if torch.cuda.is_available() else "cpu"
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dm = "cpu"
            device = "cpu"

        _aggressive_cleanup()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=dm,
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id)

        handle = _Qwen2VLHandle(model=model, processor=processor, device=device)
        return (handle,)


class Qwen2VLCaption:
    """
    Caption a still image (B=1) or a video (B>1) from an IMAGE tensor using Qwen2-VL.

    Inputs:
      - qwen2vl: handle from Qwen2VLLoader
      - images: IMAGE tensor (B,H,W,C) in [0,1] RGB
      - system_prompt: Instruction to the model (e.g., "Describe this video.")
      - max_new_tokens: generation length
      - max_frames: if >0 and B>max_frames, uniformly sample to this many frames
      - frame_stride: if >1, take every Nth frame (applied before max_frames sampling)
      - use_chat_template: if true, wrap prompt with chat template
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen2vl": ("QWEN2VL",),
                "images": ("IMAGE",),
                "system_prompt": ("STRING", {"default": "Describe concisely what is happening."}),
                "max_new_tokens": ("INT", {"default": 96, "min": 1, "max": 2048}),
                "max_frames": ("INT", {"default": 16, "min": 0, "max": 512}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 64}),
                "use_chat_template": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "ai/qwen2vl"

    def _sample_indices(self, total: int, frame_stride: int, max_frames: int) -> List[int]:
        # Stride first
        idxs = list(range(0, total, max(1, frame_stride)))
        if max_frames and len(idxs) > max_frames:
            # Uniform downsample to max_frames
            import numpy as np
            sel = np.linspace(0, len(idxs) - 1, num=max_frames)
            idxs = [idxs[int(round(s))] for s in sel]
        return idxs

    def run(
        self,
        qwen2vl: _Qwen2VLHandle,
        images: torch.Tensor,
        system_prompt: str,
        max_new_tokens: int,
        max_frames: int,
        frame_stride: int,
        use_chat_template: bool,
    ):
        assert isinstance(qwen2vl, _Qwen2VLHandle), "Invalid QWEN2VL handle. Load model first."
        assert images.ndim == 4, "IMAGE must be (B,H,W,C)"
        B, H, W, C = images.shape
        assert C >= 3, "IMAGE must have at least 3 channels (RGB)"

        # Determine if this is a still or a video-like batch
        is_video = B > 1

        # Frame sampling
        if is_video:
            indices = self._sample_indices(B, frame_stride, max_frames)
            pil_frames = _tensor_to_pil_list(images, indices)
            pil_image = None
        else:
            pil_frames = None
            pil_image = _tensor_to_pil_list(images)[0]

        processor = qwen2vl.processor
        model = qwen2vl.model

        # Build text input
        if use_chat_template:
            # Minimal chat: system + user
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please describe succinctly."}
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_inputs = [prompt]
        else:
            text_inputs = [system_prompt]

        # Prepare processor inputs
        processor_kwargs: Dict[str, Any] = dict(
            text=text_inputs,
            padding=True,
            return_tensors="pt",
        )
        if is_video:
            # Qwen2-VL accepts videos as a list of PIL images per sample (list[list[PIL]])
            processor_kwargs["videos"] = [pil_frames]
        else:
            # Single image path (list[PIL])
            processor_kwargs["images"] = [pil_image]

        try:
            inputs = processor(**processor_kwargs)
        except TypeError:
            # Fallback without return_tensors in case of version mismatch
            processor_kwargs.pop("return_tensors", None)
            inputs = processor(**processor_kwargs)
            # Best-effort tensorize
            for k, v in list(inputs.items()):
                if isinstance(v, list):
                    try:
                        inputs[k] = torch.tensor(v)
                    except Exception:
                        pass

        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))

        # Trim generated tokens (after prompt) if using chat template
        if "input_ids" in inputs:
            start = inputs["input_ids"].shape[1]
            trimmed = out_ids[:, start:]
        else:
            trimmed = out_ids

        captions = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption = captions[0].strip() if captions else ""

        # Cleanup
        del inputs, out_ids, trimmed
        _aggressive_cleanup()

        return (caption,)


# --------------------------- Utilities ---------------------------

def _time_from_frame(idx: int, fps: float) -> float:
    return 0.0 if fps <= 0 else idx / fps


def _read_video_meta(path: str) -> Tuple[int, int, int, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Video has zero frames: {path}")
    return frame_count, width, height, fps


def _iter_frames(path: str):
    """Streaming iterator from the beginning."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield frame_bgr
    finally:
        cap.release()


def _iter_frames_from(path: str, start_frame: int, end_frame: int):
    """Iterator that seeks to start_frame and yields until end_frame (exclusive)."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        # Seek to the exact frame (best-effort; accuracy depends on codec/indexing).
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Some codecs may land a few frames before; ensure we skip up to start
        while current < start_frame:
            ok, _ = cap.read()
            if not ok:
                break
            current += 1

        while current < end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield frame_bgr
            current += 1
    finally:
        cap.release()


def _resize_keep_aspect(img, shorter_to: int):
    if shorter_to <= 0:
        return img
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    short = min(h, w)
    if short == shorter_to:
        return img
    scale = shorter_to / float(short)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _compute_scene_detections(
    path: str,
    threshold: float,
    min_scene_len: int,
    downscale_factor: int,
) -> Dict[str, Any]:
    """
    Simple adaptive shot detection:
      - Luma-based mean absolute diff between consecutive frames (optionally downscaled).
      - Robustly normalized via rolling median & MAD.
      - Cuts when normalized diff > threshold and at least min_scene_len since last cut.
    """
    frame_count, width, height, fps = _read_video_meta(path)

    threshold = float(max(0.01, threshold))
    min_scene_len = int(max(1, min_scene_len))
    downscale_factor = int(max(1, downscale_factor))

    diffs: List[float] = []
    prev_y = None
    window_size = 100
    recent_absdiffs: List[float] = []

    for idx, frame_bgr in enumerate(_iter_frames(path)):
        if downscale_factor > 1:
            frame_bgr = cv2.resize(
                frame_bgr,
                (frame_bgr.shape[1] // downscale_factor, frame_bgr.shape[0] // downscale_factor),
                interpolation=cv2.INTER_AREA,
            )
        y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]

        if prev_y is None:
            diffs.append(0.0)
            prev_y = y
            continue

        mad = float(cv2.norm(prev_y, y, normType=cv2.NORM_L1)) / (y.size)
        prev_y = y

        recent_absdiffs.append(mad)
        if len(recent_absdiffs) > window_size:
            recent_absdiffs.pop(0)

        if len(recent_absdiffs) >= 10:
            med = float(sorted(recent_absdiffs)[len(recent_absdiffs) // 2])
            abs_dev = [abs(v - med) for v in recent_absdiffs]
            mad_scale = float(sorted(abs_dev)[len(abs_dev) // 2]) or 1e-6
            norm_diff = abs(mad - med) / mad_scale
        else:
            norm_diff = mad

        diffs.append(float(norm_diff))

    # Determine cut points
    cut_idxs = []
    for i in range(1, len(diffs)):
        if diffs[i] > threshold:
            if not cut_idxs or (i - cut_idxs[-1]) >= min_scene_len:
                cut_idxs.append(i)

    # Scenes [start, end)
    scenes: List[Dict[str, Any]] = []
    last = 0
    for cut in cut_idxs:
        start = last
        end = cut
        if end - start >= max(1, min_scene_len):
            scenes.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time": _time_from_frame(start, fps),
                "end_time": _time_from_frame(end, fps),
            })
        last = cut
    # Tail
    if frame_count - last >= 1:
        scenes.append({
            "start_frame": int(last),
            "end_frame": int(frame_count),
            "start_time": _time_from_frame(last, fps),
            "end_time": _time_from_frame(frame_count, fps),
        })

    return {
        "video_path": path,
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "params": {
            "threshold": float(threshold),
            "min_scene_len": int(min_scene_len),
            "downscale_factor": int(downscale_factor),
        },
        "scenes": scenes,
    }


def _extract_scene_frames_streaming(
    path: str,
    scene: Dict[str, Any],
    resize_shorter_to: int,
    max_frames: int,
) -> torch.Tensor:
    """Walks from frame 0 but only collects frames within the target scene."""
    start = int(scene["start_frame"])
    end = int(scene["end_frame"])
    if end <= start:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)

    frames: List[torch.Tensor] = []
    seen = 0
    for f_idx, frame_bgr in enumerate(_iter_frames(path)):
        if f_idx < start:
            continue
        if f_idx >= end:
            break
        if resize_shorter_to > 0:
            frame_bgr = _resize_keep_aspect(frame_bgr, resize_shorter_to)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame_rgb).to(torch.float32) / 255.0  # (H,W,3)
        frames.append(t.unsqueeze(0))
        seen += 1
        if max_frames > 0 and seen >= max_frames:
            break

    if not frames:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
    return torch.cat(frames, dim=0).contiguous()


def _extract_scene_frames_seek(
    path: str,
    scene: Dict[str, Any],
    resize_shorter_to: int,
    max_frames: int,
) -> torch.Tensor:
    """Seeks to scene start and decodes only [start, end)."""
    start = int(scene["start_frame"])
    end = int(scene["end_frame"])
    if end <= start:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)

    frames: List[torch.Tensor] = []
    seen = 0
    for frame_bgr in _iter_frames_from(path, start, end):
        if resize_shorter_to > 0:
            frame_bgr = _resize_keep_aspect(frame_bgr, resize_shorter_to)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame_rgb).to(torch.float32) / 255.0
        frames.append(t.unsqueeze(0))
        seen += 1
        if max_frames > 0 and seen >= max_frames:
            break

    if not frames:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
    return torch.cat(frames, dim=0).contiguous()


# --------------------------- Nodes ---------------------------

class VideoSceneExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "/path/to/video.mp4"}),
                "scene_index": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 10.0, "step": 0.01}),
                "min_scene_len": ("INT", {"default": 12, "min": 1, "max": 10_000}),
                "downscale_factor": ("INT", {"default": 2, "min": 1, "max": 16}),
                "resize_shorter_to": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
            "optional": {
                "detections": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "detections")
    FUNCTION = "run"
    CATEGORY = "video/io"

    def run(
        self,
        video_path: str,
        scene_index: int,
        threshold: float,
        min_scene_len: int,
        downscale_factor: int,
        resize_shorter_to: int,
        max_frames: int,
        detections: str = "",
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for VideoSceneExtractor.")

        det_meta: Optional[Dict[str, Any]] = None
        if isinstance(detections, str) and detections.strip():
            try:
                det_meta = json.loads(detections)
            except Exception:
                det_meta = None

        if not det_meta or det_meta.get("video_path") != video_path or "scenes" not in det_meta:
            det_meta = _compute_scene_detections(
                video_path,
                threshold=threshold,
                min_scene_len=min_scene_len,
                downscale_factor=downscale_factor,
            )

        scenes: List[Dict[str, Any]] = det_meta.get("scenes", [])
        if not scenes:
            empty = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
            return (empty, json.dumps(det_meta))

        si = max(0, min(scene_index, len(scenes) - 1))
        scene = scenes[si]

        images = _extract_scene_frames_streaming(
            video_path,
            scene,
            resize_shorter_to=resize_shorter_to,
            max_frames=max_frames,
        )

        if images.ndim != 4:
            if images.ndim == 3:
                images = images.unsqueeze(0)
            else:
                images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

        return (images, json.dumps(det_meta))


class VideoSceneExtractorSeek:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "/path/to/video.mp4"}),
                "scene_index": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 10.0, "step": 0.01}),
                "min_scene_len": ("INT", {"default": 12, "min": 1, "max": 10_000}),
                "downscale_factor": ("INT", {"default": 2, "min": 1, "max": 16}),
                "resize_shorter_to": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
            "optional": {
                "detections": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "detections")
    FUNCTION = "run"
    CATEGORY = "video/io"

    def run(
        self,
        video_path: str,
        scene_index: int,
        threshold: float,
        min_scene_len: int,
        downscale_factor: int,
        resize_shorter_to: int,
        max_frames: int,
        detections: str = "",
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for VideoSceneExtractorSeek.")

        det_meta: Optional[Dict[str, Any]] = None
        if isinstance(detections, str) and detections.strip():
            try:
                det_meta = json.loads(detections)
            except Exception:
                det_meta = None

        if not det_meta or det_meta.get("video_path") != video_path or "scenes" not in det_meta:
            det_meta = _compute_scene_detections(
                video_path,
                threshold=threshold,
                min_scene_len=min_scene_len,
                downscale_factor=downscale_factor,
            )

        scenes: List[Dict[str, Any]] = det_meta.get("scenes", [])
        if not scenes:
            empty = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
            return (empty, json.dumps(det_meta))

        si = max(0, min(scene_index, len(scenes) - 1))
        scene = scenes[si]

        images = _extract_scene_frames_seek(
            video_path,
            scene,
            resize_shorter_to=resize_shorter_to,
            max_frames=max_frames,
        )

        if images.ndim != 4:
            if images.ndim == 3:
                images = images.unsqueeze(0)
            else:
                images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

        return (images, json.dumps(det_meta))


# --------------------------- ComfyUI registration ---------------------------

NODE_CLASS_MAPPINGS = {
    "AnimeJitter": AnimeJitter,
    "AlphaToSolidBackground": AlphaToSolidBackground,
    "VideoSceneExtractor": VideoSceneExtractor,
    "VideoSceneExtractorSeek": VideoSceneExtractorSeek,
    "Qwen2VLLoader": Qwen2VLLoader,
    "Qwen2VLCaption": Qwen2VLCaption,
    "ImagesToDiskTensor": ImagesToDiskTensor,
    "DiskTensorMerge": DiskTensorMerge,
    "DiskTensorToVideo": DiskTensorToVideo,
    "DiskTensorToImages": DiskTensorToImages,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeJitter": "Anime Jitter",
    "AlphaToSolidBackground": "Alpha → Solid Background",
    "VideoSceneExtractor": "Video Scene Extractor",
    "VideoSceneExtractorSeek": "Video Scene Extractor (Seek)",
    "Qwen2VLLoader": "Qwen2-VL Loader",
    "Qwen2VLCaption": "Qwen2-VL Caption (Image/Video)",
    "ImagesToDiskTensor": "Images → Disk Tensor",
    "DiskTensorMerge": "Disk Tensor Merge (Stitch)",
    "DiskTensorToVideo": "Disk Tensor → Video",
    "DiskTensorToImages": "Disk Tensor → Images",
}
