# ✨ Magix ComfyUI Nodes Pack

Welcome to **Magix Nodes** — a mixed bag of handy utilities for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).  
This pack brings together a set of tools I use for video workflows, animation flair, disk-backed pipelines, and AI captioning with **Qwen2-VL**.

👉 [magixworld.com](http://magixworld.com)

---

## 🚀 What's inside

### 🎬 Video Scene Tools
- **VideoSceneExtractor**  
  Simple & robust shot/scene detector. Scans your video, finds cuts, and spits out the *N-th scene* as an `IMAGE` tensor.

- **VideoSceneExtractorSeek**  
  A speedier, seek-optimized version. If you’ve already got detections, it jumps right to the scene you want instead of crawling from frame 0.

Both return:
- The frames of your chosen scene as `(B,H,W,C)` tensors in `[0,1]`.
- A JSON string with all scene metadata (start/end frames, times, etc).

---

### 🌸 Image FX
- **AnimeJitter**  
  Adds that subtle hand-drawn *boil* effect — tiny shake/scale/rotate per frame, optional RGB micro-shifts (chromatic aberration), and a touch of film grain. Perfect for making your renders feel less sterile and more “alive”.

- **AlphaToSolidBackground**  
  Composites RGBA onto a solid color (`white`, `black`, `red`, `green`, `blue`, or `custom RGB`) or passes through unchanged if already opaque/transparent selected. Outputs clean RGB.

---

### 🧠 Qwen2-VL Captioning
- **Qwen2VLLoader**  
  Loads the [Qwen2-VL](https://huggingface.co/Qwen) vision-language model into memory. Handles dtype and device map for you.

- **Qwen2VLCaption**  
  Point it at an `IMAGE` tensor (single still or batched frames as video). It generates a neat caption using your custom system prompt. Handy for auto-tagging, quick descriptions, or building datasets.

---

### 🧳 Disk-Backed Video Tensors (avoid RAM/VRAM blowups)
Work with long clips and big batches without loading everything at once.

- **ImagesToDiskTensor**  
  Saves an `IMAGE` tensor `(B,H,W,C)` to a temporary folder as numbered images (PNG/JPG) and returns a lightweight `DISK_TENSOR` handle. Frees RAM/VRAM immediately.

- **DiskTensorMerge**  
  *Metadata-only* stitching of multiple disk tensors. No frame copying or loading — it just records segments to play back in order.

- **DiskTensorToVideo** *(Auto-Number)*  
  Converts a (possibly merged) disk tensor to a video using `ffmpeg`, linking frames from disk in a continuous sequence.  
  Auto-numbers if the target filename exists (`out.mp4`, `out_001.mp4`, …) unless you force overwrite. Configure codec (`h264/hevc/prores/vp9/av1`), `crf`, `pix_fmt`, `fps`.

- **DiskTensorToImages**  
  Loads a disk tensor back into a regular `IMAGE` tensor with **stride**, **uniform downsample (max_frames)**, and **resize-shorter-side** options to stay memory-safe.

---

## 🛠 Requirements

- **Python** 3.10+
- **ComfyUI** (latest)
- **Common**
  - `pillow`
  - `numpy`
- **Video scene tools**
  - `opencv-python`
  - `ffmpeg` (installed on your system, in `PATH`)
- **Qwen2-VL nodes**
  - `transformers >= 4.41`
  - `torch` (CUDA recommended)

> All image tensors follow `(B, H, W, C)` format, values in `[0,1]` RGB.

---

## 📦 Installation

Drop this repo into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/magix-nodes.git
````

Then restart ComfyUI — the new nodes will show up under:

* `video/io` (scene extractors)
* `image/animation` (AnimeJitter)
* `image/alpha` (AlphaToSolidBackground)
* `ai/qwen2vl` (Qwen loader + caption)
* `video/disk` (disk-backed tensor workflow)

---

## 🧩 Quick Recipes

**Extract a scene → add jitter → caption**

1. `VideoSceneExtractorSeek` → pick `scene_index`
2. `AnimeJitter` → sprinkle boil
3. `Qwen2VLLoader` → `Qwen2VLCaption` → set your system prompt

**Keep memory low for long clips**

1. `VideoSceneExtractorSeek` (or any frame source) → `ImagesToDiskTensor`
2. (Optional) `DiskTensorMerge` to concatenate sequences
3. `DiskTensorToVideo` to render (choose codec/fps/quality), **or**
4. `DiskTensorToImages` to rehydrate as an `IMAGE` tensor with stride/limit/resize

---

## 🌐 Links

✨ More of my projects at [magixworld.com](http://magixworld.com)

---

## 💡 Notes

* **Scene detectors**: detect once, pass the detections JSON to downstream nodes to skip re-detection.
* **Disk tensors**: manifests reference frame folders; merges are just segment lists — fast and storage-light.
* **Video export**: uses hardlinks/symlinks to build a contiguous sequence without duplicating data.

---

## 🖤 Thanks

These nodes are made to scratch my own creative itches — hope you find them useful too!
PRs, issues, and ideas are welcome.
