# ✨ Magix ComfyUI Nodes Pack

Welcome to **Magix Nodes** — a mixed bag of handy utilities for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).  
This pack brings together a set of tools I use for video workflows, animation flair, and AI captioning with **Qwen2-VL**.

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

### 🌸 Anime Jitter
- **AnimeJitter**  
  Adds that subtle hand-drawn *boil* effect — tiny shake/scale/rotate per frame, optional RGB micro-shifts (chromatic aberration), and some gentle film grain.  
  Perfect for making your renders feel less sterile and more “alive”.

---

### 🧠 Qwen2-VL Captioning
- **Qwen2VLLoader**  
  Loads the [Qwen2-VL](https://huggingface.co/Qwen) vision-language model into memory. Handles dtype and device map for you.

- **Qwen2VLCaption**  
  Point it at an `IMAGE` tensor (single still or batched frames as video).  
  It’ll generate a neat caption using your custom system prompt.  
  Handy for auto-tagging, quick descriptions, or building datasets.

---

## 🛠 Requirements

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- For video nodes:
  - `opencv-python`
  - `ffmpeg` (installed on your system, for accurate metadata & preprocessing)
- For Qwen2-VL nodes:
  - `transformers >= 4.41`
  - `pillow`
  - `torch` (CUDA recommended)

---

## 📦 Installation

Drop this repo into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/magix-nodes.git
````

Then restart ComfyUI — the new nodes will show up under:

* `video/io`
* `image/animation`
* `ai/qwen2vl`

---

## 🌐 Links

✨ More of my projects at [magixworld.com](http://magixworld.com)

---

## 💡 Notes

* All image tensors follow `(B, H, W, C)` format, values in `[0,1]`.
* **Scene detectors**: you can chain them — detect once, reuse detections downstream without re-running.
* **Qwen2-VL**: supports both stills and short clips (sample frames via stride & cap).

---

## 🖤 Thanks

These nodes are made to scratch my own creative itches — hope you find them useful too!
PRs, issues, and ideas are welcome.
