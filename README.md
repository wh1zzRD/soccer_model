# Soccer AI Commentator

An end-to-end AI pipeline that takes raw soccer match videos and automatically generates natural audio commentary. The pipeline detects key actions, describes them visually, reformulates them into sports-style narration, and synthesizes spoken audio.

Live demo: [goalmelody.lovable.app/pitch](https://goalmelody.lovable.app/pitch)

## How it works

The pipeline runs in 5 stages:

1. **Action Spotting** — EfficientNetV2 CNN detects 17 action types (goals, fouls, corners, etc.) with timestamps
2. **Player & Ball Tracking** — YOLOv8 + ByteTrack draws bounding boxes and persistent IDs on players and the ball
3. **Visual Description** — Qwen2.5-VL-7B watches each clip and writes a timestamped description of what's happening
4. **Commentary Generation** — Gemma 3-4B rewrites the description into broadcast-style sports commentary
5. **Audio Synthesis** — Zonos TTS clones a commentator's voice and speaks the commentary with emotion control

## Requirements

- NVIDIA GPU with 24GB VRAM minimum
- Python 3.10
- Ubuntu (recommended for GPU server deployment)

## Setup

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs ffmpeg espeak-ng
git lfs install
```

Set up the Python environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install fastapi uvicorn
python download_vlm_weights.py
```

## Adding a video

Place your match video (720p, `.mkv`) in:

```
data/soccernet/action-spotting-2023/{match-name}/1_720p.mkv
```

For a full match with two halves, add both `1_720p.mkv` and `2_720p.mkv`.

## Running

```bash
python server.py
```

Open `http://<server-ip>:7860` in your browser.

For a quick UI preview without ML dependencies:

```bash
python preview.py
```

## Output

Results are saved at each stage under `data/app_test/action/{match-name}/`:

- `clips/` — raw action clips
- `clips_tracked/` — clips with player/ball overlays
- `clips_commented/` — final clips with audio commentary merged in
- `cache/` — intermediate outputs (descriptions, commentary text)
