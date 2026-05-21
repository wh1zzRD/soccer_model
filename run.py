"""
CLI wrapper for the soccer AI commentary pipeline.

Usage:
    python run.py path/to/match.mkv
    python run.py path/to/match.mkv --match-name "Barcelona vs PSG"
    python run.py path/to/match.mkv --skip-tracking --skip-audio
"""

import argparse
import os
import sys
import shutil
import pickle
import time
import cv2

os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/zonos"))

from pathlib import Path
from src import constants
from scripts.action.predict import predict_single_video
from src.vlm.vlm_utils import get_key_actions, extract_video_clips, describe_actions
from src.vlm.llm_utils import generate_commentary
from src.player_tracking.main import create_tracked_video
from zonos.sample import generate_audio_commentary

import subprocess

VIDEO_DIRECTORY = "./data/soccernet/action-spotting-2023"


def log(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def convert_wav_to_mp3(input_wav, output_mp3, bitrate="192k"):
    command = ["ffmpeg", "-i", input_wav, "-c:a", "libmp3lame", "-b:a", bitrate, output_mp3]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def merge_video_with_audio(video_path, audio_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run(video_path, match_name, match_context, use_tracking, generate_audio):
    # Set up directory structure
    folder_name = match_name
    video_filename = os.path.basename(video_path)
    dest_dir = os.path.join(VIDEO_DIRECTORY, folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    dest_video = os.path.join(dest_dir, video_filename)
    if not os.path.exists(dest_video):
        print(f"Copying video to {dest_video}...")
        shutil.copy2(video_path, dest_video)

    # Constants
    experiment, split = "action_sampling_weights_002", "test"
    game = folder_name
    video = cv2.VideoCapture(dest_video)
    constants.video_fps = float(video.get(cv2.CAP_PROP_FPS))
    constants.test_games = [game]

    prediction_dir = os.path.join("data", "app_test", "action/")
    clips_dir = os.path.join(prediction_dir, game, "clips")
    output_dir = os.path.join(prediction_dir, game, "clips_commented")
    clip_cache_path = os.path.join(prediction_dir, game, "cache", "original_clips.pkl")
    desc_cache_path = os.path.join(prediction_dir, game, "cache", "descriptions.pkl")
    commentary_cache_path = os.path.join(prediction_dir, game, "cache", "commentary.pkl")

    for p in [clips_dir, output_dir, os.path.dirname(clip_cache_path)]:
        os.makedirs(p, exist_ok=True)

    # 1. Action spotting
    log("Step 1/5 — Action Spotting")
    predict_single_video(experiment, split, 0, False, Path(prediction_dir), constants, single_match=False)

    # 2. Clip extraction
    log("Step 2/5 — Extracting highlight clips")
    actions = get_key_actions(game, confidence_threshold=0.8)
    print(f"Found {len(actions)} key actions")
    original_clips = extract_video_clips(os.path.join(VIDEO_DIRECTORY, game), actions, clips_dir, clip_length=10)
    with open(clip_cache_path, "wb") as f:
        pickle.dump(original_clips, f)

    original_clips = []
    for i, action in enumerate(actions):
        label = action["label"].replace(" ", "_")
        clip_path = os.path.join(clips_dir, f"clip_{i}_{label}.mp4")
        original_clips.append({"path": clip_path, "label": action["label"], "gameTime": action["gameTime"]})

    # 3. Player tracking
    log("Step 3/5 — Player & ball tracking")
    if use_tracking:
        tracked_dir = os.path.join(prediction_dir, game, "clips_tracked")
        os.makedirs(tracked_dir, exist_ok=True)
        for clip in original_clips:
            clip_path = clip["path"].replace(" ", "_")
            tracked_path = clip_path.replace("clips", "clips_tracked")
            create_tracked_video(source_video_path=clip_path, target_video_path=tracked_path, device="cuda")
        clips = [{"path": c["path"].replace("clips", "clips_tracked"), "label": c["label"], "gameTime": c["gameTime"]} for c in original_clips]
        commentary_dir = os.path.join(prediction_dir, game, "clips_tracked", "commentary")
    else:
        clips = original_clips
        commentary_dir = os.path.join(prediction_dir, game, "clips", "commentary")

    # 4. VLM description
    log("Step 4a/5 — Generating visual descriptions (VLM)")
    descriptions = describe_actions(clips, clip_length=10, use_tracking=use_tracking)
    with open(desc_cache_path, "wb") as f:
        pickle.dump(descriptions, f)

    # 5. LLM commentary
    log("Step 4b/5 — Generating sport commentary (LLM)")
    results = generate_commentary(descriptions, match_context)
    with open(commentary_cache_path, "wb") as f:
        pickle.dump(results, f)

    # 6. Audio + merge
    if generate_audio:
        log("Step 5/5 — Generating audio and merging with video")
        os.makedirs(commentary_dir, exist_ok=True)
        for clip in clips:
            generate_audio_commentary(clip["path"], results[clip["path"]]["Generated Commentary"], clip["label"])

        os.makedirs(output_dir, exist_ok=True)
        for clip in original_clips:
            video_path_clip = clip["path"].replace("clips", "clips_tracked") if use_tracking else clip["path"]
            audio_name = os.path.splitext(os.path.basename(video_path_clip))[0]
            wav_path = os.path.join(commentary_dir, audio_name + ".wav")
            mp3_path = os.path.join(commentary_dir, audio_name + ".mp3")
            output_clip = os.path.join(output_dir, os.path.basename(video_path_clip))
            if os.path.exists(wav_path) and not os.path.exists(mp3_path):
                convert_wav_to_mp3(wav_path, mp3_path)
            if os.path.exists(mp3_path):
                merge_video_with_audio(video_path_clip, mp3_path, output_clip)
    else:
        log("Step 5/5 — Skipping audio generation")

    log(f"Done! Commented clips saved to: {output_dir}")
    print("\nCommentary generated:")
    for clip in clips:
        path = clip["path"]
        commentary = results.get(path, {}).get("Generated Commentary", "N/A")
        print(f"\n[{clip['gameTime']}] {clip['label']}\n  → {commentary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer AI Commentary Pipeline")
    parser.add_argument("video", help="Path to the input video (.mkv or .mp4)")
    parser.add_argument("--match-name", default=None, help="Name for the match folder (default: video filename without extension)")
    parser.add_argument("--match-context", default="It's a football match between two teams wearing different jerseys.", help="Context description for the LLM")
    parser.add_argument("--skip-tracking", action="store_true", help="Skip player & ball tracking step")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio generation step")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    match_name = args.match_name or Path(args.video).stem

    run(
        video_path=args.video,
        match_name=match_name,
        match_context=args.match_context,
        use_tracking=not args.skip_tracking,
        generate_audio=not args.skip_audio,
    )
