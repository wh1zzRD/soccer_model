import gradio as gr
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

import io
import time
import re
import sys
import cv2
import subprocess
import pickle
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from contextlib import redirect_stdout
from pathlib import Path

from src import constants
from scripts.action.predict import predict_single_video
from src.vlm.vlm_utils import get_key_actions, extract_video_clips, describe_actions
from src.vlm.llm_utils import generate_commentary
sys.path.append(os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("src/zonos"))

try:
    from zonos.sample import generate_audio_commentary
    ZONOS_IMPORT_ERROR = None
except ImportError as exc:
    generate_audio_commentary = None
    ZONOS_IMPORT_ERROR = exc


VIDEO_DIRECTORY = str(constants.soccernet_dir)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _video_choices(folder_path):
    videos = [
        f for f in os.listdir(folder_path)
        if f.endswith(VIDEO_EXTENSIONS) and not f.startswith("original_")
    ]
    return sorted(videos, key=lambda name: (name != "1_720p.mkv", name))


def _all_video_choices():
    choices = []
    for folder in list_folders():
        folder_path = os.path.join(VIDEO_DIRECTORY, folder)
        for video in _video_choices(folder_path):
            choices.append(f"{folder}/{video}")
    return choices


def _split_video_choice(folder_name, video_name):
    if isinstance(video_name, str) and "/" in video_name:
        folder, video = video_name.split("/", 1)
        return folder, video
    return folder_name, video_name


def _source_video_path(folder_name, video_name, single_match):
    if single_match:
        return os.path.join(VIDEO_DIRECTORY, folder_name, video_name)

    canonical_path = os.path.join(VIDEO_DIRECTORY, folder_name, "1_720p.mkv")
    if os.path.exists(canonical_path):
        return canonical_path

    fallback_path = os.path.join(VIDEO_DIRECTORY, folder_name, video_name)
    if os.path.exists(fallback_path):
        return fallback_path

    folder_path = os.path.join(VIDEO_DIRECTORY, folder_name)
    videos = _video_choices(folder_path) if os.path.exists(folder_path) else []
    if videos:
        return os.path.join(folder_path, videos[0])

    return fallback_path


def list_folders():
    """Liste les dossiers disponibles contenant des vidéos"""
    folders = [f for f in os.listdir(VIDEO_DIRECTORY) if os.path.isdir(os.path.join(VIDEO_DIRECTORY, f))]
    return sorted(folders)

def list_videos_in_folder(folder_name):
    """Liste les vidéos dans le dossier sélectionné"""
    if folder_name is None or folder_name == "Aucun dossier disponible":
        return gr.update(choices=[], value=None)  # Pas de vidéos sélectionnables

    folder_path = os.path.join(VIDEO_DIRECTORY, folder_name)

    if not os.path.exists(folder_path):  # Vérifie que le dossier existe
        return gr.update(choices=[], value=None)

    videos = _video_choices(folder_path)

    if videos:
        return gr.update(choices=videos, value=videos[0])  # Met la première vidéo par défaut
    else:
        return gr.update(choices=[], value=None)  # Aucune vidéo trouvée

def toggle_video_dropdown(mode_choice):
    return gr.update(visible=True)

def merge_video_with_audio(video_path: str, audio_path: str, output_path: str):
    """
    Merges a video (.mp4) with an audio commentary (.mp3) and saves the result.
    If the original video has no audio stream, it first adds a silent track.

    Args:
        video_path (str): Path to the original video file.
        audio_path (str): Path to the corresponding audio commentary (.mp3).
        output_path (str): Path where the final video will be saved.

    Returns:
        str: Path to the saved merged video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output folder exists

    with VideoFileClip(video_path) as video_clip:
        video_duration_seconds = float(video_clip.duration)
    with AudioFileClip(audio_path) as audio_clip:
        audio_duration_seconds = float(audio_clip.duration)

    output_duration_seconds = max(video_duration_seconds, audio_duration_seconds)
    video_extension_seconds = max(0.0, output_duration_seconds - video_duration_seconds)
    video_duration = f"{output_duration_seconds:.3f}"
    video_filter = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={video_extension_seconds:.3f},"
        "format=yuv420p[vout];"
    )

    crowd_path = os.environ.get("SOCCER_MODEL_CROWD_AUDIO")
    crowd_volume = os.environ.get("SOCCER_MODEL_CROWD_VOLUME", "0.14")
    crowd_offset = os.environ.get("SOCCER_MODEL_CROWD_OFFSET", "0.75")
    commentary_volume = os.environ.get("SOCCER_MODEL_COMMENTARY_VOLUME", "1.35")

    # Keep the full video duration. If commentary is shorter than the clip,
    # pad it with silence and optionally keep a crowd bed under the whole clip.
    if crowd_path and os.path.exists(crowd_path):
        merge_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-stream_loop",
            "-1",
            "-i",
            crowd_path,
            "-filter_complex",
            (
                video_filter +
                f"[1:a]aresample=44100,loudnorm=I=-16:LRA=8:TP=-1.5,"
                f"volume={commentary_volume},apad[a1];"
                f"[2:a]atrim=start={crowd_offset},asetpts=PTS-STARTPTS,aresample=44100,"
                f"volume={crowd_volume},highpass=f=120,lowpass=f=6500[a2];"
                "[a1][a2]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                "alimiter=limit=0.95[aout]"
            ),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-t",
            video_duration,
            output_path,
        ]
    else:
        merge_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            (
                video_filter +
                f"[1:a]aresample=44100,loudnorm=I=-16:LRA=8:TP=-1.5,"
                f"volume={commentary_volume},apad,alimiter=limit=0.95[aout]"
            ),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-t",
            video_duration,
            output_path,
        ]

    try:
        subprocess.run(merge_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Merged: {video_path} + {audio_path} → {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging {video_path} and {audio_path}:\n{e.stderr.decode()}")
        return None


def convert_wav_to_mp3(input_wav: str, output_mp3: str, bitrate="192k"):
    """
    Converts a WAV file to MP3 using FFmpeg.
    Args:
        input_wav (str): Path to the input WAV file.
        output_mp3 (str): Path to the output MP3 file.
        bitrate (str, optional): Bitrate for the MP3 file (default: 192k).
    Returns:
        str: Path to the converted MP3 file, or None if an error occurred.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_wav,  # Input WAV file
        "-c:a", "libmp3lame",  # Use LAME MP3 encoder
        "-b:a", bitrate,  # Set audio bitrate
        output_mp3  # Output MP3 file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {input_wav} → {output_mp3}")
        return output_mp3
    except subprocess.CalledProcessError as e:
        print(f" Error converting {input_wav} to {output_mp3}:\n{e.stderr.decode()}")
        return None


def merge_all_videos_with_audio(CLIPS_DIR, COMMENTARY_DIR, OUTPUT_DIR):
    """
    Merges all videos in `data/clips/` with their corresponding audio commentaries in `data/clips/commentary/`
    and saves them in `data/commented_clips/`.
    """
    # Define paths
    #BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    #CLIPS_DIR = os.path.join(BASE_DIR, "data/clips")
    #COMMENTARY_DIR = os.path.join(CLIPS_DIR, "commentary")
    #OUTPUT_DIR = os.path.join(BASE_DIR, "data/commented_clips")
    
    os.makedirs(COMMENTARY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output folder exists

    # Convert all WAV files to MP3
    for audio_file in os.listdir(COMMENTARY_DIR):
        if audio_file.endswith(".wav"):
            wav_path = os.path.join(COMMENTARY_DIR, audio_file)
            mp3_path = os.path.join(COMMENTARY_DIR, os.path.splitext(audio_file)[0] + ".mp3")
            convert_wav_to_mp3(wav_path, mp3_path)

    # Process each video in `data/clips/`
    for video_file in os.listdir(CLIPS_DIR):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(CLIPS_DIR, video_file)
            audio_path = os.path.join(COMMENTARY_DIR, os.path.splitext(video_file)[0] + ".mp3")
            output_path = os.path.join(OUTPUT_DIR, video_file) #video_file.split(".")[0]+".mkv")  # Save with same name but in `commented_clips/`

            # Ensure the corresponding audio exists
            if os.path.exists(audio_path):
                merge_video_with_audio(video_path, audio_path, output_path)
            else:
                print(f"⚠ No matching audio found for {video_file}. Skipping.")


def build_highlight_filters(original_clips):
    unique_labels = sorted(set([clip["label"] for clip in original_clips]))
    label_to_highlights = {}
    for label in unique_labels:
        label_to_highlights[label] = [
            _clip_display_name(clip)
            for clip in original_clips if clip["label"] == label
        ]
    return unique_labels, label_to_highlights


def _clip_display_name(clip: dict) -> str:
    return f"{clip['gameTime']} {clip['label']}"


def _normalise_highlight_name(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _resolve_highlight_name(clip_name: str, clip_mapping: dict) -> str | None:
    if clip_name in clip_mapping:
        return clip_name

    wanted = _normalise_highlight_name(clip_name)
    for mapped_name in clip_mapping:
        if _normalise_highlight_name(mapped_name) == wanted:
            return mapped_name

    return None


def _cache_missing(path: str, step: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{step} cache not found at {path}. Enable the required previous step once, then rerun."
    )


def _find_clip_key(data: dict, clip_path: str) -> str | None:
    candidates = [
        clip_path,
        clip_path.replace("clips", "clips_tracked"),
        clip_path.replace("clips_tracked", "clips"),
    ]
    for candidate in candidates:
        if candidate in data:
            return candidate
    return None


def _is_video_readable(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and "video" in result.stdout


def _remove_unreadable_video(path: str):
    if os.path.exists(path) and not _is_video_readable(path):
        print(f"Removing unreadable cached video: {path}")
        os.remove(path)


def _has_entries_for_clips(data: dict, clips: list[dict], field: str | None = None) -> bool:
    if not isinstance(data, dict):
        return False
    for clip in clips:
        key = _find_clip_key(data, clip["path"])
        if key is None:
            return False
        if field and field not in data.get(key, {}):
            return False
    return True


def _clips_are_readable(clips: list[dict]) -> bool:
    return bool(clips) and all(_is_video_readable(clip["path"]) for clip in clips)


def _run_action_spotting_subprocess(
    experiment: str,
    split: str,
    game: str,
    prediction_dir: str,
    video_fps: float,
    single_match: bool,
):
    """Run action spotting in a separate process so CUDA library fixes stay isolated."""
    env = os.environ.copy()
    venv = env.get("VIRTUAL_ENV")
    if venv and env.get("SOCCER_MODEL_ACTION_USE_LD_FIX", "1").lower() not in {"0", "false", "no"}:
        nvidia_libs = [
            f"{venv}/lib/python3.10/site-packages/nvidia/cublas/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/cudnn/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/cufft/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/curand/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/cusolver/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/cusparse/lib",
            f"{venv}/lib/python3.10/site-packages/nvidia/nccl/lib",
        ]
        env["LD_LIBRARY_PATH"] = ":".join(nvidia_libs + [env.get("LD_LIBRARY_PATH", "")])
        env["LD_PRELOAD"] = (
            f"{nvidia_libs[0]}/libcublasLt.so.12:"
            f"{nvidia_libs[0]}/libcublas.so.12"
        )

    code = f"""
from pathlib import Path
from src import constants
from scripts.action.predict import predict_single_video
constants.video_fps = {video_fps!r}
constants.test_games = [{game!r}]
constants.halves = list(range(1, int(__import__('os').environ.get('SOCCER_MODEL_NUM_HALVES', '1')) + 1))
predict_single_video({experiment!r}, {split!r}, 0, False, Path({prediction_dir!r}), constants, single_match={single_match!r})
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def _create_tracked_video_subprocess(source_path: str, target_path: str, device: str) -> bool:
    """Create tracked video outside the Gradio process so crashes do not kill the app."""
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    timeout = int(env.get("SOCCER_MODEL_TRACKING_TIMEOUT", "900"))
    code = f"""
from src.player_tracking.main import create_tracked_video
create_tracked_video({source_path!r}, {target_path!r}, device={device!r})
"""
    try:
        subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"Tracking failed for {source_path}: {exc}")
        _remove_unreadable_video(target_path)
        return False
    if not _is_video_readable(target_path):
        _remove_unreadable_video(target_path)
        return False
    return True
    

def full_pipeline(folder_name, video_name, video_prediction=True, clip_extraction=True, use_tracking=True, generate_description=True, gen_commentary=True, generate_audio=True, single_match=False):
    experiment, split = "action_sampling_weights_002", "test"
    game = f"{folder_name}/{video_name}" if single_match else folder_name
    source_video_path = _source_video_path(folder_name, video_name, single_match)
    video = cv2.VideoCapture(source_video_path)
    constants.video_fps = float(video.get(cv2.CAP_PROP_FPS))
    video.release()
    if not constants.video_fps:
        raise ValueError(f"Could not read FPS from video: {source_video_path}")

    constants.test_games = [game]
    prediction_dir = os.path.join("data", "app_test", "action")
    game_prediction_dir = os.path.join(prediction_dir, game)
    clips_dir = os.path.join(game_prediction_dir, "clips")
    tracked_clips_dir = os.path.join(game_prediction_dir, "clips_tracked")
    output_dir = os.path.join(game_prediction_dir, "clips_commented")
    cache_dir = os.path.join(game_prediction_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    clip_cache_path = os.path.join(cache_dir, "original_clips.pkl")
    desc_cache_path = os.path.join(cache_dir, "descriptions.pkl")
    commentary_cache_path = os.path.join(cache_dir, "commentary.pkl")
    spotting_path = os.path.join(game_prediction_dir, "results_spotting.json")

    empty_update = ([], {}, {}, {}, [], {}, gr.update())
    if video_prediction or not os.path.exists(spotting_path):
        status = "Running Action Spotting model on video..."
        if not video_prediction:
            status = "Action Spotting cache missing, running it now..."
        yield status, *empty_update
        _run_action_spotting_subprocess(
            experiment=experiment,
            split=split,
            game=game,
            prediction_dir=prediction_dir,
            video_fps=constants.video_fps,
            single_match=single_match,
        )

    actions = get_key_actions(game, confidence_threshold=0.8)
    yield f"Action Spotting ready, {len(actions)} key actions identified.", *empty_update

    original_clips = []
    if not clip_extraction and os.path.exists(clip_cache_path):
        with open(clip_cache_path, "rb") as f:
            original_clips = pickle.load(f)
        if not _clips_are_readable(original_clips):
            print("Clip cache exists, but one or more clip files are missing or unreadable. Regenerating clips.")
            original_clips = []

    if clip_extraction or not original_clips:
        status = "Extracting highlight clips..."
        if not clip_extraction:
            status = "Clip cache missing or invalid, extracting highlight clips now..."
        yield status, *empty_update
        original_clips = extract_video_clips(
            os.path.join(VIDEO_DIRECTORY, game),
            actions,
            clips_dir,
            clip_length=10,
            single_match=single_match,
        )
        with open(clip_cache_path, "wb") as f:
            pickle.dump(original_clips, f)

    if not _clips_are_readable(original_clips):
        raise RuntimeError("Clip extraction did not produce readable video clips.")

    active_tracking = False
    if use_tracking:
        os.makedirs(tracked_clips_dir, exist_ok=True)
        tracked_clips = []
        tracking_device = os.environ.get("SOCCER_MODEL_TRACKING_DEVICE", "cuda")
        yield "Preparing tracked clips...", *empty_update
        for clip in original_clips:
            source_path = clip["path"].replace(" ", "_")
            tracked_path = clip["path"].replace("clips", "clips_tracked").replace(" ", "_")
            _remove_unreadable_video(tracked_path)
            if _is_video_readable(tracked_path):
                print(f"Using cached tracked video: {tracked_path}")
                tracking_ok = True
            else:
                tracking_ok = _create_tracked_video_subprocess(source_path, tracked_path, tracking_device)
            if not tracking_ok:
                print("Player tracking failed. Continuing with the original clips.")
                tracked_clips = []
                break
            tracked_clips.append({"path": tracked_path, "label": clip["label"], "gameTime": clip["gameTime"]})

        active_tracking = _clips_are_readable(tracked_clips)
        if not active_tracking:
            yield "⚠️ Player tracking failed or timed out. Continuing with normal clips.", *empty_update

    if active_tracking:
        clips = tracked_clips
        commentary_dir = os.path.join(tracked_clips_dir, "commentary")
        display_clip_dir = tracked_clips_dir
    else:
        clips = original_clips
        commentary_dir = os.path.join(clips_dir, "commentary")
        display_clip_dir = clips_dir

    clip_choices = [_clip_display_name(clip) for clip in original_clips]
    clip_mapping = {
        _clip_display_name(clip): os.path.join(output_dir, os.path.basename(clip["path"]))
        for clip in original_clips
    }
    unique_labels, label_to_highlights = build_highlight_filters(original_clips)

    descriptions = {}
    if os.path.exists(desc_cache_path):
        with open(desc_cache_path, "rb") as f:
            descriptions = pickle.load(f)

    needs_descriptions = generate_description or gen_commentary or generate_audio
    if needs_descriptions and (generate_description or not _has_entries_for_clips(descriptions, clips, "description")):
        status = "Generating visual descriptions..."
        if not generate_description:
            status = "Description cache missing or outdated, generating descriptions now..."
        yield status, gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), clip_mapping, {}, {}, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)
        descriptions = describe_actions(clips, clip_length=10, use_tracking=active_tracking)
        with open(desc_cache_path, "wb") as f:
            pickle.dump(descriptions, f)

    results = {}
    if os.path.exists(commentary_cache_path):
        with open(commentary_cache_path, "rb") as f:
            results = pickle.load(f)

    needs_commentary = gen_commentary or generate_audio
    match_context = "It's a football game between two teams wearing different jerseys. Avoid naming teams unless the team identity is certain."
    timed_commentary_enabled = os.environ.get("SOCCER_MODEL_TIMED_COMMENTARY", "1").lower() not in {"0", "false", "no"}
    required_commentary_field = "Timed Commentary" if timed_commentary_enabled and generate_audio else "Generated Commentary"
    if needs_commentary and (gen_commentary or not _has_entries_for_clips(results, clips, required_commentary_field)):
        if not _has_entries_for_clips(descriptions, clips, "description"):
            yield "Description cache missing or outdated, generating descriptions now...", gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), clip_mapping, {}, {}, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)
            descriptions = describe_actions(clips, clip_length=10, use_tracking=active_tracking)
            with open(desc_cache_path, "wb") as f:
                pickle.dump(descriptions, f)
        status = "Generating commentary text..."
        if not gen_commentary:
            status = "Commentary cache missing or outdated, generating commentary text now..."
        yield status, gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), clip_mapping, {}, descriptions, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)
        results = generate_commentary(descriptions, match_context)
        with open(commentary_cache_path, "wb") as f:
            pickle.dump(results, f)

    if generate_audio:
        yield "Generating audio and merging videos...", gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), clip_mapping, results, descriptions, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)
        if generate_audio_commentary is None:
            raise ImportError(
                "Audio generation requires the Zonos package/module used by "
                "`from zonos.sample import generate_audio_commentary`. "
                "Install it on the server or disable Audio Generation in the UI."
            ) from ZONOS_IMPORT_ERROR
        for clip in clips:
            result_key = _find_clip_key(results, clip["path"])
            if result_key is None:
                raise KeyError(f"No commentary text found for clip: {clip['path']}")
            result_data = results[result_key]
            commentary_payload = (
                result_data.get("Timed Commentary")
                if timed_commentary_enabled
                else None
            ) or result_data["Generated Commentary"]
            generate_audio_commentary(clip["path"], commentary_payload, clip["label"])
        merge_all_videos_with_audio(CLIPS_DIR=display_clip_dir, COMMENTARY_DIR=commentary_dir, OUTPUT_DIR=output_dir)

    final_status = "✅ Generation complete! Select a highlight to view."
    if not needs_descriptions and not generate_audio:
        final_status = "✅ Clips ready. Select a highlight to view."
    yield final_status, gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), clip_mapping, results, descriptions, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)

def run_pipeline_with_mode(folder, video, mode,
                           video_prediction, clip_extraction, use_tracking,
                           generate_description, gen_commentary, generate_audio):
    folder, video = _split_video_choice(folder, video)
    single_match = (mode == "Analyze specific extract")
    
    # Consomme le générateur pour récupérer le dernier yield
    pipeline = full_pipeline(
        folder_name=folder,
        video_name=video,
        video_prediction=video_prediction,
        clip_extraction=clip_extraction,
        use_tracking=use_tracking,
        generate_description=generate_description,
        gen_commentary=gen_commentary,
        generate_audio=generate_audio,
        single_match=single_match
    )

    final_output = None
    for output in pipeline:
        final_output = output  # keep updating until we get the last one
    
    return final_output

def display_highlight(clip_name, clip_mapping, results, descriptions):
    """Updates video display and text when a highlight is selected."""
    if not clip_name or (isinstance(clip_name, list) and len(clip_name) == 0):
        # Return only two outputs: video (None) and an error message.
        return None, None, "⚠️ Please select a highlight.", "⚠️ Please select a highlight."
    
    # Ensure `clip_name` is a string
    if isinstance(clip_name, list):
        clip_name = clip_name[0]
    if not isinstance(clip_name, str):
        return None, None, "⚠️ Invalid selection.", "⚠️ Please select a highlight."
    if clip_name in {"Choose a highlight", "No available highlight"}:
        return None, None, "⚠️ Please select a highlight.", "⚠️ Please select a highlight."
    
    # Ensure `clip_mapping` is a dictionary
    if not isinstance(clip_mapping, dict):
        return None, None, "⚠️ Error: Clip mapping is missing.", "⚠️ Please select a highlight."
    
    # Retrieve the correct video path
    resolved_clip_name = _resolve_highlight_name(clip_name, clip_mapping)
    if resolved_clip_name is not None:
        video_path = clip_mapping[resolved_clip_name]
        video_path = video_path.replace("clips_tracked", "clips_commented")
        result_path = video_path.replace("clips_commented", "clips_tracked")
        if not os.path.exists(result_path):
            result_path = video_path.replace("clips_commented", "clips")
        print(f"video path : {video_path}, result path {result_path}")
        result_key = _find_clip_key(results, result_path) if isinstance(results, dict) else None
        desc_key = _find_clip_key(descriptions, result_path) if isinstance(descriptions, dict) else None
        result = results.get(result_key, {}).get("Generated Commentary", "ℹ️ No commentary available.") if result_key else "ℹ️ No commentary available."
        description = descriptions.get(desc_key, {}).get("description", "ℹ️ No description available.") if desc_key else "ℹ️ No description available."
        tracked_video = result_path if os.path.exists(result_path) else None
        commented_video = video_path if os.path.exists(video_path) else None
        return tracked_video, commented_video, result, description 

    return None, None, "❌ Selected highlight not found.", "❌ Selected highlight not found."

def update_highlight_dropdown(selected_label, label_to_highlights):
    if not selected_label or selected_label not in label_to_highlights:
        return gr.update(choices=[], value=None)
    choices = label_to_highlights[selected_label]
    return gr.update(choices=choices, value=choices[0] if choices else None)


original_clips = [{'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_0_Corner.mp4', 'label': 'Corner', 'gameTime': '1 - 00:37'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_4_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:34'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_6_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:52'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_7_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 02:07'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_8_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 02:11'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_9_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 03:40'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_10_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 04:31'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_12_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 06:21'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_13_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:33'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_14_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:41'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_15_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 07:48'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_16_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:44'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_17_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:46'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_18_Kick-off.mp4', 'label': 'Kick-off', 'gameTime': '1 - 09:17'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_20_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 10:05'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_21_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 11:00'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_22_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:06'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_23_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:08'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_24_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:39'}]


# Callback pour changer la visibilité des zones vidéo en fonction de la sélection de la source
def update_video_source(source):
    if source == "Tracked Video":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks() as demo:
    gr.Markdown("# 🎥 AI-Powered Soccer Match Commentary")

    initial_folders = list_folders()
    initial_folder = initial_folders[0] if initial_folders else None
    initial_videos = _all_video_choices()

    folder_dropdown = gr.Dropdown(
        choices=initial_folders,
        value=initial_folder,
        label="📂 Internal match folder",
        visible=False,
    )

    #################### CHOSE THE PARAMETERS ######################
    mode_radio = gr.Radio(
        choices=["Analyze specific extract", "Analyze full match (2 halves)"],
        label="🎯 Choose Analysis Mode",
        value="Analyze full match (2 halves)"
    )
    gr.Markdown("Check the following boxes if you want to run the different steps or use the cached results to reduce inference time.")
    video_prediction_checkbox = gr.Checkbox(label="🎯 Run Action Spotting", value=True)
    clip_extraction_checkbox = gr.Checkbox(label="🎬 Run Clips Extraction", value=True)
    use_tracking_checkbox = gr.Checkbox(label="📌 Use Player Tracking", value=True)
    generate_description_checkbox = gr.Checkbox(label="📝 Descriptions Generation", value=True)
    gen_commentary_checkbox = gr.Checkbox(label="🗣️ Commentary Generation", value=True)
    generate_audio_checkbox = gr.Checkbox(label="🔊 Audio Generation", value=True)

    ########################### VIDEO SELECTION IF NEEDED ############################
    video_dropdown = gr.Dropdown(
        label="🎥 Select a soccer match",
        choices=initial_videos if initial_videos else ["No .mkv files available"],
        value=initial_videos[0] if initial_videos else None,
        visible=True,
    )
    mode_radio.change(fn=toggle_video_dropdown, inputs=mode_radio, outputs=video_dropdown)

    ###################### START GENERATION #######################
    start_button = gr.Button("🚀 Start Commentary Generation")
    output_textbox = gr.Textbox(label="Generation Status", interactive=False)

    video_source_radio = gr.Radio(choices=["Tracked Video", "Commented Video"], label="Select a video", value="Tracked Video")

    ###################### HIGHLIGHT SELECTION ########################
    action_type_dropdown = gr.Dropdown(label="🏷️ Select Action Type", choices=["No types yet"], value=None, visible=True)
    highlights_dropdown = gr.Dropdown(label="📌 Select a Highlight", choices=["No available highlight"], visible=False)# Zones vidéo
    video_player_tracked = gr.Video(label="🎬 Tracked Video", visible=True)
    video_player_commented = gr.Video(label="🎬 Commented Video", visible=False)
    # video_player = gr.Video(label="🎬 Highlight Clip", visible=False)
    description_textbox = gr.Textbox(label="📝 VLM Output Description", visible=True)
    result_textbox = gr.Textbox(label="📝 AI-Generated Commentary", visible=False)

    stored_clip_mapping = gr.State({})  
    stored_results = gr.State({})
    stored_description = gr.State({})
    stored_action_labels = gr.State([])  # ⬅️ Ajoute ces 2 états
    stored_action_to_highlights = gr.State({})

    start_button.click(
        fn=run_pipeline_with_mode,
        inputs=[
            folder_dropdown, video_dropdown, mode_radio,
            video_prediction_checkbox, clip_extraction_checkbox, use_tracking_checkbox,
            generate_description_checkbox, gen_commentary_checkbox, generate_audio_checkbox
        ],
        outputs=[
            output_textbox, highlights_dropdown, stored_clip_mapping,
            stored_results, stored_description, stored_action_labels, stored_action_to_highlights, action_type_dropdown
        ]
    )

    ###################### FILTER HIGHLIGHTS BY LABEL ########################
    action_type_dropdown.change(
        fn=update_highlight_dropdown,
        inputs=[action_type_dropdown, stored_action_to_highlights],
        outputs=highlights_dropdown
    )

    ###################### DISPLAY HIGHLIGHT ########################
    highlights_dropdown.change(
        fn=display_highlight,
        inputs=[highlights_dropdown, stored_clip_mapping, stored_results, stored_description],
        outputs=[video_player_tracked, video_player_commented, result_textbox, description_textbox]
    )

    ###################### VIDEO TOGGLE ########################
    video_source_radio.change(
        fn=update_video_source,
        inputs=video_source_radio,
        outputs=[video_player_tracked, video_player_commented]
    )

    # highlights_dropdown.change(fn=display_highlight, inputs=[highlights_dropdown, stored_clip_mapping, stored_results, stored_description], outputs=[video_player_tracked, video_player_commented, result_textbox, description_textbox])

    # video_source_radio.change(
    #     fn=update_video_source, 
    #     inputs=video_source_radio, 
    #     outputs=[video_player_tracked, video_player_commented]
    # )

    highlights_dropdown.change(lambda *args: gr.update(visible=True), outputs=highlights_dropdown)
    highlights_dropdown.change(lambda *args: gr.update(visible=True), outputs=description_textbox)
    highlights_dropdown.change(lambda *args: gr.update(visible=True), outputs=result_textbox)


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes", "on"}
    demo.launch(server_name=server_name, server_port=server_port, share=share)
