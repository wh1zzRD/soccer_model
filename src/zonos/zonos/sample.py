from __future__ import annotations

import re
import subprocess
import os
import sys
from pathlib import Path

_ZONOS_MODEL = None
_ZONOS_SPEAKER = None


def _probe_duration(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _atempo_filter(speed: float) -> str:
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.3f}")
    return ",".join(parts)


def _audio_delay_filter(delay_seconds: float) -> str:
    delay_ms = max(0, int(delay_seconds * 1000))
    return f"adelay={delay_ms}|{delay_ms}"


def _clean_text(text: str) -> str:
    text = re.sub(r"[*_`#\[\]{}]", "", text)
    text = re.sub(r"\bG+O+A+L+\b!*", "GOAL!", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgoal\b!*", "GOAL!", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_short_goal_cue(text: str) -> str:
    if "GOAL!" not in text.upper():
        return text
    words = text.strip(" .!-").split()
    if len(words) <= 2:
        return os.environ.get("SOCCER_MODEL_GOAL_FINAL_CUE", "It's in... GOAL!")
    return text


def _goal_reaction_enabled() -> bool:
    return os.environ.get("SOCCER_MODEL_USE_GOAL_REACTION", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } and bool(os.environ.get("SOCCER_MODEL_GOAL_REACTION_AUDIO"))


def _mix_crowd_bed(output_wav: Path) -> str:
    if os.environ.get("SOCCER_MODEL_TTS_MIX_CROWD", "0").lower() not in {"1", "true", "yes", "on"}:
        return str(output_wav)

    crowd_audio = os.environ.get("SOCCER_MODEL_CROWD_AUDIO")
    if not crowd_audio:
        return str(output_wav)

    crowd_path = Path(crowd_audio).expanduser()
    if not crowd_path.exists():
        raise FileNotFoundError(f"Crowd audio not found: {crowd_path}")

    volume = os.environ.get("SOCCER_MODEL_CROWD_VOLUME", "0.16")
    mixed_wav = output_wav.with_name(f"{output_wav.stem}.mixed.wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_wav),
            "-stream_loop",
            "-1",
            "-i",
            str(crowd_path),
            "-filter_complex",
            (
                f"[0:a]volume=1.0[a0];"
                f"[1:a]volume={volume},highpass=f=120,lowpass=f=6500[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0,"
                "alimiter=limit=0.95[out]"
            ),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(mixed_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    mixed_wav.replace(output_wav)
    print(f"Mixed crowd background into: {output_wav}")
    return str(output_wav)


def _enhance_goal_delivery(text: str, output_wav: Path) -> str:
    if "GOAL!" not in text.upper():
        return str(output_wav)

    boosted_wav = output_wav.with_name(f"{output_wav.stem}.goal.wav")
    goal_gain = os.environ.get("SOCCER_MODEL_GOAL_GAIN_DB", "1.5")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_wav),
            "-af",
            (
                f"acompressor=threshold=-24dB:ratio=2.2:attack=12:release=220,"
                f"equalizer=f=2500:t=q:w=1.2:g=2.5,"
                f"volume={goal_gain}dB,"
                "alimiter=limit=0.96"
            ),
            str(boosted_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    boosted_wav.replace(output_wav)
    print(f"Enhanced goal delivery: {output_wav}")
    return str(output_wav)


def _fit_audio_to_clip(output_wav: Path, clip_path: str, delay_seconds: float = 0.0) -> str:
    clip_duration = _probe_duration(Path(clip_path))
    audio_duration = _probe_duration(output_wav)
    if not clip_duration or not audio_duration:
        return str(output_wav)

    target_duration = max(
        1.5,
        clip_duration * float(os.environ.get("SOCCER_MODEL_TTS_TARGET_RATIO", "0.82")) - delay_seconds,
    )
    if audio_duration <= target_duration:
        return str(output_wav)

    max_speed = float(os.environ.get("SOCCER_MODEL_TTS_MAX_SPEEDUP", "1.45"))
    speed = min(audio_duration / target_duration, max_speed)
    fitted_wav = output_wav.with_name(f"{output_wav.stem}.fit.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_wav),
            "-af",
            _atempo_filter(speed),
            str(fitted_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fitted_wav.replace(output_wav)
    print(f"Fitted commentary audio to clip timing: {output_wav} speed={speed:.2f}")
    return str(output_wav)


def _delay_audio(output_wav: Path, delay_seconds: float) -> str:
    if delay_seconds <= 0:
        return str(output_wav)

    delayed_wav = output_wav.with_name(f"{output_wav.stem}.delay.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_wav),
            "-af",
            _audio_delay_filter(delay_seconds),
            str(delayed_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    delayed_wav.replace(output_wav)
    print(f"Delayed commentary audio: {output_wav} delay={delay_seconds:.2f}s")
    return str(output_wav)


def _auto_goal_delay(output_wav: Path, clip_path: str) -> float:
    if os.environ.get("SOCCER_MODEL_AUTO_GOAL_DELAY", "1").lower() in {"0", "false", "no", "off"}:
        return 0.0

    clip_duration = _probe_duration(Path(clip_path)) or 10.0
    audio_duration = _probe_duration(output_wav)
    if not audio_duration:
        return 0.0

    target_time = float(os.environ.get("SOCCER_MODEL_GOAL_CUE_TIME", "5.0"))
    target_time = min(max(0.0, target_time), max(0.0, clip_duration - 0.8))

    # In the generated goal text, GOAL! is forced to be the final word.
    # Estimate the start of that final word and delay the whole line so the
    # peak lands near the event time instead of immediately at clip start.
    goal_word_offset = float(os.environ.get("SOCCER_MODEL_GOAL_WORD_OFFSET", "0.65"))
    estimated_goal_start = max(0.25, audio_duration - goal_word_offset)
    delay_seconds = max(0.0, target_time - estimated_goal_start)

    max_delay = max(0.0, clip_duration - audio_duration - 0.2)
    return min(delay_seconds, max_delay)


def _mix_goal_reaction(output_wav: Path, clip_path: str) -> str:
    if not _goal_reaction_enabled():
        return str(output_wav)

    reaction_audio = os.environ.get("SOCCER_MODEL_GOAL_REACTION_AUDIO")
    if not reaction_audio:
        return str(output_wav)

    reaction_path = Path(reaction_audio).expanduser()
    if not reaction_path.exists():
        raise FileNotFoundError(f"Goal reaction audio not found: {reaction_path}")

    clip_duration = _probe_duration(Path(clip_path)) or 10.0
    goal_time = float(os.environ.get("SOCCER_MODEL_GOAL_REACTION_TIME", str(clip_duration * 0.58)))
    reaction_offset = float(os.environ.get("SOCCER_MODEL_GOAL_REACTION_OFFSET", "0"))
    max_reaction_duration = float(os.environ.get("SOCCER_MODEL_GOAL_REACTION_DURATION", "3.2"))
    reaction_volume = os.environ.get("SOCCER_MODEL_GOAL_REACTION_VOLUME", "1.8")
    base_volume = os.environ.get("SOCCER_MODEL_GOAL_BASE_VOLUME", "0.75")
    delay_ms = max(0, int(goal_time * 1000))
    mixed_wav = output_wav.with_name(f"{output_wav.stem}.goal_reaction.wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_wav),
            "-i",
            str(reaction_path),
            "-filter_complex",
            (
                f"[0:a]volume={base_volume}[base];"
                f"[1:a]atrim=start={reaction_offset}:duration={max_reaction_duration},asetpts=PTS-STARTPTS,"
                f"volume={reaction_volume},{_audio_delay_filter(goal_time)}[goal];"
                "[base][goal]amix=inputs=2:duration=longest:dropout_transition=0,"
                "alimiter=limit=0.98[out]"
            ),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(mixed_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    mixed_wav.replace(output_wav)
    print(f"Mixed real goal reaction at {goal_time:.2f}s: {output_wav}")
    return str(output_wav)


def _normalize_timed_cues(commentary) -> list[dict]:
    if isinstance(commentary, dict):
        commentary = (
            commentary.get("Timed Commentary")
            or commentary.get("timed_commentary")
            or commentary.get("cues")
            or commentary.get("cue")
            or []
        )
    if not isinstance(commentary, list):
        return []

    cues = []
    for cue in commentary:
        if isinstance(cue, dict):
            text = cue.get("text") or cue.get("commentary") or cue.get("line")
            time_value = cue.get("time", cue.get("start", cue.get("start_time", 0)))
        elif isinstance(cue, (list, tuple)) and len(cue) >= 2:
            time_value, text = cue[0], cue[1]
        else:
            continue

        try:
            time_seconds = float(time_value)
        except (TypeError, ValueError):
            time_seconds = 0.0
        text = _expand_short_goal_cue(_clean_text(str(text or "")))
        if text:
            cues.append({"time": max(0.0, time_seconds), "text": text})
    return cues


def _synthesize_text_to_wav(text: str, output_wav: Path) -> str:
    backend = os.environ.get("SOCCER_MODEL_TTS_BACKEND", "").lower()
    if backend in {"", "zonos"}:
        try:
            zonos_output = _generate_with_zonos(text, output_wav)
            if zonos_output is not None:
                return zonos_output
        except Exception:
            if backend == "zonos":
                raise
            print("Zonos generation failed, falling back to the next available TTS backend.")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI TTS requires the openai package. Install it with: "
                "python -m pip install openai"
            ) from exc

        client = OpenAI()
        model = os.environ.get("SOCCER_MODEL_TTS_MODEL", "gpt-4o-mini-tts")
        voice = os.environ.get("SOCCER_MODEL_TTS_VOICE", "cedar")
        instructions = os.environ.get(
            "SOCCER_MODEL_TTS_INSTRUCTIONS",
            "Speak like a natural live soccer commentator: energetic, clear, "
            "dramatic during big moments, but not cartoonish. Keep it broadcast-ready.",
        )

        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="wav",
        ) as response:
            response.stream_to_file(output_wav)
        print(f"Generated OpenAI audio commentary: {output_wav}")
        _enhance_goal_delivery(text, output_wav)
        return str(output_wav)

    subprocess.run(
        [
            "espeak-ng",
            "-v",
            "en-us",
            "-s",
            "165",
            "-p",
            "55",
            "-a",
            "180",
            "-w",
            str(output_wav),
            text,
        ],
        check=True,
    )
    print(f"Generated audio commentary: {output_wav}")
    _enhance_goal_delivery(text, output_wav)
    return str(output_wav)


def _mix_timed_segments(output_wav: Path, clip_path: str, segments: list[dict]) -> str:
    clip_duration = _probe_duration(Path(clip_path)) or 10.0
    output_duration = clip_duration
    for segment in segments:
        segment_duration = _probe_duration(Path(segment["path"])) or 0.0
        output_duration = max(output_duration, float(segment["time"]) + segment_duration + 0.15)

    if not segments:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-t",
                str(output_duration),
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                str(output_wav),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return str(output_wav)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-t",
        str(output_duration),
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
    ]
    for segment in segments:
        command.extend(["-i", str(segment["path"])])

    filter_parts = []
    mix_inputs = ["[0:a]"]
    normal_volume = os.environ.get("SOCCER_MODEL_TIMED_CUE_VOLUME", "1.25")
    goal_volume = os.environ.get("SOCCER_MODEL_TIMED_GOAL_VOLUME", "3.0")
    for index, segment in enumerate(segments, start=1):
        delay_ms = max(0, int(float(segment["time"]) * 1000))
        volume = goal_volume if "GOAL!" in segment["text"].upper() else normal_volume
        label = f"a{index}"
        filter_parts.append(
            f"[{index}:a]aresample=44100,volume={volume},adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_inputs.append(f"[{label}]")

    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0,"
        + "alimiter=limit=0.98[out]"
    )
    command.extend(["-filter_complex", ";".join(filter_parts), "-map", "[out]", str(output_wav)])
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Mixed timed commentary cues into: {output_wav}")
    return str(output_wav)


def _generate_timed_audio_commentary(clip_path: str, cues: list[dict], label: str | None = None) -> str:
    clip = Path(clip_path)
    output_dir = clip.parent / "commentary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / f"{clip.stem}.wav"

    has_real_goal_reaction = _goal_reaction_enabled()
    segments = []
    for index, cue in enumerate(cues):
        text = cue["text"]
        if has_real_goal_reaction and "GOAL!" in text.upper():
            continue
        segment_path = output_dir / f"{clip.stem}.cue_{index}.wav"
        _synthesize_text_to_wav(text, segment_path)
        segments.append({"path": segment_path, "time": cue["time"], "text": text})

    _mix_timed_segments(output_wav, clip_path, segments)
    if label and label.lower() == "goal":
        _mix_goal_reaction(output_wav, clip_path)
    return _mix_crowd_bed(output_wav)


def _generate_with_zonos(text: str, output_wav: Path) -> str | None:
    global _ZONOS_MODEL, _ZONOS_SPEAKER

    repo_dir = Path(os.environ.get("ZONOS_REPO_DIR", "~/Zonos")).expanduser()
    zonos_pkg_dir = repo_dir / "zonos"
    if not zonos_pkg_dir.exists():
        return None

    import zonos as local_zonos

    pkg_path = str(zonos_pkg_dir)
    if pkg_path not in local_zonos.__path__:
        local_zonos.__path__.insert(0, pkg_path)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import torch
    import torchaudio
    from zonos.conditioning import make_cond_dict
    from zonos.model import Zonos
    from zonos.utils import DEFAULT_DEVICE as device

    if _ZONOS_MODEL is None:
        model_id = os.environ.get("ZONOS_MODEL_ID", "Zyphra/Zonos-v0.1-transformer")
        _ZONOS_MODEL = Zonos.from_pretrained(model_id, device=device)

    if _ZONOS_SPEAKER is None:
        speaker_audio = Path(os.environ.get("ZONOS_SPEAKER_AUDIO", str(repo_dir / "assets/exampleaudio.mp3"))).expanduser()
        if not speaker_audio.exists():
            raise FileNotFoundError(
                f"Zonos speaker audio not found: {speaker_audio}. "
                "Set ZONOS_SPEAKER_AUDIO to a 10-30 second voice sample."
            )
        wav, sampling_rate = torchaudio.load(str(speaker_audio))
        _ZONOS_SPEAKER = _ZONOS_MODEL.make_speaker_embedding(wav, sampling_rate)

    cond_dict = make_cond_dict(
        text=text,
        speaker=_ZONOS_SPEAKER,
        language=os.environ.get("ZONOS_LANGUAGE", "en-us"),
    )
    conditioning = _ZONOS_MODEL.prepare_conditioning(cond_dict)

    with torch.inference_mode():
        codes = _ZONOS_MODEL.generate(conditioning)
        wavs = _ZONOS_MODEL.autoencoder.decode(codes).cpu()

    torchaudio.save(str(output_wav), wavs[0], _ZONOS_MODEL.autoencoder.sampling_rate)
    print(f"Generated Zonos audio commentary: {output_wav}")
    _enhance_goal_delivery(text, output_wav)
    return str(output_wav)


def generate_audio_commentary(clip_path: str, commentary, label: str | None = None) -> str:
    """Generate a WAV commentary file next to a clip.

    The main app expects a function with this name from `zonos.sample`. If
    OPENAI_API_KEY is set, use OpenAI TTS for natural commentary. Otherwise,
    fall back to espeak-ng so the pipeline still produces audio.
    """
    timed_cues = _normalize_timed_cues(commentary)
    if timed_cues:
        return _generate_timed_audio_commentary(clip_path, timed_cues, label)

    clip = Path(clip_path)
    output_dir = clip.parent / "commentary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / f"{clip.stem}.wav"

    text = _clean_text(commentary)
    if not text:
        text = label or "Soccer highlight."
    is_goal = bool(label and label.lower() == "goal")
    has_real_goal_reaction = _goal_reaction_enabled()
    if is_goal:
        text = re.sub(r"\s*\bGOAL!\s*", " ", text, flags=re.IGNORECASE)
        text = text.strip(" .!-")
        if not has_real_goal_reaction:
            text = f"{text}... GOAL!" if text else "GOAL!"
        elif not text:
            text = "It's in."

    delay_env_name = "SOCCER_MODEL_GOAL_COMMENTARY_DELAY" if is_goal else "SOCCER_MODEL_COMMENTARY_DELAY"
    explicit_delay = os.environ.get(delay_env_name)
    delay_seconds = float(explicit_delay) if explicit_delay is not None else 0.0

    _synthesize_text_to_wav(text, output_wav)
    _fit_audio_to_clip(output_wav, clip_path, delay_seconds=delay_seconds)
    if is_goal and explicit_delay is None and not has_real_goal_reaction:
        delay_seconds = _auto_goal_delay(output_wav, clip_path)
    _delay_audio(output_wav, delay_seconds)
    if is_goal:
        _mix_goal_reaction(output_wav, clip_path)
    return _mix_crowd_bed(output_wav)
