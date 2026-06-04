import argparse
import os
import time
from collections.abc import Iterable
from pathlib import Path

import requests


DEFAULT_TEXT = "你好，我正在测试流式语音播放。"
SAMPLE_WIDTH = 2
CHANNELS = 1


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.environ.get("TTS_ENV_FILE", ".env"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Streaming TTS PCM playback test.")
    parser.add_argument("text", nargs="?", default=env_str("TTS_TEXT", DEFAULT_TEXT))
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument("--base-url", default=env_str("TTS_BASE_URL", "http://127.0.0.1:51010"))
    parser.add_argument("--model", default=env_str("TTS_MODEL", "/workspace/model/Qwen3-TTS-12Hz-1.7B-Base"))
    parser.add_argument("--voice", default=env_str("TTS_VOICE", "custom_voice_1"))
    parser.add_argument("--sample-rate", type=int, default=env_int("TTS_SAMPLE_RATE", 24000))
    parser.add_argument("--language", default=env_str("TTS_LANGUAGE", "Auto"))
    parser.add_argument("--task-type", default=env_str("TTS_TASK_TYPE", "CustomVoice"))
    parser.add_argument("--chunk-size", type=int, default=env_int("TTS_CHUNK_SIZE", 4096))
    args = parser.parse_args(argv)

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def stream_tts(args: argparse.Namespace) -> Iterable[bytes]:
    url = f"{args.base_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "model": args.model,
        "voice": args.voice,
        "input": args.text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "language": args.language,
        "task_type": args.task_type,
    }

    with requests.post(url, json=payload, stream=True, timeout=(10, None)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=args.chunk_size):
            if chunk:
                yield chunk


def play_pcm(chunks: Iterable[bytes], sample_rate: int) -> None:
    import sounddevice as sd

    frame_bytes = CHANNELS * SAMPLE_WIDTH
    pending = b""
    started_at = time.perf_counter()
    first_write_at = None
    audio_bytes = 0

    with sd.RawOutputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="int16",
        latency="low",
    ) as stream:
        for chunk in chunks:
            audio_bytes += len(chunk)
            pending += chunk
            writable = len(pending) - (len(pending) % frame_bytes)
            if writable:
                stream.write(pending[:writable])
                if first_write_at is None:
                    first_write_at = time.perf_counter()
                    print(f"首包播放时间: {first_write_at - started_at:.3f} 秒")
                pending = pending[writable:]

        if pending:
            stream.write(pending.ljust(frame_bytes, b"\x00"))
            if first_write_at is None:
                first_write_at = time.perf_counter()
                print(f"首包播放时间: {first_write_at - started_at:.3f} 秒")

    total_frames = (audio_bytes + frame_bytes - 1) // frame_bytes
    total_audio_seconds = total_frames / sample_rate
    print(f"总音频播放时间: {total_audio_seconds:.3f} 秒")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    play_pcm(stream_tts(args), args.sample_rate)


if __name__ == "__main__":
    main()
