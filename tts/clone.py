import argparse
import base64
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_TEXT = "你好，我正在测试克隆声音。"
DEFAULT_REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
DEFAULT_OUTPUT = "voice_clone.wav"


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


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--env-file",
        default=os.environ.get("TTS_CLONE_ENV_FILE", os.environ.get("TTS_ENV_FILE", ".env")),
    )
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Qwen3-TTS voice clone API smoke test.")
    parser.add_argument("text", nargs="?", default=env_str("TTS_CLONE_TEXT", env_str("TTS_TEXT", DEFAULT_TEXT)))
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument(
        "--base-url",
        default=env_str("TTS_CLONE_BASE_URL", env_str("TTS_BASE_URL", "http://127.0.0.1:51010")),
        help="TTS server base URL, with or without /v1",
    )
    parser.add_argument(
        "--endpoint",
        default=env_str("TTS_CLONE_ENDPOINT", "/v1/audio/voice-clone"),
        help="voice clone endpoint path or absolute URL",
    )
    parser.add_argument(
        "--ref-audio",
        default=env_str("TTS_CLONE_REF_AUDIO", DEFAULT_REF_AUDIO_URL),
        help="reference audio file path, URL, or raw base64 audio",
    )
    parser.add_argument(
        "--ref-text",
        default=env_str("TTS_CLONE_REF_TEXT", DEFAULT_REF_TEXT),
        help="transcript of the reference audio; required unless --x-vector-only is enabled",
    )
    parser.add_argument(
        "--x-vector-only",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TTS_CLONE_X_VECTOR_ONLY", False),
        help="use speaker embedding only; ref-text is optional but quality may be lower",
    )
    parser.add_argument("--language", default=env_str("TTS_CLONE_LANGUAGE", env_str("TTS_LANGUAGE", "Auto")))
    parser.add_argument(
        "--response-format",
        choices=("mp3", "opus", "aac", "flac", "wav", "pcm"),
        default=env_str("TTS_CLONE_RESPONSE_FORMAT", "wav"),
    )
    parser.add_argument("--speed", type=float, default=env_float("TTS_CLONE_SPEED", 1.0))
    parser.add_argument("--output", default=env_str("TTS_CLONE_OUTPUT", DEFAULT_OUTPUT))
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=env_float("TTS_CLONE_CONNECT_TIMEOUT", env_float("TTS_CONNECT_TIMEOUT", 10.0)),
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=env_float("TTS_CLONE_READ_TIMEOUT", env_float("TTS_READ_TIMEOUT", 60.0)),
    )
    args = parser.parse_args(argv)

    if not args.ref_audio:
        parser.error("--ref-audio is required")
    if not args.x_vector_only and not args.ref_text:
        parser.error("--ref-text is required unless --x-vector-only is enabled")
    if not 0.25 <= args.speed <= 4.0:
        parser.error("--speed must be between 0.25 and 4.0")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    if args.read_timeout <= 0:
        parser.error("--read-timeout must be positive")
    return args


def build_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint

    base = base_url.rstrip("/")
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1") :]
    return f"{base}{path}"


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def encode_ref_audio(ref_audio: str, timeout: tuple[float, float]) -> str:
    source = ref_audio.strip()
    if source.startswith("@"):
        source = source[1:]

    path = Path(source).expanduser()
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")

    if looks_like_url(source):
        print(f"[clone] GET ref_audio {source}", flush=True)
        response = requests.get(source, timeout=timeout)
        response.raise_for_status()
        if not response.content:
            raise RuntimeError(f"reference audio URL returned empty body: {source}")
        return base64.b64encode(response.content).decode("ascii")

    try:
        base64.b64decode(source, validate=True)
    except Exception as exc:
        raise ValueError(
            "--ref-audio must be an existing file, http(s) URL, or valid base64 audio"
        ) from exc
    return source


def build_payload(args: argparse.Namespace, ref_audio_base64: str) -> dict[str, object]:
    return {
        "input": args.text,
        "ref_audio": ref_audio_base64,
        "ref_text": args.ref_text or None,
        "x_vector_only_mode": args.x_vector_only,
        "language": args.language,
        "response_format": args.response_format,
        "speed": args.speed,
    }


def write_response(response: requests.Response, output: str, chunk_size: int = 1024 * 64) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    with output_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            file_obj.write(chunk)
            total_bytes += len(chunk)

    if total_bytes == 0:
        raise RuntimeError("voice clone API returned an empty audio response")
    return total_bytes


def clone_voice(args: argparse.Namespace) -> tuple[Path, int, float]:
    url = build_url(args.base_url, args.endpoint)
    timeout = (args.connect_timeout, args.read_timeout)
    ref_audio_base64 = encode_ref_audio(args.ref_audio, timeout)
    payload = build_payload(args, ref_audio_base64)

    print(
        f"[clone] POST {url} language={args.language} format={args.response_format} "
        f"x_vector_only={args.x_vector_only}",
        flush=True,
    )
    started_at = time.perf_counter()
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        elapsed = time.perf_counter() - started_at
        print(f"[clone] status={response.status_code} elapsed={elapsed:.3f}s", flush=True)
        if response.status_code >= 400:
            print(response.text, flush=True)
        response.raise_for_status()
        audio_bytes = write_response(response, args.output)

    return Path(args.output), audio_bytes, elapsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output_path, audio_bytes, elapsed = clone_voice(args)
    except requests.Timeout as exc:
        print(
            "[clone] error: request timed out "
            f"connect={args.connect_timeout}s/read={args.read_timeout}s",
            flush=True,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"[clone] error: {exc}", flush=True)
        raise SystemExit(1) from exc

    print(
        f"[clone] saved {audio_bytes} bytes -> {output_path} ({elapsed:.3f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
