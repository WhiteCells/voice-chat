import argparse
import asyncio
import contextlib
import json
import os
import ssl
import struct
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_WAV_ID = "wav_default_id"
DEFAULT_OTHER_BLOCK_BYTES = 204800


@dataclass(frozen=True)
class AudioPayload:
    data: bytes
    sample_rate: int
    wav_format: str


@dataclass(frozen=True)
class WavItem:
    path: Path
    wav_id: str


def parse_chunk_size(value: str) -> list[int]:
    parts = value.split("-")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk-size must look like 5-10-5")
    try:
        chunk_size = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk-size must contain integers") from exc
    if any(part <= 0 for part in chunk_size):
        raise argparse.ArgumentTypeError("chunk-size values must be positive")
    return chunk_size


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


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.environ.get("ASR_ENV_FILE", ".env"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        prog="asr",
        description="FunASR websocket streaming client implemented in Python.",
    )
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument("--server-ip", default=env_str("ASR_HOST", "127.0.0.1"), help="FunASR server host")
    parser.add_argument("--port", default=env_str("ASR_PORT", "10095"), help="FunASR websocket server port")
    parser.add_argument(
        "--wav-path",
        default=env_str("ASR_WAV_PATH", ""),
        help="wav/pcm/audio file path, or a Kaldi-style wav.scp file",
    )
    parser.add_argument("--audio-fs", type=int, default=env_int("ASR_AUDIO_FS", 16000), help="PCM sample rate")
    parser.add_argument(
        "--record",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_RECORD", False)),
        help="1 streams microphone audio; 0 streams files",
    )
    parser.add_argument(
        "--mode",
        "--asr-mode",
        dest="mode",
        default=env_str("ASR_MODE", "2pass"),
        help="ASR mode: offline, online, or 2pass",
    )
    parser.add_argument(
        "--chunk-size",
        type=parse_chunk_size,
        default=parse_chunk_size(env_str("ASR_CHUNK_SIZE", "5-10-5")),
        help="FunASR chunk size, for example 5-10-5",
    )
    parser.add_argument("--thread-num", type=int, default=env_int("ASR_THREAD_NUM", 1), help="parallel file streams")
    parser.add_argument(
        "--is-ssl",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_SSL", True)),
        help="1 uses wss; 0 uses ws",
    )
    parser.add_argument(
        "--ssl-verify",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_SSL_VERIFY", False)),
        help="1 verifies TLS certificates; 0 allows self-signed local servers",
    )
    parser.add_argument(
        "--use-itn",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_USE_ITN", True)),
        help="1 enables inverse text normalization",
    )
    parser.add_argument(
        "--svs-itn",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_SVS_ITN", True)),
        help="1 enables ITN and punctuation for SVS output",
    )
    parser.add_argument(
        "--hotword",
        default=env_str("ASR_HOTWORD", ""),
        help="hotword file; one item per line like: word weight",
    )
    parser.add_argument(
        "--chunk-interval",
        type=int,
        default=env_int("ASR_CHUNK_INTERVAL", 10),
        help="FunASR chunk interval sent in the initial JSON message",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=env_int("ASR_BLOCK_SIZE", 0),
        help="binary bytes per send; 0 derives a streaming PCM chunk size",
    )
    parser.add_argument(
        "--send-interval-ms",
        type=float,
        default=env_optional_float("ASR_SEND_INTERVAL_MS"),
        help="sleep after each binary send; omitted derives realtime PCM pacing",
    )
    parser.add_argument(
        "--final-timeout",
        type=float,
        default=env_float("ASR_FINAL_TIMEOUT", 60.0),
        help="seconds to wait for a final result after audio is sent",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=env_float("ASR_RECORD_SECONDS", 0.0),
        help="microphone streaming duration; 0 means until Ctrl+C",
    )
    args = parser.parse_args(argv)

    if args.thread_num <= 0:
        parser.error("--thread-num must be positive")
    if args.audio_fs <= 0:
        parser.error("--audio-fs must be positive")
    if args.chunk_interval <= 0:
        parser.error("--chunk-interval must be positive")
    if args.block_size < 0:
        parser.error("--block-size cannot be negative")
    if args.final_timeout <= 0:
        parser.error("--final-timeout must be positive")
    if args.record == 0 and not args.wav_path:
        parser.error("--wav-path is required when --record 0")

    return args


def build_uri(args: argparse.Namespace) -> str:
    scheme = "wss" if args.is_ssl else "ws"
    return f"{scheme}://{args.server_ip}:{args.port}"


def build_ssl_context(args: argparse.Namespace) -> ssl.SSLContext | None:
    if not args.is_ssl:
        return None
    if args.ssl_verify:
        return ssl.create_default_context()
    context = ssl._create_unverified_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def read_hotwords(path: str) -> dict[str, int]:
    if not path:
        return {}

    hotwords: dict[str, int] = {}
    hotword_path = Path(path)
    with hotword_path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                word, weight = line.rsplit(maxsplit=1)
                hotwords[word] = int(weight)
            except ValueError as exc:
                raise ValueError(
                    f"invalid hotword at {hotword_path}:{line_no}; expected: word weight"
                ) from exc
    return hotwords


def read_wav_items(wav_path: str) -> list[WavItem]:
    path = Path(wav_path)
    if path.suffix.lower() != ".scp":
        return [WavItem(path=path, wav_id=DEFAULT_WAV_ID)]

    items: list[WavItem] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split(maxsplit=1)
            if len(columns) != 2:
                raise ValueError(f"invalid scp line at {path}:{line_no}; expected: wav_id path")
            wav_id, item_path = columns
            items.append(WavItem(path=Path(item_path), wav_id=wav_id))

    if not items:
        raise ValueError(f"no audio entries found in {path}")
    return items


def load_audio_payload(path: Path, audio_fs: int) -> AudioPayload:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return load_wav_payload(path)
    if suffix == ".pcm":
        return AudioPayload(data=path.read_bytes(), sample_rate=audio_fs, wav_format="pcm")
    return AudioPayload(data=path.read_bytes(), sample_rate=audio_fs, wav_format="others")


def load_wav_payload(path: Path) -> AudioPayload:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels <= 0:
        raise ValueError(f"{path} has no audio channels")
    return AudioPayload(
        data=wav_frames_to_pcm16(frames, channels, sample_width),
        sample_rate=sample_rate,
        wav_format="pcm",
    )


def wav_frames_to_pcm16(frames: bytes, channels: int, sample_width: int) -> bytes:
    if sample_width == 2 and channels == 1:
        return frames
    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported wav sample width: {sample_width} bytes")

    frame_width = channels * sample_width
    if frame_width <= 0 or len(frames) % frame_width != 0:
        raise ValueError("wav frame data is not aligned")

    output = bytearray((len(frames) // frame_width) * 2)
    output_offset = 0
    for frame_offset in range(0, len(frames), frame_width):
        total = 0
        for channel in range(channels):
            sample_offset = frame_offset + channel * sample_width
            total += sample_to_int16(frames[sample_offset : sample_offset + sample_width])
        mixed = max(-32768, min(32767, round(total / channels)))
        struct.pack_into("<h", output, output_offset, mixed)
        output_offset += 2
    return bytes(output)


def sample_to_int16(sample: bytes) -> int:
    sample_width = len(sample)
    if sample_width == 1:
        return (sample[0] - 128) << 8
    if sample_width == 2:
        return int.from_bytes(sample, "little", signed=True)
    if sample_width == 3:
        value = int.from_bytes(sample, "little", signed=False)
        if value & 0x800000:
            value -= 0x1000000
        return value >> 8
    if sample_width == 4:
        return int.from_bytes(sample, "little", signed=True) >> 16
    raise ValueError(f"unsupported sample width: {sample_width}")


def make_begin_message(
    args: argparse.Namespace,
    wav_id: str,
    payload: AudioPayload,
    hotwords: dict[str, int],
) -> str:
    message: dict[str, object] = {
        "mode": args.mode,
        "chunk_size": args.chunk_size,
        "chunk_interval": args.chunk_interval,
        "wav_name": wav_id,
        "wav_format": payload.wav_format,
        "audio_fs": payload.sample_rate,
        "is_speaking": True,
        "itn": bool(args.use_itn),
        "svs_itn": bool(args.svs_itn),
    }
    if hotwords:
        message["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
    return json.dumps(message, ensure_ascii=False)


def make_end_message() -> str:
    return json.dumps({"is_speaking": False}, ensure_ascii=False)


def iter_chunks(data: bytes, chunk_size: int) -> Iterable[bytes]:
    for offset in range(0, len(data), chunk_size):
        yield data[offset : offset + chunk_size]


def pcm_stream_chunk_bytes(args: argparse.Namespace, sample_rate: int) -> int:
    if args.block_size:
        return args.block_size
    seconds = 60 * args.chunk_size[1] / args.chunk_interval / 1000
    return max(2, int(seconds * sample_rate * 2))


def send_interval_seconds(
    args: argparse.Namespace,
    payload: AudioPayload,
    chunk_bytes: int,
) -> float:
    if args.send_interval_ms is not None:
        return max(0.0, args.send_interval_ms / 1000)
    if payload.wav_format != "pcm":
        return 0.0
    bytes_per_second = payload.sample_rate * 2
    return chunk_bytes / bytes_per_second


async def receive_messages(ws: object, wav_id: str, final_seen: asyncio.Event) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            print(f"[{wav_id}] received binary message: {len(message)} bytes", flush=True)
            continue

        print(f"[{wav_id}] {message}", flush=True)
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        if payload.get("is_final") is True:
            final_seen.set()
            await ws.close()
            return


async def stream_file(
    item: WavItem,
    args: argparse.Namespace,
    uri: str,
    ssl_context: ssl.SSLContext | None,
    hotwords: dict[str, int],
) -> None:
    websockets = import_websockets()
    payload = load_audio_payload(item.path, args.audio_fs)
    chunk_bytes = (
        pcm_stream_chunk_bytes(args, payload.sample_rate)
        if payload.wav_format == "pcm"
        else args.block_size or DEFAULT_OTHER_BLOCK_BYTES
    )
    interval = send_interval_seconds(args, payload, chunk_bytes)

    final_seen = asyncio.Event()
    async with websockets.connect(uri, ssl=ssl_context, max_size=None) as ws:
        receiver = asyncio.create_task(receive_messages(ws, item.wav_id, final_seen))
        await ws.send(make_begin_message(args, item.wav_id, payload, hotwords))

        for chunk in iter_chunks(payload.data, chunk_bytes):
            await ws.send(chunk)
            if interval > 0:
                await asyncio.sleep(interval)

        await ws.send(make_end_message())
        try:
            await asyncio.wait_for(final_seen.wait(), timeout=args.final_timeout)
        except asyncio.TimeoutError:
            print(
                f"[{item.wav_id}] timed out waiting for final result after "
                f"{args.final_timeout:g}s",
                file=sys.stderr,
                flush=True,
            )
            await ws.close()
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver


async def stream_microphone(
    args: argparse.Namespace,
    uri: str,
    ssl_context: ssl.SSLContext | None,
    hotwords: dict[str, int],
) -> None:
    websockets = import_websockets()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "microphone mode requires sounddevice; run `uv sync` in the repository root"
        ) from exc

    chunk_bytes = pcm_stream_chunk_bytes(args, args.audio_fs)
    block_samples = max(1, chunk_bytes // 2)
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        if status:
            print(f"[record] {status}", file=sys.stderr, flush=True)
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    payload = AudioPayload(data=b"", sample_rate=args.audio_fs, wav_format="pcm")
    final_seen = asyncio.Event()
    async with websockets.connect(uri, ssl=ssl_context, max_size=None) as ws:
        receiver = asyncio.create_task(receive_messages(ws, "record", final_seen))
        await ws.send(make_begin_message(args, "record", payload, hotwords))
        started_at = time.monotonic()

        try:
            with sd.RawInputStream(
                samplerate=args.audio_fs,
                channels=1,
                dtype="int16",
                blocksize=block_samples,
                callback=callback,
            ):
                while True:
                    if args.record_seconds > 0 and time.monotonic() - started_at >= args.record_seconds:
                        break
                    await ws.send(await queue.get())
        except asyncio.CancelledError:
            raise
        finally:
            await ws.send(make_end_message())
            try:
                await asyncio.wait_for(final_seen.wait(), timeout=args.final_timeout)
            except asyncio.TimeoutError:
                await ws.close()
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver


def import_websockets() -> object:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("missing dependency: run `uv sync` in the repository root") from exc
    return websockets


async def run_file_mode(
    args: argparse.Namespace,
    uri: str,
    ssl_context: ssl.SSLContext | None,
    hotwords: dict[str, int],
) -> None:
    items = read_wav_items(args.wav_path)
    for start in range(0, len(items), args.thread_num):
        batch = items[start : start + args.thread_num]
        await asyncio.gather(
            *(stream_file(item, args, uri, ssl_context, hotwords) for item in batch)
        )


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    uri = build_uri(args)
    ssl_context = build_ssl_context(args)
    hotwords = read_hotwords(args.hotword)

    if args.record:
        await stream_microphone(args, uri, ssl_context, hotwords)
    else:
        await run_file_mode(args, uri, ssl_context, hotwords)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
