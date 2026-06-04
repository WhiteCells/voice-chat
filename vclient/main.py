import argparse
import asyncio
import contextlib
import json
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets


@dataclass(frozen=True)
class Settings:
    uri: str
    input_sample_rate: int
    output_sample_rate: int
    channels: int
    sample_width: int
    block_ms: int
    input_device: str | int | None
    output_device: str | int | None

    @property
    def input_block_frames(self) -> int:
        return max(1, self.input_sample_rate * self.block_ms // 1000)

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width


class RawAudioPlayer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self.thread = threading.Thread(target=self._run, name="vchat-playback", daemon=True)
        self.started = threading.Event()
        self.errors: queue.Queue[BaseException] = queue.Queue()

    def start(self) -> None:
        self.thread.start()
        self.started.wait(timeout=5)
        if not self.errors.empty():
            raise self.errors.get()

    def write(self, chunk: bytes) -> None:
        self.chunks.put(chunk)

    def stop(self) -> None:
        self.chunks.put(None)
        self.thread.join(timeout=5)

    def _run(self) -> None:
        sd = import_sounddevice()
        pending = b""
        try:
            with sd.RawOutputStream(
                samplerate=self.settings.output_sample_rate,
                channels=self.settings.channels,
                dtype="int16",
                latency="low",
                device=self.settings.output_device,
            ) as stream:
                self.started.set()
                while True:
                    chunk = self.chunks.get()
                    if chunk is None:
                        break
                    pending += chunk
                    writable = len(pending) - (len(pending) % self.settings.frame_bytes)
                    if writable:
                        stream.write(pending[:writable])
                        pending = pending[writable:]
                if pending:
                    stream.write(pending.ljust(self.settings.frame_bytes, b"\x00"))
        except BaseException as exc:
            self.errors.put(exc)
            self.started.set()


def parse_device(value: str | None) -> str | int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("missing dependency: run `uv sync` in the repository root") from exc
    return sd


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


def parse_args(argv: list[str] | None = None) -> Settings:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.environ.get("VCLIENT_ENV_FILE", ".env"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Local microphone/speaker client for vchat.")
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument("--uri", default=env_str("VCHAT_URI", "ws://127.0.0.1:8765"))
    parser.add_argument("--input-sample-rate", type=int, default=env_int("VCLIENT_INPUT_SAMPLE_RATE", 16000))
    parser.add_argument("--output-sample-rate", type=int, default=env_int("VCLIENT_OUTPUT_SAMPLE_RATE", 24000))
    parser.add_argument("--channels", type=int, default=env_int("VCLIENT_CHANNELS", 1))
    parser.add_argument("--sample-width", type=int, default=env_int("VCLIENT_SAMPLE_WIDTH", 2))
    parser.add_argument("--block-ms", type=int, default=env_int("VCLIENT_BLOCK_MS", 40))
    parser.add_argument(
        "--input-device",
        default=env_str("VCLIENT_INPUT_DEVICE", ""),
        help="sounddevice input device id or name",
    )
    parser.add_argument(
        "--output-device",
        default=env_str("VCLIENT_OUTPUT_DEVICE", ""),
        help="sounddevice output device id or name",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        sd = import_sounddevice()
        print(sd.query_devices())
        raise SystemExit(0)
    if args.input_sample_rate <= 0:
        parser.error("--input-sample-rate must be positive")
    if args.output_sample_rate <= 0:
        parser.error("--output-sample-rate must be positive")
    if args.channels <= 0:
        parser.error("--channels must be positive")
    if args.sample_width <= 0:
        parser.error("--sample-width must be positive")
    if args.block_ms <= 0:
        parser.error("--block-ms must be positive")

    return Settings(
        uri=args.uri,
        input_sample_rate=args.input_sample_rate,
        output_sample_rate=args.output_sample_rate,
        channels=args.channels,
        sample_width=args.sample_width,
        block_ms=args.block_ms,
        input_device=parse_device(args.input_device),
        output_device=parse_device(args.output_device),
    )


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def wait_for_enter(prompt: str) -> None:
    await asyncio.to_thread(input, prompt)


async def record_turn(ws: Any, settings: Settings) -> None:
    sd = import_sounddevice()
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    stopped = asyncio.Event()

    def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        if status:
            print(f"[record] {status}", file=sys.stderr, flush=True)
        data = bytes(indata)

        def enqueue() -> None:
            if chunks.full():
                chunks.get_nowait()
            chunks.put_nowait(data)

        loop.call_soon_threadsafe(enqueue)

    async def sender() -> None:
        while not stopped.is_set() or not chunks.empty():
            try:
                chunk = await asyncio.wait_for(chunks.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await ws.send(chunk)

    await ws.send(
        json_dumps(
            {
                "type": "audio_start",
                "sample_rate": settings.input_sample_rate,
                "channels": settings.channels,
                "sample_width": settings.sample_width,
            }
        )
    )

    sender_task = asyncio.create_task(sender())
    try:
        with sd.RawInputStream(
            samplerate=settings.input_sample_rate,
            channels=settings.channels,
            dtype="int16",
            blocksize=settings.input_block_frames,
            latency="low",
            device=settings.input_device,
            callback=callback,
        ):
            await wait_for_enter("录音中，再按 Enter 结束本轮...")
    finally:
        stopped.set()
        await sender_task
        await ws.send(json_dumps({"type": "audio_end"}))


async def receive_loop(ws: Any, player: RawAudioPlayer, turn_done: asyncio.Event) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            player.write(message)
            continue

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(f"[server] {message}", flush=True)
            continue

        message_type = payload.get("type")
        if message_type == "ready":
            print(
                "[ready] "
                f"ASR {payload.get('asr_sample_rate')} Hz, "
                f"TTS {payload.get('tts_sample_rate')} Hz",
                flush=True,
            )
        elif message_type == "asr_partial":
            print(f"\r[asr] {payload.get('text', '')}", end="", flush=True)
        elif message_type == "asr_final":
            print(f"\n[you] {payload.get('text', '')}", flush=True)
        elif message_type == "llm_delta":
            print(payload.get("text", ""), end="", flush=True)
        elif message_type == "llm_start":
            print("[assistant] ", end="", flush=True)
        elif message_type == "llm_done":
            print("", flush=True)
        elif message_type == "turn_done":
            turn_done.set()
        elif message_type == "warning":
            print(f"\n[warning] {payload.get('message', '')}", flush=True)
        elif message_type == "error":
            print(f"\n[error] {payload.get('message', '')}", flush=True)
            turn_done.set()
        else:
            print(f"\n[{message_type}] {payload}", flush=True)


async def async_main(argv: list[str] | None = None) -> None:
    settings = parse_args(argv)
    player = RawAudioPlayer(settings)
    player.start()
    print(f"connecting to {settings.uri}")

    try:
        async with websockets.connect(settings.uri, max_size=None) as ws:
            turn_done = asyncio.Event()
            receiver = asyncio.create_task(receive_loop(ws, player, turn_done))
            try:
                while True:
                    await wait_for_enter("按 Enter 开始录音，Ctrl+C 退出...")
                    turn_done.clear()
                    await record_turn(ws, settings)
                    print("等待回复...")
                    await turn_done.wait()
            finally:
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
    finally:
        player.stop()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
