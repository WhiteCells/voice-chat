import argparse
import asyncio
import contextlib
import json
import math
import os
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets


CLEAR_PLAYBACK = object()
DEFAULT_VAD_MODEL_PATH = Path(__file__).with_name("models") / "silero_vad.onnx"


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
    vad_model_path: Path
    vad_threshold: float
    vad_end_silence_ms: int
    vad_speech_pad_ms: int
    vad_pre_roll_ms: int
    vad_min_turn_ms: int
    vad_max_turn_seconds: float
    vad_debug: bool
    vad_preprocess: bool
    vad_target_rms: float
    vad_rescue: bool
    vad_rescue_min_rms: float
    vad_rescue_min_peak: float
    vad_rescue_start_ms: int
    vad_noise_gate_rms: float
    asr_preprocess: bool
    asr_target_rms: float
    allow_interrupt: bool
    interrupt_min_turn_ms: int

    @property
    def input_block_frames(self) -> int:
        return max(1, self.input_sample_rate * self.block_ms // 1000)

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width

    @property
    def vad_window_samples(self) -> int:
        return 512 if self.input_sample_rate == 16000 else 256

    @property
    def vad_pre_roll_blocks(self) -> int:
        return max(0, math.ceil(self.vad_pre_roll_ms / self.block_ms))


@dataclass
class ClientState:
    response_active: bool = False
    response_turn_id: int | None = None
    discard_audio: bool = False
    discard_turn_id: int | None = None

    def mark_response_started(self, turn_id: int | None) -> None:
        self.response_active = True
        self.response_turn_id = turn_id
        if turn_id != self.discard_turn_id:
            self.discard_audio = False

    def mark_response_finished(self, turn_id: int | None) -> None:
        if turn_id is None or turn_id == self.response_turn_id or turn_id == self.discard_turn_id:
            self.response_active = False
            self.response_turn_id = None
        if turn_id is None or turn_id == self.discard_turn_id:
            self.discard_audio = False
            self.discard_turn_id = None

    def mark_interrupted(self, player: "RawAudioPlayer") -> None:
        self.discard_turn_id = self.response_turn_id
        self.discard_audio = True
        self.response_active = False
        player.clear()


class RawAudioPlayer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks: queue.Queue[bytes | object | None] = queue.Queue(maxsize=200)
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

    def clear(self) -> None:
        with self.chunks.mutex:
            self.chunks.queue.clear()
            self.chunks.not_full.notify_all()
        self.chunks.put(CLEAR_PLAYBACK)

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
                    if chunk is CLEAR_PLAYBACK:
                        pending = b""
                        continue
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


class NeuralVad:
    def __init__(self, settings: Settings) -> None:
        try:
            import numpy as np
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("missing neural VAD dependency: run `uv sync` in the repository root") from exc

        self.np = np
        self.settings = settings
        self.threshold = settings.vad_threshold
        self.neg_threshold = max(settings.vad_threshold - 0.15, 0.01)
        self.window_samples = settings.vad_window_samples
        self.context_samples = 64 if settings.input_sample_rate == 16000 else 32
        self.end_silence_samples = settings.input_sample_rate * settings.vad_end_silence_ms / 1000
        self.speech_pad_samples = settings.input_sample_rate * settings.vad_speech_pad_ms / 1000
        self.rescue_start_samples = settings.input_sample_rate * settings.vad_rescue_start_ms / 1000

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(
            str(settings.vad_model_path),
            sess_options=options,
            providers=providers,
        )
        self.last_prob = 0.0
        self.last_rms = 0.0
        self.last_peak = 0.0
        self.last_vad_rms = 0.0
        self.last_vad_peak = 0.0
        self.last_zcr = 0.0
        self.last_flatness = 0.0
        self.last_rescue = False
        self.reset()

    def process(self, chunk: bytes) -> dict[str, int] | None:
        audio = self.np.frombuffer(chunk, dtype="<i2").astype(self.np.float32) / 32768.0
        if audio.shape[0] != self.window_samples:
            raise ValueError(f"expected {self.window_samples} samples, got {audio.shape[0]}")
        self.last_rms = float(self.np.sqrt(self.np.mean(audio * audio)))
        self.last_peak = float(self.np.max(self.np.abs(audio))) if audio.size else 0.0

        vad_audio = self._prepare_audio(audio)
        self.last_vad_rms = float(self.np.sqrt(self.np.mean(vad_audio * vad_audio)))
        self.last_vad_peak = float(self.np.max(self.np.abs(vad_audio))) if vad_audio.size else 0.0
        self.last_zcr, self.last_flatness = self._voice_shape(vad_audio)
        self.current_sample += self.window_samples
        model_input = self.np.concatenate([self.context, vad_audio.reshape(1, -1)], axis=1)
        output, state = self.session.run(
            None,
            {
                "input": model_input,
                "state": self.state,
                "sr": self.np.array(self.settings.input_sample_rate, dtype=self.np.int64),
            },
        )
        self.state = state
        self.context = model_input[:, -self.context_samples :]
        speech_prob = float(output.squeeze())
        self.last_prob = speech_prob
        rescue_active = self._looks_like_clipped_speech()
        audible = self.last_rms >= self.settings.vad_noise_gate_rms
        self.last_rescue = rescue_active

        if (speech_prob >= self.threshold or rescue_active) and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold or rescue_active:
            self.rescue_speech_samples += self.window_samples
        else:
            self.rescue_speech_samples = 0

        should_start = audible and (
            speech_prob >= self.threshold or (
                rescue_active
                and self.settings.vad_rescue
                and self.rescue_speech_samples >= self.rescue_start_samples
            )
        )

        if should_start and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - self.window_samples)
            return {"start": int(speech_start)}

        should_end = self.triggered and speech_prob < self.neg_threshold and not rescue_active
        if should_end:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.end_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - self.window_samples
            self.temp_end = 0
            self.triggered = False
            return {"end": int(speech_end)}

        return None

    def _prepare_audio(self, audio: Any) -> Any:
        if not self.settings.vad_preprocess:
            return audio

        prepared = audio - float(self.np.mean(audio))
        rms = float(self.np.sqrt(self.np.mean(prepared * prepared)))
        if rms > self.settings.vad_target_rms:
            prepared = prepared * (self.settings.vad_target_rms / rms)

        peak = float(self.np.max(self.np.abs(prepared))) if prepared.size else 0.0
        if peak > 0.98:
            prepared = prepared * (0.98 / peak)
        return prepared.astype(self.np.float32, copy=False)

    def _voice_shape(self, audio: Any) -> tuple[float, float]:
        if audio.size < 2:
            return 0.0, 1.0
        signs = self.np.signbit(audio)
        zcr = float(self.np.mean(signs[1:] != signs[:-1]))
        windowed = audio * self.np.hanning(audio.shape[0]).astype(self.np.float32)
        spectrum = self.np.abs(self.np.fft.rfft(windowed)) + 1e-8
        flatness = float(self.np.exp(self.np.mean(self.np.log(spectrum))) / self.np.mean(spectrum))
        return zcr, flatness

    def _looks_like_clipped_speech(self) -> bool:
        if not self.settings.vad_rescue:
            return False
        return (
            self.last_prob < self.threshold
            and self.last_rms >= self.settings.vad_rescue_min_rms
            and self.last_peak >= self.settings.vad_rescue_min_peak
            and self.last_zcr <= 0.18
            and self.last_flatness <= 0.45
        )

    def reset(self) -> None:
        self.state = self.np.zeros((2, 1, 128), dtype=self.np.float32)
        self.context = self.np.zeros((1, self.context_samples), dtype=self.np.float32)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.rescue_speech_samples = 0


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


def env_str_nonempty(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))


def preprocess_pcm16(chunk: bytes, target_rms: float) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("missing dependency: run `uv sync` in the repository root") from exc

    audio = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        return chunk

    prepared = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(prepared * prepared)))
    if rms > target_rms:
        prepared = prepared * (target_rms / rms)

    peak = float(np.max(np.abs(prepared))) if prepared.size else 0.0
    if peak > 0.98:
        prepared = prepared * (0.98 / peak)
    return (np.clip(prepared, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def device_label(sd: Any, device: str | int | None, kind: str) -> str:
    try:
        info = sd.query_devices(device, kind=kind)
    except Exception as exc:
        label = "default" if device is None else str(device)
        return f"{label} ({exc})"

    name = info.get("name", "unknown")
    index = info.get("index", "?")
    samplerate = info.get("default_samplerate")
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    channels = info.get(channel_key, "?")
    rate_label = f"{int(samplerate)} Hz" if isinstance(samplerate, (int, float)) else "? Hz"
    return f"{index}:{name} ({channels} ch, default {rate_label})"


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
    parser.add_argument("--block-ms", type=int, default=env_int("VCLIENT_BLOCK_MS", 32))
    parser.add_argument(
        "--vad-model-path",
        default=env_str_nonempty("VCLIENT_VAD_MODEL_PATH", str(DEFAULT_VAD_MODEL_PATH)),
        help="Silero VAD ONNX model path",
    )
    parser.add_argument("--vad-threshold", type=float, default=env_float("VCLIENT_VAD_THRESHOLD", 0.35))
    parser.add_argument(
        "--vad-end-silence-ms",
        "--vad-min-silence-ms",
        dest="vad_end_silence_ms",
        type=int,
        default=env_int(
            "VCLIENT_VAD_END_SILENCE_MS",
            env_int("VCLIENT_VAD_MIN_SILENCE_MS", 420),
        ),
        help="silence debounce used by the neural VAD state machine before emitting an end event",
    )
    parser.add_argument("--vad-speech-pad-ms", type=int, default=env_int("VCLIENT_VAD_SPEECH_PAD_MS", 64))
    parser.add_argument("--vad-pre-roll-ms", type=int, default=env_int("VCLIENT_VAD_PRE_ROLL_MS", 256))
    parser.add_argument("--vad-min-turn-ms", type=int, default=env_int("VCLIENT_VAD_MIN_TURN_MS", 260))
    parser.add_argument(
        "--vad-max-turn-seconds",
        type=float,
        default=env_float("VCLIENT_VAD_MAX_TURN_SECONDS", 0.0),
        help="force-finish one utterance after this many seconds; 0 disables the limit",
    )
    parser.add_argument(
        "--vad-debug",
        type=int,
        choices=(0, 1),
        default=int(env_bool("VCLIENT_VAD_DEBUG", False)),
        help="print live mic level and neural VAD probability",
    )
    parser.add_argument(
        "--vad-preprocess",
        type=int,
        choices=(0, 1),
        default=int(env_bool("VCLIENT_VAD_PREPROCESS", True)),
        help="normalize microphone audio before neural VAD",
    )
    parser.add_argument(
        "--vad-target-rms",
        type=float,
        default=env_float("VCLIENT_VAD_TARGET_RMS", 0.08),
        help="target RMS used when reducing clipped microphone input before VAD",
    )
    parser.add_argument(
        "--vad-rescue",
        type=int,
        choices=(0, 1),
        default=int(env_bool("VCLIENT_VAD_RESCUE", True)),
        help="fallback trigger for clipped speech when neural VAD stays low",
    )
    parser.add_argument(
        "--vad-rescue-min-rms",
        type=float,
        default=env_float("VCLIENT_VAD_RESCUE_MIN_RMS", 0.08),
        help="minimum raw RMS required before clipped-speech fallback can trigger",
    )
    parser.add_argument(
        "--vad-rescue-min-peak",
        type=float,
        default=env_float("VCLIENT_VAD_RESCUE_MIN_PEAK", 0.30),
        help="minimum raw peak required before clipped-speech fallback can trigger",
    )
    parser.add_argument(
        "--vad-rescue-start-ms",
        type=int,
        default=env_int("VCLIENT_VAD_RESCUE_START_MS", 160),
        help="consecutive clipped-speech duration required before fallback trigger",
    )
    parser.add_argument(
        "--vad-noise-gate-rms",
        type=float,
        default=env_float("VCLIENT_VAD_NOISE_GATE_RMS", 0.018),
        help="raw RMS below this value is treated as silence while a turn is active",
    )
    parser.add_argument(
        "--asr-preprocess",
        type=int,
        choices=(0, 1),
        default=int(env_bool("VCLIENT_ASR_PREPROCESS", True)),
        help="normalize microphone audio before sending it to ASR",
    )
    parser.add_argument(
        "--asr-target-rms",
        type=float,
        default=env_float("VCLIENT_ASR_TARGET_RMS", 0.12),
        help="target RMS used when reducing clipped microphone input before ASR",
    )
    parser.add_argument(
        "--allow-interrupt",
        type=int,
        choices=(0, 1),
        default=int(env_bool("VCLIENT_ALLOW_INTERRUPT", True)),
    )
    parser.add_argument(
        "--interrupt-min-turn-ms",
        type=int,
        default=env_int("VCLIENT_INTERRUPT_MIN_TURN_MS", 220),
        help="speech duration required before cancelling an active response",
    )
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
    vad_model_path = Path(args.vad_model_path).expanduser()
    if not vad_model_path.exists():
        parser.error(f"--vad-model-path does not exist: {vad_model_path}")
    if args.input_sample_rate not in {8000, 16000}:
        parser.error("--input-sample-rate must be 8000 or 16000 for Silero VAD")
    if args.output_sample_rate <= 0:
        parser.error("--output-sample-rate must be positive")
    if args.channels != 1:
        parser.error("--channels must be 1 because Silero VAD expects mono audio")
    if args.sample_width != 2:
        parser.error("--sample-width must be 2 because vclient records 16-bit PCM")
    if args.block_ms <= 0:
        parser.error("--block-ms must be positive")
    expected_frames = 512 if args.input_sample_rate == 16000 else 256
    actual_frames = args.input_sample_rate * args.block_ms // 1000
    if actual_frames != expected_frames:
        parser.error(
            f"--block-ms must produce {expected_frames} samples per VAD frame "
            f"for sample_rate={args.input_sample_rate}"
        )
    if not 0 < args.vad_threshold < 1:
        parser.error("--vad-threshold must be between 0 and 1")
    if args.vad_end_silence_ms <= 0:
        parser.error("--vad-end-silence-ms must be positive")
    if args.vad_speech_pad_ms < 0:
        parser.error("--vad-speech-pad-ms must be greater than or equal to 0")
    if args.vad_pre_roll_ms < 0:
        parser.error("--vad-pre-roll-ms must be greater than or equal to 0")
    if args.vad_min_turn_ms <= 0:
        parser.error("--vad-min-turn-ms must be positive")
    if args.vad_max_turn_seconds < 0:
        parser.error("--vad-max-turn-seconds must be greater than or equal to 0")
    if not 0 < args.vad_target_rms <= 1:
        parser.error("--vad-target-rms must be between 0 and 1")
    if not 0 < args.vad_rescue_min_rms <= 1:
        parser.error("--vad-rescue-min-rms must be between 0 and 1")
    if not 0 < args.vad_rescue_min_peak <= 1:
        parser.error("--vad-rescue-min-peak must be between 0 and 1")
    if args.vad_rescue_start_ms <= 0:
        parser.error("--vad-rescue-start-ms must be positive")
    if not 0 < args.vad_noise_gate_rms <= 1:
        parser.error("--vad-noise-gate-rms must be between 0 and 1")
    if not 0 < args.asr_target_rms <= 1:
        parser.error("--asr-target-rms must be between 0 and 1")
    if args.interrupt_min_turn_ms <= 0:
        parser.error("--interrupt-min-turn-ms must be positive")

    return Settings(
        uri=args.uri,
        input_sample_rate=args.input_sample_rate,
        output_sample_rate=args.output_sample_rate,
        channels=args.channels,
        sample_width=args.sample_width,
        block_ms=args.block_ms,
        input_device=parse_device(args.input_device),
        output_device=parse_device(args.output_device),
        vad_model_path=vad_model_path,
        vad_threshold=args.vad_threshold,
        vad_end_silence_ms=args.vad_end_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        vad_pre_roll_ms=args.vad_pre_roll_ms,
        vad_min_turn_ms=args.vad_min_turn_ms,
        vad_max_turn_seconds=args.vad_max_turn_seconds,
        vad_debug=bool(args.vad_debug),
        vad_preprocess=bool(args.vad_preprocess),
        vad_target_rms=args.vad_target_rms,
        vad_rescue=bool(args.vad_rescue),
        vad_rescue_min_rms=args.vad_rescue_min_rms,
        vad_rescue_min_peak=args.vad_rescue_min_peak,
        vad_rescue_start_ms=args.vad_rescue_start_ms,
        vad_noise_gate_rms=args.vad_noise_gate_rms,
        asr_preprocess=bool(args.asr_preprocess),
        asr_target_rms=args.asr_target_rms,
        allow_interrupt=bool(args.allow_interrupt),
        interrupt_min_turn_ms=args.interrupt_min_turn_ms,
    )


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def payload_turn_id(payload: dict[str, Any]) -> int | None:
    raw = payload.get("turn_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def send_audio_start(ws: Any, settings: Settings) -> None:
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


async def receive_loop(ws: Any, player: RawAudioPlayer, state: ClientState) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            if not state.discard_audio:
                player.write(message)
            continue

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(f"[server] {message}", flush=True)
            continue

        message_type = payload.get("type")
        turn_id = payload_turn_id(payload)
        if message_type == "ready":
            print(
                "[ready] "
                f"ASR {payload.get('asr_sample_rate')} Hz, "
                f"TTS {payload.get('tts_sample_rate')} Hz",
                flush=True,
            )
        elif message_type == "turn_started":
            print(f"[turn] started #{turn_id}", flush=True)
        elif message_type == "asr_partial":
            print(f"\r[asr] {payload.get('text', '')}", end="", flush=True)
        elif message_type == "asr_final":
            print(f"\n[you] {payload.get('text', '')}", flush=True)
        elif message_type == "llm_start":
            state.mark_response_started(turn_id)
            print("[assistant] ", end="", flush=True)
        elif message_type == "llm_delta":
            print(payload.get("text", ""), end="", flush=True)
        elif message_type == "llm_done":
            print("", flush=True)
        elif message_type == "tts_start":
            state.mark_response_started(turn_id)
        elif message_type == "turn_done":
            state.mark_response_finished(turn_id)
        elif message_type == "response_cancelled":
            state.mark_response_finished(turn_id)
            print("\n[interrupt] 已打断上一轮回复", flush=True)
        elif message_type == "warning":
            print(f"\n[warning] {payload.get('message', '')}", flush=True)
        elif message_type == "error":
            state.mark_response_finished(turn_id)
            print(f"\n[error] {payload.get('message', '')}", flush=True)
        else:
            print(f"\n[{message_type}] {payload}", flush=True)


async def vad_record_loop(ws: Any, settings: Settings, player: RawAudioPlayer, state: ClientState) -> None:
    sd = import_sounddevice()
    vad = NeuralVad(settings)
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)

    pre_roll: deque[bytes] = deque(maxlen=settings.vad_pre_roll_blocks)
    buffered_chunks: list[bytes] = []
    speech_active = False
    audio_started = False
    deferred_interrupt = False
    ignored_segment = False
    recorded_ms = 0
    speech_ms = 0
    max_turn_ms = int(settings.vad_max_turn_seconds * 1000)
    debug_elapsed_ms = 0
    debug_max_prob = 0.0
    debug_max_peak = 0.0
    silent_input_ms = 0
    voiced_but_untriggered_ms = 0
    silent_input_warned = False
    low_vad_warned = False

    def chunk_for_asr(chunk: bytes) -> bytes:
        if not settings.asr_preprocess:
            return chunk
        return preprocess_pcm16(chunk, settings.asr_target_rms)

    def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        if status:
            print(f"[record] {status}", file=sys.stderr, flush=True)
        data = bytes(indata)

        def enqueue() -> None:
            if chunks.full():
                chunks.get_nowait()
            chunks.put_nowait(data)

        loop.call_soon_threadsafe(enqueue)

    async def open_audio_turn(interrupted: bool) -> None:
        nonlocal audio_started
        if interrupted:
            state.mark_interrupted(player)
            await ws.send(json_dumps({"type": "cancel_response", "reason": "barge_in"}))
            print("\n[interrupt] 检测到用户插话，停止播报", flush=True)
        print(
            f"[vad] 检测到语音，prob={vad.last_prob:.3f}, "
            f"rms={dbfs(vad.last_rms):.1f} dBFS，"
            f"mode={'rescue' if vad.last_rescue else 'neural'}，开始发送 audio_start",
            flush=True,
        )
        await send_audio_start(ws, settings)
        for buffered_chunk in buffered_chunks:
            await ws.send(buffered_chunk)
        audio_started = True

    async def finish_segment(reason: str) -> None:
        nonlocal audio_started
        nonlocal buffered_chunks
        nonlocal deferred_interrupt
        nonlocal ignored_segment
        nonlocal recorded_ms
        nonlocal speech_active
        nonlocal speech_ms
        if audio_started:
            await ws.send(json_dumps({"type": "audio_end"}))
            print(f"[vad] {reason}，提交本轮", flush=True)
        elif deferred_interrupt:
            print("[interrupt] 插话太短，忽略", flush=True)
        speech_active = False
        audio_started = False
        deferred_interrupt = False
        ignored_segment = False
        buffered_chunks = []
        recorded_ms = 0
        speech_ms = 0
        pre_roll.clear()

    print(
        "监听中，Silero 神经网络 VAD 已启用；开口说话即可，Ctrl+C 退出...",
        flush=True,
    )
    print(
        f"[record] input={device_label(sd, settings.input_device, 'input')}, "
        f"sample_rate={settings.input_sample_rate}, block={settings.vad_window_samples} samples",
        flush=True,
    )
    print(
        f"[vad] model={settings.vad_model_path}, threshold={settings.vad_threshold:.2f}, "
        f"vad_end_debounce={settings.vad_end_silence_ms} ms, "
        f"max_turn={'off' if not max_turn_ms else f'{settings.vad_max_turn_seconds:.1f}s'}, "
        f"start_gate={dbfs(settings.vad_noise_gate_rms):.1f} dBFS, "
        f"preprocess={'on' if settings.vad_preprocess else 'off'}, "
        f"rescue={'on' if settings.vad_rescue else 'off'}, "
        f"asr_preprocess={'on' if settings.asr_preprocess else 'off'}, "
        f"interrupt={'on' if settings.allow_interrupt else 'off'}",
        flush=True,
    )
    try:
        with sd.RawInputStream(
            samplerate=settings.input_sample_rate,
            channels=settings.channels,
            dtype="int16",
            blocksize=settings.vad_window_samples,
            latency="low",
            device=settings.input_device,
            callback=callback,
        ):
            while True:
                chunk = await chunks.get()
                asr_chunk = chunk_for_asr(chunk)
                pre_roll.append(asr_chunk)
                event = vad.process(chunk)
                debug_elapsed_ms += settings.block_ms
                debug_max_prob = max(debug_max_prob, vad.last_prob)
                debug_max_peak = max(debug_max_peak, vad.last_peak)

                if vad.last_peak < 0.002:
                    silent_input_ms += settings.block_ms
                else:
                    silent_input_ms = 0

                if (
                    not speech_active
                    and vad.last_peak > 0.025
                    and vad.last_rms > 0.004
                    and vad.last_prob < settings.vad_threshold
                ):
                    voiced_but_untriggered_ms += settings.block_ms
                else:
                    voiced_but_untriggered_ms = 0

                if settings.vad_debug and debug_elapsed_ms >= 480:
                    print(
                        f"[vad-debug] prob={vad.last_prob:.3f}, max_prob={debug_max_prob:.3f}, "
                        f"rms={dbfs(vad.last_rms):.1f} dBFS, peak={dbfs(debug_max_peak):.1f} dBFS, "
                        f"vad_rms={dbfs(vad.last_vad_rms):.1f} dBFS, "
                        f"zcr={vad.last_zcr:.3f}, flat={vad.last_flatness:.3f}, "
                        f"rescue={'1' if vad.last_rescue else '0'}, "
                        f"state={'speech' if speech_active else 'idle'}",
                        flush=True,
                    )
                    debug_elapsed_ms = 0
                    debug_max_prob = 0.0
                    debug_max_peak = 0.0

                if not silent_input_warned and silent_input_ms >= 3000:
                    print(
                        "[record] 麦克风输入接近静音超过 3 秒；如果你正在说话，请检查 VCLIENT_INPUT_DEVICE 或系统输入音量",
                        flush=True,
                    )
                    silent_input_warned = True

                if not low_vad_warned and voiced_but_untriggered_ms >= 800:
                    print(
                        f"[vad] 检测到麦克风有音量，但神经 VAD 分数低于阈值 "
                        f"({vad.last_prob:.3f} < {settings.vad_threshold:.2f})；"
                        "如果没有触发 rescue，请降低系统输入音量或检查输入设备是否为真实麦克风",
                        flush=True,
                    )
                    low_vad_warned = True

                if not speech_active:
                    if not event or "start" not in event:
                        continue
                    speech_active = True
                    buffered_chunks = list(pre_roll)
                    recorded_ms = len(buffered_chunks) * settings.block_ms
                    speech_ms = settings.block_ms
                    response_active = state.response_active or state.response_turn_id is not None
                    ignored_segment = response_active and not settings.allow_interrupt
                    deferred_interrupt = (
                        response_active
                        and settings.allow_interrupt
                        and speech_ms < settings.interrupt_min_turn_ms
                    )
                    if ignored_segment:
                        print("[vad] 回复播放中，当前插话被配置忽略", flush=True)
                    elif deferred_interrupt:
                        print("[interrupt] 检测到插话，确认中...", flush=True)
                    else:
                        await open_audio_turn(interrupted=response_active)
                    continue

                speech_ms += settings.block_ms
                if not ignored_segment:
                    if audio_started:
                        await ws.send(asr_chunk)
                        recorded_ms += settings.block_ms
                    else:
                        buffered_chunks.append(asr_chunk)
                        recorded_ms += settings.block_ms
                        if deferred_interrupt and speech_ms >= settings.interrupt_min_turn_ms:
                            await open_audio_turn(interrupted=True)

                if event and "end" in event:
                    if recorded_ms < settings.vad_min_turn_ms and not ignored_segment:
                        await finish_segment("语音过短")
                    else:
                        await finish_segment("收到 VAD end")
                    continue

                if max_turn_ms and recorded_ms >= max_turn_ms:
                    await finish_segment(f"达到最长单轮 {settings.vad_max_turn_seconds:.1f}s")
                    vad.reset()
    finally:
        if audio_started:
            await ws.send(json_dumps({"type": "audio_end"}))


async def async_main(argv: list[str] | None = None) -> None:
    settings = parse_args(argv)
    player = RawAudioPlayer(settings)
    player.start()
    state = ClientState()
    print(f"connecting to {settings.uri}")

    try:
        async with websockets.connect(settings.uri, max_size=None) as ws:
            receiver = asyncio.create_task(receive_loop(ws, player, state))
            recorder = asyncio.create_task(vad_record_loop(ws, settings, player, state))
            try:
                done, pending = await asyncio.wait(
                    {receiver, recorder},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    task.result()
            finally:
                for task in (receiver, recorder):
                    task.cancel()
                for task in (receiver, recorder):
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
    finally:
        player.stop()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
