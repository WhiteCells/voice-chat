import argparse
import asyncio
import contextlib
import json
import os
import ssl
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
import websockets


DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Reply in concise Chinese."
DEFAULT_ASR_CHUNK_SIZE = "5-10-5"
TEXT_FLUSH_PUNCTUATION = set("。！？；.!?;\n")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    asr_host: str
    asr_port: int
    asr_ssl: bool
    asr_ssl_verify: bool
    asr_proxy: str | bool | None
    asr_mode: str
    asr_sample_rate: int
    asr_chunk_size: list[int]
    asr_chunk_interval: int
    asr_final_timeout: float
    use_itn: bool
    svs_itn: bool
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_system_prompt: str
    llm_max_tokens: int
    llm_temperature: float
    llm_top_p: float
    llm_thinking_disabled: bool
    tts_base_url: str
    tts_model: str
    tts_voice: str
    tts_sample_rate: int
    tts_language: str
    tts_task_type: str
    tts_chunk_size: int
    tts_min_chars: int
    tts_flush_chars: int

    @property
    def asr_uri(self) -> str:
        scheme = "wss" if self.asr_ssl else "ws"
        return f"{scheme}://{self.asr_host}:{self.asr_port}"

    @property
    def tts_url(self) -> str:
        return f"{self.tts_base_url.rstrip('/')}/v1/audio/speech"

    @property
    def asr_proxy_label(self) -> str:
        if self.asr_proxy is True:
            return "auto"
        if self.asr_proxy is None:
            return "disabled"
        return self.asr_proxy


class AsrSession:
    def __init__(self, settings: Settings, turn_id: int, client_ws: Any) -> None:
        self.settings = settings
        self.turn_id = turn_id
        self.client_ws = client_ws
        self.ws: Any | None = None
        self.receiver: asyncio.Task[None] | None = None
        self.final_seen = asyncio.Event()
        self.final_text = ""
        self.last_text = ""
        self.started_at = time.perf_counter()

    async def __aenter__(self) -> "AsrSession":
        self.ws = await websockets.connect(
            self.settings.asr_uri,
            ssl=build_asr_ssl_context(self.settings),
            max_size=None,
            proxy=self.settings.asr_proxy,
        )
        await self.ws.send(json_dumps(self._begin_message()))
        self.receiver = asyncio.create_task(self._receive_asr_messages())
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.receiver:
            self.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.receiver
        if self.ws:
            await self.ws.close()

    async def send_audio(self, chunk: bytes) -> None:
        if not self.ws:
            raise RuntimeError("ASR session is not open")
        await self.ws.send(chunk)

    async def finish(self) -> str:
        if not self.ws:
            raise RuntimeError("ASR session is not open")
        await self.ws.send(json_dumps({"is_speaking": False}))
        try:
            await asyncio.wait_for(
                self.final_seen.wait(),
                timeout=self.settings.asr_final_timeout,
            )
        except asyncio.TimeoutError:
            await send_event(
                self.client_ws,
                "asr_timeout",
                text=self.last_text,
                timeout=self.settings.asr_final_timeout,
            )
        return (self.final_text or self.last_text).strip()

    def _begin_message(self) -> dict[str, Any]:
        return {
            "mode": self.settings.asr_mode,
            "chunk_size": self.settings.asr_chunk_size,
            "chunk_interval": self.settings.asr_chunk_interval,
            "wav_name": f"turn-{self.turn_id}",
            "wav_format": "pcm",
            "audio_fs": self.settings.asr_sample_rate,
            "is_speaking": True,
            "itn": self.settings.use_itn,
            "svs_itn": self.settings.svs_itn,
        }

    async def _receive_asr_messages(self) -> None:
        assert self.ws is not None
        async for message in self.ws:
            if isinstance(message, bytes):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await send_event(self.client_ws, "asr_message", raw=message)
                continue

            text = str(payload.get("text") or "").strip()
            if text:
                self.last_text = text

            is_final = bool(payload.get("is_final"))
            mode = str(payload.get("mode") or "")
            is_offline_result = "offline" in mode
            if text and (is_final or is_offline_result):
                self.final_text = text
                await send_event(self.client_ws, "asr_final", text=text, asr=payload)
            elif text:
                await send_event(self.client_ws, "asr_partial", text=text, asr=payload)

            if is_final or (text and is_offline_result):
                self.final_seen.set()
                return


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


def parse_chunk_size(value: str) -> list[int]:
    parts = value.split("-")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk size must look like 5-10-5")
    try:
        result = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk size must contain integers") from exc
    if any(part <= 0 for part in result):
        raise argparse.ArgumentTypeError("chunk size values must be positive")
    return result


def parse_asr_proxy(value: str) -> str | bool | None:
    normalized = value.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"0", "false", "no", "none", "off", "direct", "disabled"}:
        return None
    if lowered == "auto":
        return True
    return normalized


def parse_args(argv: list[str] | None = None) -> Settings:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.environ.get("VCHAT_ENV_FILE", ".env"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Streaming voice chat websocket server.")
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument("--host", default=env_str("VCHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("VCHAT_PORT", 8765))

    parser.add_argument("--asr-host", default=env_str("ASR_HOST", "127.0.0.1"))
    parser.add_argument("--asr-port", type=int, default=env_int("ASR_PORT", 10095))
    parser.add_argument("--asr-ssl", type=int, choices=(0, 1), default=int(env_bool("ASR_SSL", False)))
    parser.add_argument(
        "--asr-ssl-verify",
        type=int,
        choices=(0, 1),
        default=int(env_bool("ASR_SSL_VERIFY", False)),
    )
    parser.add_argument("--asr-proxy", default=env_str("ASR_PROXY", "none"))
    parser.add_argument("--asr-mode", default=env_str("ASR_MODE", "2pass"))
    parser.add_argument("--asr-sample-rate", type=int, default=env_int("ASR_SAMPLE_RATE", 16000))
    parser.add_argument(
        "--asr-chunk-size",
        type=parse_chunk_size,
        default=parse_chunk_size(env_str("ASR_CHUNK_SIZE", DEFAULT_ASR_CHUNK_SIZE)),
    )
    parser.add_argument("--asr-chunk-interval", type=int, default=env_int("ASR_CHUNK_INTERVAL", 10))
    parser.add_argument("--asr-final-timeout", type=float, default=env_float("ASR_FINAL_TIMEOUT", 20.0))
    parser.add_argument("--use-itn", type=int, choices=(0, 1), default=int(env_bool("ASR_USE_ITN", True)))
    parser.add_argument("--svs-itn", type=int, choices=(0, 1), default=int(env_bool("ASR_SVS_ITN", True)))

    parser.add_argument("--llm-base-url", default=env_str("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument(
        "--llm-api-key",
        default=env_str("LLM_API_KEY", env_str("MIMO_API_KEY", env_str("OPENAI_API_KEY", "EMPTY"))),
    )
    parser.add_argument("--llm-model", default=env_str("LLM_MODEL", "Qwen3.5-4B"))
    parser.add_argument("--llm-system-prompt", default=env_str("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--llm-max-tokens", type=int, default=env_int("LLM_MAX_TOKENS", 1024))
    parser.add_argument("--llm-temperature", type=float, default=env_float("LLM_TEMPERATURE", 0.0))
    parser.add_argument("--llm-top-p", type=float, default=env_float("LLM_TOP_P", 0.95))
    parser.add_argument(
        "--llm-thinking-disabled",
        type=int,
        choices=(0, 1),
        default=int(env_bool("LLM_THINKING_DISABLED", True)),
    )

    parser.add_argument("--tts-base-url", default=env_str("TTS_BASE_URL", "http://127.0.0.1:51010"))
    parser.add_argument("--tts-model", default=env_str("TTS_MODEL", "/workspace/model/Qwen3-TTS-12Hz-1.7B-Base"))
    parser.add_argument("--tts-voice", default=env_str("TTS_VOICE", "custom_voice_1"))
    parser.add_argument("--tts-sample-rate", type=int, default=env_int("TTS_SAMPLE_RATE", 24000))
    parser.add_argument("--tts-language", default=env_str("TTS_LANGUAGE", "Auto"))
    parser.add_argument("--tts-task-type", default=env_str("TTS_TASK_TYPE", "CustomVoice"))
    parser.add_argument("--tts-chunk-size", type=int, default=env_int("TTS_CHUNK_SIZE", 4096))
    parser.add_argument("--tts-min-chars", type=int, default=env_int("TTS_MIN_CHARS", 12))
    parser.add_argument("--tts-flush-chars", type=int, default=env_int("TTS_FLUSH_CHARS", 48))
    args = parser.parse_args(argv)

    if args.port <= 0:
        parser.error("--port must be positive")
    if args.asr_port <= 0:
        parser.error("--asr-port must be positive")
    if args.asr_sample_rate <= 0:
        parser.error("--asr-sample-rate must be positive")
    if args.asr_chunk_interval <= 0:
        parser.error("--asr-chunk-interval must be positive")
    if args.asr_final_timeout <= 0:
        parser.error("--asr-final-timeout must be positive")
    if args.llm_max_tokens <= 0:
        parser.error("--llm-max-tokens must be positive")
    if args.tts_sample_rate <= 0:
        parser.error("--tts-sample-rate must be positive")
    if args.tts_chunk_size <= 0:
        parser.error("--tts-chunk-size must be positive")
    if args.tts_min_chars <= 0:
        parser.error("--tts-min-chars must be positive")
    if args.tts_flush_chars < args.tts_min_chars:
        parser.error("--tts-flush-chars must be greater than or equal to --tts-min-chars")

    return Settings(
        host=args.host,
        port=args.port,
        asr_host=args.asr_host,
        asr_port=args.asr_port,
        asr_ssl=bool(args.asr_ssl),
        asr_ssl_verify=bool(args.asr_ssl_verify),
        asr_proxy=parse_asr_proxy(args.asr_proxy),
        asr_mode=args.asr_mode,
        asr_sample_rate=args.asr_sample_rate,
        asr_chunk_size=args.asr_chunk_size,
        asr_chunk_interval=args.asr_chunk_interval,
        asr_final_timeout=args.asr_final_timeout,
        use_itn=bool(args.use_itn),
        svs_itn=bool(args.svs_itn),
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_system_prompt=args.llm_system_prompt,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        llm_thinking_disabled=bool(args.llm_thinking_disabled),
        tts_base_url=args.tts_base_url,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_sample_rate=args.tts_sample_rate,
        tts_language=args.tts_language,
        tts_task_type=args.tts_task_type,
        tts_chunk_size=args.tts_chunk_size,
        tts_min_chars=args.tts_min_chars,
        tts_flush_chars=args.tts_flush_chars,
    )


def build_asr_ssl_context(settings: Settings) -> ssl.SSLContext | None:
    if not settings.asr_ssl:
        return None
    if settings.asr_ssl_verify:
        return ssl.create_default_context()
    context = ssl._create_unverified_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def send_event(ws: Any, event_type: str, **payload: Any) -> None:
    await ws.send(json_dumps({"type": event_type, **payload}))


async def stream_llm_reply(settings: Settings, user_text: str) -> AsyncIterator[str]:
    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    extra_body = {"thinking": {"type": "disabled"}} if settings.llm_thinking_disabled else None
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": settings.llm_system_prompt},
            {"role": "user", "content": user_text},
        ],
        max_completion_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        stream=True,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body=extra_body,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def stream_tts_audio(settings: Settings, text: str) -> AsyncIterator[bytes]:
    payload = {
        "model": settings.tts_model,
        "voice": settings.tts_voice,
        "input": text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "language": settings.tts_language,
        "task_type": settings.tts_task_type,
    }
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", settings.tts_url, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=settings.tts_chunk_size):
                if chunk:
                    yield chunk


async def speak_text(settings: Settings, ws: Any, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        return
    await send_event(ws, "tts_start", text=normalized)
    audio_bytes = 0
    started_at = time.perf_counter()
    async for chunk in stream_tts_audio(settings, normalized):
        audio_bytes += len(chunk)
        await ws.send(chunk)
    await send_event(
        ws,
        "tts_done",
        text=normalized,
        audio_bytes=audio_bytes,
        elapsed=round(time.perf_counter() - started_at, 3),
    )


def should_flush_tts(buffer: str, settings: Settings) -> bool:
    stripped = buffer.strip()
    if len(stripped) >= settings.tts_flush_chars:
        return True
    if len(stripped) < settings.tts_min_chars:
        return False
    return stripped[-1] in TEXT_FLUSH_PUNCTUATION


async def answer_turn(settings: Settings, ws: Any, user_text: str) -> None:
    await send_event(ws, "llm_start")
    full_reply: list[str] = []
    tts_buffer = ""

    async for delta in stream_llm_reply(settings, user_text):
        full_reply.append(delta)
        tts_buffer += delta
        await send_event(ws, "llm_delta", text=delta)
        if should_flush_tts(tts_buffer, settings):
            await speak_text(settings, ws, tts_buffer)
            tts_buffer = ""

    if tts_buffer.strip():
        await speak_text(settings, ws, tts_buffer)

    reply = "".join(full_reply).strip()
    await send_event(ws, "llm_done", text=reply)


async def handle_client(ws: Any, settings: Settings) -> None:
    await send_event(
        ws,
        "ready",
        asr_sample_rate=settings.asr_sample_rate,
        tts_sample_rate=settings.tts_sample_rate,
        channels=1,
        sample_width=2,
    )

    asr_session: AsrSession | None = None
    turn_id = 0
    try:
        async for message in ws:
            if isinstance(message, bytes):
                if not asr_session:
                    await send_event(ws, "error", message="binary audio received before audio_start")
                    continue
                await asr_session.send_audio(message)
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await send_event(ws, "error", message="invalid json control message")
                continue

            message_type = payload.get("type")
            if message_type == "audio_start":
                if asr_session:
                    await send_event(ws, "error", message="audio_start received while a turn is active")
                    continue
                turn_id += 1
                sample_rate = int(payload.get("sample_rate") or settings.asr_sample_rate)
                if sample_rate != settings.asr_sample_rate:
                    await send_event(
                        ws,
                        "warning",
                        message=(
                            f"client sample_rate={sample_rate}, "
                            f"ASR expects {settings.asr_sample_rate}"
                        ),
                    )
                asr_session = AsrSession(settings, turn_id, ws)
                try:
                    await asr_session.__aenter__()
                except Exception as exc:
                    asr_session = None
                    message = f"ASR connection failed: {settings.asr_uri} ({exc})"
                    print(message, flush=True)
                    await send_event(ws, "error", message=message)
                    continue
                await send_event(ws, "turn_started", turn_id=turn_id)
            elif message_type == "audio_end":
                if not asr_session:
                    await send_event(ws, "error", message="audio_end received without audio_start")
                    continue
                current_session = asr_session
                asr_session = None
                user_text = await current_session.finish()
                await current_session.__aexit__(None, None, None)
                if not user_text:
                    await send_event(ws, "turn_done", turn_id=turn_id, text="", reply="")
                    continue
                if user_text != current_session.final_text:
                    await send_event(ws, "asr_final", text=user_text)
                await answer_turn(settings, ws, user_text)
                await send_event(ws, "turn_done", turn_id=turn_id, text=user_text)
            elif message_type == "ping":
                await send_event(ws, "pong")
            else:
                await send_event(ws, "error", message=f"unsupported message type: {message_type}")
    finally:
        if asr_session:
            await asr_session.__aexit__(None, None, None)


async def async_main(argv: list[str] | None = None) -> None:
    settings = parse_args(argv)
    print(
        f"vchat listening on ws://{settings.host}:{settings.port} "
        f"(ASR {settings.asr_uri}, ASR proxy {settings.asr_proxy_label}, "
        f"LLM {settings.llm_base_url}, TTS {settings.tts_url})",
        flush=True,
    )

    async def handler(ws: Any) -> None:
        await handle_client(ws, settings)

    async with websockets.serve(handler, settings.host, settings.port, max_size=None):
        await asyncio.Future()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("interrupted")


if __name__ == "__main__":
    main()
