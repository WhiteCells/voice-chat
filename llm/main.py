import argparse
import os
import time
from pathlib import Path

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "You are AI assistant"
DEFAULT_PROMPT = "你好，介绍一下你自己"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.environ.get("LLM_ENV_FILE", ".env"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="OpenAI-compatible LLM smoke test.")
    parser.add_argument("--env-file", default=pre_args.env_file)
    parser.add_argument("--base-url", default=env_str("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument(
        "--api-key",
        default=env_str("LLM_API_KEY", env_str("MIMO_API_KEY", env_str("OPENAI_API_KEY", "EMPTY"))),
    )
    parser.add_argument("--model", default=env_str("LLM_MODEL", "Qwen3.5-4B"))
    parser.add_argument("--system-prompt", default=env_str("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--prompt", default=env_str("LLM_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--max-tokens", type=int, default=env_int("LLM_MAX_TOKENS", 1024))
    parser.add_argument("--temperature", type=float, default=env_float("LLM_TEMPERATURE", 0.0))
    parser.add_argument("--top-p", type=float, default=env_float("LLM_TOP_P", 0.95))
    parser.add_argument(
        "--thinking-disabled",
        type=int,
        choices=(0, 1),
        default=int(env_bool("LLM_THINKING_DISABLED", True)),
    )
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    extra_body = {"thinking": {"type": "disabled"}} if args.thinking_disabled else None

    started_at = time.time()
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        max_completion_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=False,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body=extra_body,
    )
    ended_at = time.time()

    print(completion.model_dump_json())
    print("cast time:", ended_at - started_at)


if __name__ == "__main__":
    main()
