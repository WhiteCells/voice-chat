# voice-chat workspace

这个仓库现在是单一根项目加多个源码模块的结构，不再让每个子目录各自带环境、配置和项目元数据。
所有第三方依赖统一声明在根目录 `pyproject.toml`，`asr/`、`llm/`、`tts/`、`vchat/`、`vclient/` 里只保留 Python 源码文件。

## 模块

- `asr`: FunASR WebSocket 客户端
- `llm`: OpenAI-compatible LLM 冒烟测试
- `tts`: 流式 TTS 播放测试
- `vchat`: 语音对话 WebSocket 服务
- `vclient`: 本地麦克风/扬声器测试客户端

## 统一初始化

在仓库根目录执行一次：

```bash
uv sync
cp .env.example .env
```

之后所有模块都从根目录启动，不再需要 `cd` 到单独目录里建环境。

## 启动模块

```bash
uv run asr --server-ip 127.0.0.1 --port 10095 --is-ssl 0 --wav-path test.wav
uv run llm
uv run tts
uv run vchat
uv run vclient
```

也可以继续给每个模块传自己的命令行参数：

```bash
uv run vchat --host 0.0.0.0 --port 8765
uv run vclient --uri ws://127.0.0.1:8765
uv run tts "下午三点提醒我检查邮件。" --voice custom_voice_1
```

## 配置

默认读取仓库根目录的 `.env`。根 `.env.example` 已经汇总了各模块常用配置项。

如果你只想给某个模块使用单独配置文件，也可以显式指定：

```bash
uv run vchat --env-file .env.example
```
