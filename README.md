# J.A.R.V.I.S.-Style Local Voice Assistant

This project is a minimal local voice assistant built in Python. It listens for a wake word, records your command, sends the prompt to an LLM, optionally executes local tools, and speaks a response back to you.

## Features

- Wake word detection using microphone input
- Speech-to-text using `SpeechRecognition` with optional `faster-whisper`
- LLM integration for OpenAI or a local Ollama server
- Tool execution through a registry with built-in examples
- Text-to-speech using `pyttsx3`
- Graceful shutdown and helpful debugging logs

## Project files

- `jarvis.py` – main application logic
- `requirements.txt` – dependency list

## Installation

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

   Windows:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure your microphone is available and configured.

## Runtime configuration

Set one of the following model backends before running the assistant.

### Option A: Ollama (local)

1. Install and run Ollama: https://ollama.com/
2. Pull a model:

   ```bash
   ollama pull llama3.1
   ```

3. Start the assistant:

   ```bash
   $env:LLM_PROVIDER="ollama"
   $env:LLM_MODEL="llama3.1"
   $env:OLLAMA_BASE_URL="http://localhost:11434"
   python jarvis.py
   ```

### Option B: OpenAI

```bash
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="your-key-here"
python jarvis.py
```

## Usage

- Say the wake word to begin. The default is `jarvis`.
- After the wake word, ask a question or give a command.
- The assistant can execute local tools such as:
  - system stats
  - opening an application
  - getting the current date/time

## Custom tool example

You can add more tools by editing the `ToolRegistry` section in `jarvis.py` and registering a function with a short description and JSON schema.

## Troubleshooting

- If microphone access fails, check OS permissions and make sure the default input device is active.
- If the assistant cannot hear you, increase the microphone gain or lower the ambient noise.
- If the LLM request fails, confirm your API key or local Ollama server is running.
- If `pyttsx3` fails, ensure the OS voice engine is installed and available.
