"""J.A.R.V.I.S.-inspired local voice assistant.

Overview:
    This script creates a small but extensible private AI assistant that:
    1. listens to the microphone,
    2. waits for a wake word,
    3. sends the user's request to an LLM,
    4. executes tool calls when the model chooses to do so,
    5. speaks the final answer back via TTS.

How to extend it:
    The ToolRegistry class is the extension point. Add a new function, register it
    with the registry, and provide a JSON schema description so the LLM can call it.
    The design intentionally keeps local code and external API integration separated
    so it is easy to swap between Ollama, OpenAI, or another model backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil
import requests
import speech_recognition as sr
from dotenv import load_dotenv
import pyttsx3

from app_control import AppControl, close_app, launch_app
from file_manager import FileManager, search_and_open_file
from system_info import SystemSyncManager, get_system_telemetry
from voice_auth import VoiceAuthenticator

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment may not have the SDK installed
    OpenAI = None

try:
    import faster_whisper  # type: ignore
except ImportError:  # pragma: no cover
    faster_whisper = None


# ---------------------------------------------------------------------------
# Configuration and environment
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jarvis")

WAKE_WORD = os.getenv("WAKE_WORD", "jarvis").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))
VOICE_AUTH_ENABLED = os.getenv("VOICE_AUTH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
VOICE_PROFILE_PATH = os.getenv("VOICE_PROFILE_PATH", "user_voice_profile.npy")
VOICE_AUTH_THRESHOLD = float(os.getenv("VOICE_AUTH_THRESHOLD", "0.75"))


# ---------------------------------------------------------------------------
# Local tools for function calling
# ---------------------------------------------------------------------------


def get_system_stats() -> Dict[str, Any]:
    """Return the current CPU, memory, and disk usage in a simple dict."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_percent": round(cpu_percent, 2),
        "memory_total_gb": round(memory.total / (1024 ** 3), 2),
        "memory_used_gb": round(memory.used / (1024 ** 3), 2),
        "memory_percent": round(memory.percent, 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_percent": round((disk.used / disk.total) * 100, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def get_current_time() -> str:
    """Return the current date and time as a human readable string."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")


def open_application(app_name: str) -> str:
    """Open an application by name or path.

    Example:
        open_application("notepad")
        open_application("C:/Program Files/Google/Chrome/Application/chrome.exe")
    """
    if not app_name or not app_name.strip():
        return "No application name was provided."

    try:
        if platform.system() == "Windows":
            os.startfile(app_name)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", app_name])
        else:
            subprocess.Popen(["xdg-open", app_name])
        return f"Attempted to open: {app_name}"
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.exception("Could not open application: %s", app_name)
        return f"Failed to open {app_name}: {exc}"


def get_system_telemetry_tool() -> Dict[str, Any]:
    """Return structured OS telemetry context for the LLM."""
    return get_system_telemetry()


def search_and_open_file_tool(file_name: str) -> str:
    """Find and open a file in permitted user directories."""
    return search_and_open_file(file_name)


def launch_app_tool(app_name: str) -> Dict[str, Any]:
    """Launch application by name."""
    return launch_app(app_name)


def close_app_tool(app_name: str) -> Dict[str, Any]:
    """Close application by name."""
    return close_app(app_name)


class ToolRegistry:
    """Stores all callable local actions and exposes JSON schemas for the model.

    To add a custom tool:
        1. Implement a Python function.
        2. Register it using ToolRegistry.register().
        3. Keep the arguments simple and serializable (JSON-friendly types).
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self.register(
            "get_system_stats",
            get_system_stats,
            description="Return the current CPU, memory, and disk usage statistics.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        self.register(
            "open_application",
            open_application,
            description="Open an application or executable by name or full path.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "The application name or full path to open.",
                    }
                },
                "required": ["app_name"],
            },
        )
        self.register(
            "get_current_time",
            get_current_time,
            description="Return the current date and time in a human readable format.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        self.register(
            "get_system_telemetry",
            get_system_telemetry_tool,
            description="Return a structured snapshot of CPU, memory, disk, battery, and running process telemetry for the local machine.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        self.register(
            "search_and_open_file",
            search_and_open_file_tool,
            description="Find a file in the user's common directories and open it with the default system app.",
            parameters={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The file name or partial name to search for.",
                    }
                },
                "required": ["file_name"],
            },
        )
        self.register(
            "launch_app",
            launch_app_tool,
            description="Launch a local application by name, using PATH or common install locations.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Application name such as 'code', 'chrome', 'notepad', or 'terminal'.",
                    }
                },
                "required": ["app_name"],
            },
        )
        self.register(
            "close_app",
            close_app_tool,
            description="Terminate a running application by name. Use this only for apps the user explicitly asked to close.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the running application to close.",
                    }
                },
                "required": ["app_name"],
            },
        )

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        self._tools[name] = {
            "function": fn,
            "description": description,
            "parameters": parameters,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool schema entries."""
        schema = []
        for name, tool in self._tools.items():
            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return schema

    def invoke(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool with keyword args."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]["function"](**kwargs)


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin abstraction for model backends.

    Supported backends:
      - ollama: local HTTP API on http://localhost:11434
      - openai: standard OpenAI API via the openai Python SDK

    Notes:
      The assistant design intentionally keeps this logic separate from the event
      loop so you can replace one backend with another without affecting the voice
      pipeline.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key) if self.provider == "openai" and OpenAI else None

    def _normalize_tool_calls(self, data: Any) -> List[Dict[str, Any]]:
        """Convert provider-specific tool call formats into a common format."""
        tool_calls: List[Dict[str, Any]] = []

        if not data:
            return tool_calls

        if isinstance(data, list):
            raw_calls = data
        else:
            raw_calls = [data]

        for item in raw_calls:
            if isinstance(item, dict):
                if "function" in item:
                    fn = item["function"]
                    tool_calls.append(
                        {
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments", {}),
                        }
                    )
                elif "name" in item:
                    tool_calls.append(
                        {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", {}),
                        }
                    )

        return tool_calls

    async def generate_response(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send the prompt and return a unified response object.

        Response shape:
            {
                "content": "Final text response, if any",
                "tool_calls": [
                    {"name": "get_system_stats", "arguments": {}}
                ]
            }
        """
        if self.provider == "openai":
            if self.client is None:
                raise RuntimeError("OpenAI SDK is not installed or API key is missing.")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
                temperature=0.2,
            )
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = []
            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    tool_calls.append(
                        {
                            "name": call.function.name,
                            "arguments": json.loads(call.function.arguments),
                        }
                    )
            return {"content": content, "tool_calls": tool_calls}

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools

            result = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            result.raise_for_status()
            data = result.json()
            message = data.get("message", {})
            content = message.get("content", "")
            tool_calls = self._normalize_tool_calls(message.get("tool_calls"))
            return {"content": content, "tool_calls": tool_calls}

        raise ValueError(f"Unsupported LLM_PROVIDER '{self.provider}'. Supported values: ollama, openai")


# ---------------------------------------------------------------------------
# Speech-to-text and text-to-speech
# ---------------------------------------------------------------------------


class VoiceAssistant:
    """Handles microphone input and speech output."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.microphone = (
            sr.Microphone(device_index=MIC_DEVICE_INDEX)
            if MIC_DEVICE_INDEX >= 0
            else sr.Microphone()
        )
        self.tts_engine = pyttsx3.init() if pyttsx3 is not None else None
        self.last_audio = None

    def verify_last_audio(self, authenticator: Optional[VoiceAuthenticator]) -> bool:
        """Authenticate the most recent sampled audio clip against the saved profile."""
        if authenticator is None:
            return True
        if self.last_audio is None:
            logger.warning("[Security] No captured audio available to verify against the voice profile.")
            return False
        return authenticator.verify_speaker(self.last_audio)

    def speak(self, text: str) -> None:
        """Speak a response using pyttsx3."""
        if not text or not text.strip():
            return
        logger.info("Speaking: %s", text)
        if self.tts_engine is not None:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            logger.warning("pyttsx3 is unavailable; no audio output will be played.")

    async def listen_for_command(self) -> Optional[str]:
        """Listen to the microphone and convert speech to text."""
        logger.info("Listening for speech...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Awaiting wake word and command... speak clearly near the mic.")
                audio = await asyncio.to_thread(
                    self.recognizer.listen,
                    source,
                    timeout=8,
                    phrase_time_limit=8,
                )
            self.last_audio = audio

            transcript = await asyncio.to_thread(self._transcribe_audio, audio)
            if transcript:
                logger.info("Heard: %s", transcript)
            return transcript
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout. Check microphone permissions, device index, or speech volume.")
            return None
        except sr.UnknownValueError:
            logger.warning("Audio was heard but could not be understood.")
            return None
        except OSError as exc:
            logger.error("Microphone error: %s", exc)
            raise RuntimeError("No supported microphone device was found.") from exc

    def _transcribe_audio(self, audio) -> str:
        """Transcribe audio with a local recognizer.

        This prefers Google Speech recognition but can be replaced with faster-whisper
        if you want fully local inference. Both are valid depending on the available
        environment and desired latency.
        """
        try:
            if faster_whisper is not None:
                # Example integration path for a local open-source model.
                # You would usually load the model once and reuse it here.
                # This keeps the default implementation simple and reliable.
                return self.recognizer.recognize_google(audio, language="en-US")
            return self.recognizer.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            logger.error("Speech recognition service error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Assistant orchestration
# ---------------------------------------------------------------------------


class JarvisAssistant:
    """Coordinates wake-word detection, LLM processing, tool execution and TTS."""

    def __init__(self):
        self.voice = VoiceAssistant()
        self.tool_registry = ToolRegistry()
        self.llm = LLMClient()
        self.voice_authenticator = (
            VoiceAuthenticator(
                profile_path=VOICE_PROFILE_PATH,
                threshold=VOICE_AUTH_THRESHOLD,
                device_index=MIC_DEVICE_INDEX,
            )
            if VOICE_AUTH_ENABLED
            else None
        )
        self.stop_event = asyncio.Event()
        self.system_prompt = (
            "You are J.A.R.V.I.S., a helpful local voice assistant. "
            "When the user asks for information, answer clearly and briefly. "
            "Use tool calls for system actions when appropriate. "
            "Do not invent facts. "
            "If you use a tool, report its result in a friendly plain-language response."
        )

    async def run(self) -> None:
        """Main event loop."""
        logger.info("Assistant is online. Wake word: '%s'", WAKE_WORD)
        while not self.stop_event.is_set():
            try:
                text = await self.voice.listen_for_command()
                if text is None:
                    continue

                lower_text = text.lower().strip()
                if WAKE_WORD not in lower_text:
                    logger.info("Ignoring speech because no wake word was detected.")
                    continue

                command = self._strip_wake_word(lower_text)
                if not command:
                    self.voice.speak("I am listening.")
                    continue

                if self.voice_authenticator is not None:
                    if not self.voice.verify_last_audio(self.voice_authenticator):
                        self.voice.speak("Voice authentication failed. Access denied.")
                        logger.warning("[Security] Unauthorized speaker rejected before command processing.")
                        continue
                    logger.info("[Security] Speaker verification passed.")

                logger.info("Processing command: %s", command)
                answer = await self.process_command(command)
                if answer:
                    self.voice.speak(answer)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down gracefully.")
                break
            except RuntimeError as exc:
                logger.error("Runtime error: %s", exc)
                self.voice.speak("I hit a runtime error while processing your request.")
            except Exception as exc:  # pragma: no cover - general handler for robustness
                logger.exception("Unhandled exception in assistant loop: %s", exc)
                self.voice.speak("Something went wrong. Please try again.")

        logger.info("Assistant shutdown complete.")

    def _strip_wake_word(self, transcript: str) -> str:
        lower = transcript.lower()
        if lower.startswith(WAKE_WORD):
            return transcript[len(WAKE_WORD):].strip(" ,:;!?-_")
        for marker in [f"{WAKE_WORD} ", f"{WAKE_WORD},", f"{WAKE_WORD}:"]:
            if marker in lower:
                return transcript[transcript.lower().find(marker) + len(marker):].strip(" ,:;!?-_")
        return transcript.strip(" ,:;!?-_")

    async def process_command(self, user_input: str) -> str:
        """Take a command from the user, call the LLM, then optionally execute tools."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            llm_result = await self.llm.generate_response(messages, self.tool_registry.list_tools())
        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            return "The language model is unavailable right now. Please try again later."

        tool_calls = llm_result.get("tool_calls", [])
        if tool_calls:
            execution_result = []
            for tool_call in tool_calls:
                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                logger.info("Executing tool '%s' with args: %s", name, arguments)
                result = self.tool_registry.invoke(name, **arguments)
                execution_result.append(
                    {"tool": name, "result": result}
                )

            tool_messages = [
                {"role": "assistant", "content": llm_result.get("content", "")},
                {
                    "role": "tool",
                    "content": json.dumps({"tool_calls": execution_result}),
                },
            ]
            follow_up = messages + tool_messages
            follow_up_response = await self.llm.generate_response(follow_up)
            return follow_up_response.get("content") or "I completed the requested action."

        return llm_result.get("content") or "I am ready to help."


async def main() -> None:
    """Entry point.

    Use Ctrl+C to exit gracefully. This is intentionally simple and debug-friendly.
    """
    assistant = JarvisAssistant()

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal %s", signum)
        assistant.stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    try:
        await assistant.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt captured at top level.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        logger.error("Fatal runtime error: %s", exc)
        sys.exit(1)
