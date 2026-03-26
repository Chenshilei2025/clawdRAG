"""
Audio transcription tool for video processing.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class AudioTranscriberTool(Tool):
    """Transcribe audio from video files or standalone audio files."""

    name = "transcribe_audio"
    description = "Extract and transcribe audio from video or audio files using speech-to-text."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.stt_provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize speech-to-text provider."""
        provider_type = self.config.get("provider", "openai")
        provider_config = self.config.get("provider_config", {})

        if provider_type == "openai":
            # OpenAI Whisper will be used via API
            self.stt_config = provider_config
        elif provider_type == "local":
            # Local Whisper
            try:
                import whisper
                model_size = self.config.get("model_size", "base")
                self.stt_provider = whisper.load_model(model_size)
            except ImportError:
                logger.warning("whisper not installed, install with: pip install openai-whisper")

    async def execute(
        self,
        file_path: str,
        language: Optional[str] = None,
        timestamp_segments: bool = True
    ) -> ToolResult:
        """Transcribe audio from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            # Check if it's a video file - need to extract audio first
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                audio_path = await self._extract_audio(path)
                if not audio_path:
                    return ToolResult(success=False, error="Failed to extract audio from video")
            else:
                audio_path = str(path)

            # Transcribe
            if self.stt_provider:
                result = await self._transcribe_local(audio_path, language, timestamp_segments)
            else:
                result = await self._transcribe_openai(audio_path, language, timestamp_segments)

            return ToolResult(
                success=True,
                data=result
            )

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ToolResult(success=False, error=str(e))

    async def _extract_audio(self, video_path: Path) -> Optional[str]:
        """Extract audio from video file."""
        try:
            import subprocess
            audio_path = video_path.parent / f"{video_path.stem}.wav"

            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
                "-y"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and audio_path.exists():
                return str(audio_path)

            return None

        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None

    async def _transcribe_local(
        self,
        audio_path: str,
        language: Optional[str],
        timestamp_segments: bool
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper."""
        import torch

        result = self.stt_provider.transcribe(
            audio_path,
            language=language,
            word_timestamps=timestamp_segments
        )

        segments_data = []
        if timestamp_segments and result.get("segments"):
            for seg in result["segments"]:
                segments_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                })

        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "duration": result.get("segments", [])[-1]["end"] if result.get("segments") else None,
            "segments": segments_data if timestamp_segments else []
        }

    async def _transcribe_openai(
        self,
        audio_path: str,
        language: Optional[str],
        timestamp_segments: bool
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required")

        client = AsyncOpenAI(api_key=self.stt_config.get("api_key"))

        with open(audio_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json" if timestamp_segments else "text"
            )

        if isinstance(transcript, str):
            return {
                "text": transcript,
                "segments": []
            }

        segments_data = []
        if timestamp_segments and transcript.get("segments"):
            for seg in transcript["segments"]:
                segments_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                })

        return {
            "text": transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": segments_data
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the audio or video file"
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., 'en', 'zh'). Auto-detect if not specified."
                },
                "timestamp_segments": {
                    "type": "boolean",
                    "description": "Return segments with timestamps",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
