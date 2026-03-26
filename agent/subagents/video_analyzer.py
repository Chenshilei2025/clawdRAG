"""
Video analyzer subagent for processing video files.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .base import SimpleSubagent

logger = logging.getLogger(__name__)


class VideoAnalyzerSubagent(SimpleSubagent):
    """Subagent specialized in analyzing video files."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(subagent_id, config)
        self.task_description = """You are a video analysis specialist. Your task is to analyze video content and provide insights about:

1. Visual content (scenes, actions, objects, people)
2. Audio content (speech, music, sound effects)
3. Narrative structure and flow
4. Key moments and timestamps
5. Overall themes and messages"""

        self.tool_registry = None

    async def initialize(self):
        """Initialize with video processing tools."""
        await super().initialize()

        from ..tools.base import ToolRegistry
        from ..tools.video.video_frame_extractor import VideoFrameExtractorTool
        from ..tools.video.audio_transcriber import AudioTranscriberTool
        from ..tools.multimodal.image_captioning import ImageCaptioningTool
        from ..tools.multimodal.vector_search import VectorIndexTool

        self.tool_registry = ToolRegistry()

        tool_config = self.config.get("tool_config", {})
        self.tool_registry.register(VideoFrameExtractorTool(tool_config))
        self.tool_registry.register(AudioTranscriberTool(tool_config))
        self.tool_registry.register(ImageCaptioningTool(tool_config))
        self.tool_registry.register(VectorIndexTool(tool_config))

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a video file."""
        file_path = parameters.get("file_path", "")
        frame_count = parameters.get("frame_count", 10)
        transcribe_audio = parameters.get("transcribe_audio", True)
        index_content = parameters.get("index_content", True)

        path = Path(file_path)
        if not path.exists():
            return {
                "error": f"Video not found: {file_path}",
                "subagent_id": self.subagent_id
            }

        # Extract frames
        frames_result = await self.tool_registry.execute(
            "extract_video_frames",
            video_path=str(path),
            frame_count=frame_count
        )

        if not frames_result.success:
            return {
                "error": f"Frame extraction failed: {frames_result.error}",
                "subagent_id": self.subagent_id
            }

        extracted_frames = frames_result.data.get("extracted_frames", [])

        # Analyze frames
        frame_descriptions = []
        for frame_info in extracted_frames:
            caption_result = await self.tool_registry.execute(
                "caption_image",
                image_path=frame_info["file_path"]
            )
            if caption_result.success:
                frame_descriptions.append({
                    "timestamp": frame_info["timestamp"],
                    "frame_number": frame_info["frame_number"],
                    "description": caption_result.data.get("caption", "")
                })

        # Transcribe audio if requested
        transcript = None
        if transcribe_audio:
            audio_result = await self.tool_registry.execute(
                "transcribe_audio",
                file_path=str(path)
            )
            if audio_result.success:
                transcript = audio_result.data

        # Analyze with LLM
        analysis = await self._analyze_with_llm(
            frame_descriptions,
            transcript,
            parameters
        )

        # Index for search if requested
        indexed = False
        if index_content:
            content_text = self._build_index_content(frame_descriptions, transcript)
            if content_text:
                index_result = await self.tool_registry.execute(
                    "index_document",
                    content=content_text,
                    metadata={
                        "source": str(path),
                        "type": "video",
                        "subagent_id": self.subagent_id,
                        "duration": parameters.get("duration", 0)
                    }
                )
                indexed = index_result.success

        return {
            "file_path": str(path),
            "frames_extracted": len(extracted_frames),
            "frame_analysis": frame_descriptions,
            "transcript": transcript,
            "analysis": analysis,
            "indexed": indexed,
            "subagent_id": self.subagent_id
        }

    async def _analyze_with_llm(
        self,
        frame_descriptions: list,
        transcript: Optional[Dict],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze video content with LLM."""
        prompt_parts = ["Video Analysis Request\n"]

        # Frame descriptions
        if frame_descriptions:
            frames_text = "\n".join([
                f"[{fd['timestamp']:.2f}s]: {fd['description']}"
                for fd in frame_descriptions[:10]
            ])
            prompt_parts.append(f"Visual Content:\n{frames_text}\n")

        # Transcript
        if transcript and transcript.get("text"):
            prompt_parts.append(f"Audio/Transcript:\n{transcript['text'][:2000]}\n")

        prompt_parts.append("""Please provide:
1. Content Summary
2. Key Scenes/Events (with timestamps)
3. Main Themes
4. Notable Visual Elements
5. Audio/Speech Summary

Respond in JSON format.""")

        prompt = "\n".join(prompt_parts)

        from ...providers.base import Message
        messages = [
            Message(role="system", content=self.task_description),
            Message(role="user", content=prompt)
        ]

        response = await self.llm_provider.generate(
            messages,
            temperature=0.5,
            max_tokens=2000
        )

        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "summary": "Analysis completed",
                "raw_response": response.content
            }

    def _build_index_content(self, frame_descriptions: list, transcript: Optional[Dict]) -> str:
        """Build text content for indexing."""
        parts = []

        if frame_descriptions:
            parts.append("Visual Content:")
            parts.extend([
                f"At {fd['timestamp']:.2f}s: {fd['description']}"
                for fd in frame_descriptions
            ])

        if transcript and transcript.get("text"):
            parts.append(f"\nTranscript:\n{transcript['text']}")

        return "\n".join(parts)
