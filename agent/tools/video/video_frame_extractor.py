"""
Video frame extraction tool.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class VideoFrameExtractorTool(Tool):
    """Extract frames from video files for analysis."""

    name = "extract_video_frames"
    description = "Extract frames from a video file at specified intervals or timestamps."

    async def execute(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_count: int = 10,
        interval: Optional[float] = None,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> ToolResult:
        """Extract frames from video."""
        try:
            path = Path(video_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Video not found: {video_path}")

            if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}:
                return ToolResult(success=False, error=f"Unsupported video format: {path.suffix}")

            # Set output directory
            if output_dir is None:
                output_dir = str(path.parent / f"{path.stem}_frames")
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Extract frames
            extracted_frames = await self._extract_frames(
                path,
                output_path,
                frame_count,
                interval,
                start_time,
                duration
            )

            return ToolResult(
                success=True,
                data={
                    "video_path": video_path,
                    "output_dir": str(output_path),
                    "extracted_frames": extracted_frames,
                    "frame_count": len(extracted_frames)
                }
            )

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return ToolResult(success=False, error=str(e))

    async def _extract_frames(
        self,
        video_path: Path,
        output_path: Path,
        frame_count: int,
        interval: Optional[float],
        start_time: float,
        duration: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Extract frames using OpenCV."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Calculate duration
        if duration is None:
            duration = video_duration - start_time

        # Calculate frame indices
        if interval is not None:
            # Extract at interval
            frames_to_extract = []
            current_time = start_time
            while current_time < start_time + duration:
                frame_idx = int(current_time * fps)
                if frame_idx < total_frames:
                    frames_to_extract.append((frame_idx, current_time))
                current_time += interval
        else:
            # Extract evenly spaced frames
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            frames_to_extract = []
            for i in range(frame_count):
                frame_idx = start_frame + int((end_frame - start_frame) * i / frame_count)
                timestamp = frame_idx / fps
                frames_to_extract.append((frame_idx, timestamp))

        # Extract frames
        extracted = []
        for i, (frame_idx, timestamp) in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                output_file = output_path / f"frame_{i:04d}_t{timestamp:.2f}s.jpg"
                cv2.imwrite(str(output_file), frame)
                extracted.append({
                    "frame_number": i,
                    "timestamp": timestamp,
                    "file_path": str(output_file)
                })

        cap.release()
        return extracted

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to the video file"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save extracted frames (default: video_frames)"
                },
                "frame_count": {
                    "type": "integer",
                    "description": "Number of frames to extract (default: 10)",
                    "default": 10
                },
                "interval": {
                    "type": "number",
                    "description": "Interval in seconds between frames (overrides frame_count)"
                },
                "start_time": {
                    "type": "number",
                    "description": "Start time in seconds (default: 0.0)",
                    "default": 0.0
                },
                "duration": {
                    "type": "number",
                    "description": "Duration to process in seconds (default: entire video)"
                }
            },
            "required": ["video_path"]
        }
