"""
Message tool for agent communication.
"""
from typing import Dict, Any
from datetime import datetime

from ..base import Tool, ToolResult


class MessageTool(Tool):
    """Tool for sending messages to the user."""

    name = "send_message"
    description = "Send a message to the user. Use this to communicate important information or ask questions."

    async def execute(self, message: str, message_type: str = "info") -> ToolResult:
        """Send a message to the user."""
        return ToolResult(
            success=True,
            data={
                "type": "message",
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send"
                },
                "message_type": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "success"],
                    "description": "Type of message",
                    "default": "info"
                }
            },
            "required": ["message"]
        }
