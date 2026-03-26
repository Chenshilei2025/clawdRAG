"""
Event definitions for the message bus.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class MessageType(Enum):
    """Types of messages."""
    USER_QUERY = "user_query"
    AGENT_RESPONSE = "agent_response"
    SUBAGENT_REQUEST = "subagent_request"
    SUBAGENT_RESPONSE = "subagent_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    STATUS = "status"


@dataclass
class InboundMessage:
    """Message coming into the system from user."""
    content: str
    attachments: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """Message going out from the system to user."""
    content: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_complete: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallMessage:
    """Message representing a tool call request."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str
    subagent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolResultMessage:
    """Message representing a tool call result."""
    call_id: str
    result: Any
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SubagentRequest:
    """Request to spawn a subagent."""
    task_type: str
    parameters: Dict[str, Any]
    parent_agent_id: str
    request_id: str


@dataclass
class SubagentResponse:
    """Response from a subagent."""
    request_id: str
    result: Any
    status: str  # "success", "error", "partial"
    subagent_id: str
