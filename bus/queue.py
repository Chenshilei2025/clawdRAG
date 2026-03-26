"""
Message bus implementation for inter-agent communication.
"""
import asyncio
from typing import Optional, Callable, Dict, Any
from collections import deque
import logging

from .events import (
    InboundMessage,
    OutboundMessage,
    ToolCallMessage,
    ToolResultMessage,
    SubagentRequest,
    SubagentResponse
)

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Async message bus for agent communication.
    Handles routing of messages between agents, tools, and external systems.
    """

    def __init__(self):
        self._inbound_queue: asyncio.Queue = asyncio.Queue()
        self._outbound_queue: asyncio.Queue = asyncio.Queue()
        self._tool_call_queue: asyncio.Queue = asyncio.Queue()
        self._tool_result_queue: asyncio.Queue = asyncio.Queue()
        self._subagent_requests: asyncio.Queue = asyncio.Queue()

        self._subscribers: Dict[str, list[Callable]] = {
            "inbound": [],
            "outbound": [],
            "tool_call": [],
            "tool_result": [],
            "subagent": []
        }

        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the message bus worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_messages())
        logger.info("MessageBus started")

    async def stop(self):
        """Stop the message bus worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("MessageBus stopped")

    async def send_inbound(self, message: InboundMessage):
        """Send an inbound message to the queue."""
        await self._inbound_queue.put(message)
        await self._notify_subscribers("inbound", message)

    async def send_outbound(self, message: OutboundMessage):
        """Send an outbound message to the queue."""
        await self._outbound_queue.put(message)
        await self._notify_subscribers("outbound", message)

    async def send_tool_call(self, message: ToolCallMessage):
        """Send a tool call request."""
        await self._tool_call_queue.put(message)
        await self._notify_subscribers("tool_call", message)

    async def send_tool_result(self, message: ToolResultMessage):
        """Send a tool call result."""
        await self._tool_result_queue.put(message)
        await self._notify_subscribers("tool_result", message)

    async def request_subagent(self, request: SubagentRequest):
        """Request a subagent to handle a task."""
        await self._subagent_requests.put(request)
        await self._notify_subscribers("subagent", request)

    async def receive_inbound(self) -> InboundMessage:
        """Receive an inbound message (blocking)."""
        return await self._inbound_queue.get()

    async def receive_outbound(self) -> OutboundMessage:
        """Receive an outbound message (blocking)."""
        return await self._outbound_queue.get()

    async def receive_tool_call(self) -> ToolCallMessage:
        """Receive a tool call request (blocking)."""
        return await self._tool_call_queue.get()

    async def receive_tool_result(self) -> ToolResultMessage:
        """Receive a tool result (blocking)."""
        return await self._tool_result_queue.get()

    async def receive_subagent_request(self) -> SubagentRequest:
        """Receive a subagent request (blocking)."""
        return await self._subagent_requests.get()

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def _notify_subscribers(self, event_type: str, message: Any):
        """Notify all subscribers of an event."""
        for callback in self._subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

    async def _process_messages(self):
        """Background worker for processing messages."""
        while self._running:
            await asyncio.sleep(0.01)


class StreamingMessageBus(MessageBus):
    """Message bus with support for streaming responses."""

    def __init__(self):
        super().__init__()
        self._active_streams: Dict[str, asyncio.Queue] = {}

    async def create_stream(self, stream_id: str):
        """Create a new stream for streaming responses."""
        self._active_streams[stream_id] = asyncio.Queue()

    async def send_chunk(self, stream_id: str, chunk: str):
        """Send a chunk to a stream."""
        if stream_id in self._active_streams:
            await self._active_streams[stream_id].put(chunk)

    async def end_stream(self, stream_id: str):
        """End a stream."""
        if stream_id in self._active_streams:
            await self._active_streams[stream_id].put(None)
            del self._active_streams[stream_id]

    async def receive_chunk(self, stream_id: str) -> Optional[str]:
        """Receive a chunk from a stream."""
        if stream_id in self._active_streams:
            return await self._active_streams[stream_id].get()
        return None
