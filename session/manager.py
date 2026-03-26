"""
Session manager for handling multiple user sessions.
"""
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Session:
    """Represents a single user session."""

    def __init__(
        self,
        session_id: str,
        workspace_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.workspace_path = workspace_path
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()

        self._messages: List[Dict[str, Any]] = []
        self._context: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session history."""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        self.last_active = datetime.utcnow()

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from the session."""
        if limit:
            return self._messages[-limit:]
        return self._messages

    def set_context(self, key: str, value: Any):
        """Set a context variable."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self._context.get(key, default)

    def set_state(self, key: str, value: Any):
        """Set a state variable."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state variable."""
        return self._state.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "messages": self._messages,
            "context": self._context,
            "state": self._state
        }


class SessionManager:
    """
    Manages multiple user sessions.
    Handles session creation, retrieval, and persistence.
    """

    def __init__(self, base_workspace: str = "./workspace/sessions"):
        self.base_workspace = Path(base_workspace)
        self.base_workspace.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, Session] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        _lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        session_path = self.base_workspace / session_id
        session_path.mkdir(exist_ok=True)

        session = Session(
            session_id=session_id,
            workspace_path=str(session_path),
            metadata=metadata
        )

        async with asyncio.Lock():
            self._sessions[session_id] = session
            self._locks[session_id] = asyncio.Lock()

        logger.info(f"Created session: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create new one."""
        session = await self.get_session(session_id)
        if session is None:
            session = await self.create_session(session_id)
        return session

    async def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._locks:
            del self._locks[session_id]

        session_path = self.base_workspace / session_id
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)

        logger.info(f"Deleted session: {session_id}")

    async def save_session(self, session_id: str):
        """Persist session to disk."""
        session = await self.get_session(session_id)
        if session is None:
            return

        session_file = self.base_workspace / session_id / "session.json"
        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        logger.info(f"Saved session: {session_id}")

    async def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from disk."""
        session_file = self.base_workspace / session_id / "session.json"
        if not session_file.exists():
            return None

        with open(session_file, "r") as f:
            data = json.load(f)

        session = Session(
            session_id=data["session_id"],
            workspace_path=data["workspace_path"],
            metadata=data.get("metadata", {})
        )
        session._messages = data.get("messages", [])
        session._context = data.get("context", {})
        session._state = data.get("state", {})

        self._sessions[session_id] = session
        self._locks[session_id] = asyncio.Lock()

        logger.info(f"Loaded session: {session_id}")
        return session

    async def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    async def cleanup_inactive(self, max_age_seconds: int = 86400):
        """Clean up sessions inactive for max_age_seconds."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        to_delete = []

        for session_id, session in self._sessions.items():
            if session.last_active < cutoff:
                to_delete.append(session_id)

        for session_id in to_delete:
            await self.delete_session(session_id)

        logger.info(f"Cleaned up {len(to_delete)} inactive sessions")
