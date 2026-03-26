"""
Main entry point for the MM RAG Agent system.
"""
import asyncio
import logging
import sys
from pathlib import Path
import uvicorn
from typing import Optional

from config.schema import SystemConfig
from agent.main_agent import MainAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mm_rag_agent.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load system configuration from file or environment."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    config = {
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("API_BASE"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048"))
        },
        "embedding": {
            "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
            "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            "dimension": int(os.getenv("EMBEDDING_DIMENSION", "3072"))
        },
        "vector_store": {
            "type": os.getenv("VECTOR_STORE_TYPE", "chroma"),
            "path": os.getenv("VECTOR_STORE_PATH", "./workspace/vector_store"),
            "collection_name": os.getenv("VECTOR_COLLECTION", "mm_rag")
        },
        "tool_config": {
            "provider": os.getenv("TOOL_PROVIDER", "openai"),
            "provider_config": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("VISION_MODEL", "gpt-4o")
            },
            "store_type": os.getenv("VECTOR_STORE_TYPE", "chroma"),
            "store_path": os.getenv("VECTOR_STORE_PATH", "./workspace/vector_store"),
            "collection_name": os.getenv("VECTOR_COLLECTION", "mm_rag")
        },
        "workspace_path": os.getenv("WORKSPACE_PATH", "./workspace"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
        "context_config": {
            "max_context_length": 128000,
            "max_history": 10,
            "max_retrieved": 20
        },
        "memory_config": {
            "max_short_term_items": 100,
            "max_long_term_items": 10000,
            "archive_path": "./workspace/memory_archive"
        }
    }

    return config


async def run_cli_mode(agent: MainAgent):
    """Run the agent in CLI mode."""
    print("\n=== MM RAG Agent - Interactive Mode ===")
    print("Type 'quit' or 'exit' to stop\n")

    session_id = "cli_session"

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)

            response = await agent.process_message(
                content=user_input,
                session_id=session_id
            )

            print(response.content)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


async def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    # Create workspace directory
    workspace = Path(config["workspace_path"])
    workspace.mkdir(exist_ok=True)

    # Initialize main agent
    logger.info("Initializing MM RAG Agent...")
    agent = MainAgent(config=config)

    # Start agent
    await agent.start()

    try:
        # Run CLI mode
        await run_cli_mode(agent)
    finally:
        # Cleanup
        await agent.stop()
        logger.info("Agent stopped")


class AgentAPI:
    """FastAPI wrapper for the agent."""

    def __init__(self, agent: MainAgent):
        self.agent = agent

    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "stats": self.agent.get_stats()}

    async def chat(self, message: str, session_id: str = "default"):
        """Chat endpoint."""
        response = await self.agent.process_message(
            content=message,
            session_id=session_id
        )
        return {
            "response": response.content,
            "session_id": session_id,
            "is_complete": response.is_complete
        }


def create_api_app(agent: MainAgent):
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="MM RAG Agent API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = AgentAPI(agent)

    class ChatRequest(BaseModel):
        message: str
        session_id: str = "default"

    @app.get("/health")
    async def health():
        return await api.health()

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            return await api.chat(request.message, request.session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MM RAG Agent")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="Run mode")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")

    args = parser.parse_args()

    if args.mode == "api":
        # Run API server
        config = load_config()
        agent = MainAgent(config=config)

        @app.on_event("startup")
        async def startup():
            await agent.start()

        @app.on_event("shutdown")
        async def shutdown():
            await agent.stop()

        app = create_api_app(agent)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Run CLI mode
        asyncio.run(main())
