import asyncio
import logging
import os
from pathlib import Path

from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logging.getLogger("google.genai.types").setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def multiply_numbers(a: int, b: int) -> int:
    return a * b


def ensure_api_key() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Add GOOGLE_API_KEY (or GEMINI_API_KEY) to .env "
            "or export it in your shell."
        )
    return api_key


async def main():
    ensure_api_key()

    agent = Agent(
        name="math_assistant",
        model="gemini-flash-latest",
        tools=[multiply_numbers],
        instruction="""
    You are a math assistant.
    If user asks for multiplication, call the multiply_numbers tool.
    Return answer in JSON format:
    {
    "result": number
    }
    """
    )

    app_name = "math_assistant_app"
    user_id = "local_user"
    session_id = "session_1"

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text="What is 65 multiplied by 7?")],
    )

    responses = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    )
    final_text = None
    for event in responses:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                final_text = text

    if final_text:
        print(final_text)
    else:
        print("No text response returned.")


if __name__ == "__main__":
    asyncio.run(main())
