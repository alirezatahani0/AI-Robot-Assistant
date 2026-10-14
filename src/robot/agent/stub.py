"""Placeholder agent — echo / canned replies until the real API exists."""

from __future__ import annotations

from robot.agent.base import Agent, AgentReply


class StubAgent(Agent):
    """Simple reply path so the listen → think → speak loop is end-to-end."""

    def reply(self, user_text: str) -> AgentReply:
        text = (user_text or "").strip()
        if not text:
            return AgentReply(text="I did not catch that. Please try again.")

        lower = text.lower()
        if lower in {"hi", "hello", "hey"}:
            return AgentReply(text="Hello! I am your robot. How can I help?")
        if "name" in lower:
            return AgentReply(text="I am Robot, still learning. Nice to meet you.")
        if lower in {"bye", "goodbye", "quit", "exit"}:
            return AgentReply(text="Goodbye!")

        return AgentReply(text=f"You said: {text}")
