"""Agent / brain contract — real API lands in a later phase."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentReply:
    text: str


class Agent(ABC):
    @abstractmethod
    def reply(self, user_text: str) -> AgentReply:
        """Produce a reply for spoken user text."""
