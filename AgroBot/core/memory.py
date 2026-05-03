"""
AgroBot — Conversation Memory
==============================
Thin wrapper around a session-level message list.
Keeps HumanMessage / AIMessage pairs for multi-turn context.
"""

from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    """Simple in-memory conversation history for one AgroBot session."""

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: Maximum number of conversation turns to remember.
                       Older turns are dropped to keep context window manageable.
        """
        self.max_turns = max_turns
        self._messages: list = []
        self.context = {"location": None, "crop": None}

    def add_exchange(self, user_text: str, bot_text: str) -> None:
        """Add a user query + bot response to memory."""
        self._messages.append(HumanMessage(content=user_text))
        self._messages.append(AIMessage(content=bot_text))

        # Trim to last max_turns × 2 messages (each turn = 1 human + 1 AI)
        max_msgs = self.max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]

    def update_context(self, location: str = None, crop: str = None) -> None:
        """Update the persistent context with new extracted entities."""
        if location:
            self.context["location"] = location
        if crop:
            self.context["crop"] = crop

    def get_messages(self) -> list:
        """Return the full message history list."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset conversation history and context."""
        self._messages = []
        self.context = {"location": None, "crop": None}

    def turn_count(self) -> int:
        """Return number of completed turns."""
        return len(self._messages) // 2

    def is_empty(self) -> bool:
        return len(self._messages) == 0
