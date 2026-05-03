"""
AgroBot — Abstract Base Tool
=============================
All feature modules inherit from this base class.
Enforces a consistent interface for the CLI and any future UI.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAgriTool(ABC):
    """
    Abstract base for all AgroBot feature modules.

    Each feature must implement `run()` which accepts:
    - query:      User's question (text)
    - lang:       Detected language code ("en"|"ur"|"pa"|"skr")
    - image_path: Optional path to an uploaded image

    And returns a dict with at minimum:
    - "text":    str — the formatted response
    - "intent":  str — the feature name
    - "lang":    str — language code used
    """

    name: str = "base_tool"
    description: str = "Base agricultural tool"

    @abstractmethod
    def run(
        self,
        query: str,
        lang: str = "en",
        image_path: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """
        Execute the feature and return a response dict.

        Returns:
            {
              "text":    str,          # Main response text
              "intent":  str,          # e.g. "soil_crop"
              "lang":    str,          # e.g. "ur"
              "extra":   dict | None,  # Feature-specific structured data
            }
        """
        ...

    def _error_response(self, lang: str, message: str) -> dict:
        """Standard error response format."""
        return {
            "text":   message,
            "intent": self.name,
            "lang":   lang,
            "extra":  None,
        }
