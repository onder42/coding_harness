from dataclasses import dataclass, field
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage


@dataclass
class AgentState:
    messages: List[BaseMessage] = field(default_factory=list)
    turns: int = 0

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
