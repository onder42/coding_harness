from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError


class BaseTool(ABC):
    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Must implement real execution logic."""
        pass

    def validate_and_execute(self, **kwargs) -> str:
        """Validates LLM arguments against Pydantic before running execute()."""
        try:
            validated_args = self.args_schema(**kwargs)
            return self.execute(**validated_args.model_dump())
        except ValidationError as e:
            return f"Argument Validation Error for '{self.name}': {e}"

    def to_json_schema(self) -> dict:
        """Automatically transforms the Pydantic schema for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }
