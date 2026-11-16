from typing import Literal
from pydantic import BaseModel, Field

from src.agent.agent_state import AgentState
from src.tools.base import BaseTool


class PlanStep(BaseModel):
    id: int = Field(..., description="Sequential step number starting at 1.")
    task: str = Field(..., description="Short, actionable description of the step.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        ..., description="Current status of the step."
    )


class UpdatePlanArgs(BaseModel):
    steps: list[PlanStep] = Field(
        ..., description="The complete list of plan steps with updated statuses."
    )


class UpdatePlanTool(BaseTool):
    name = "update_plan"
    description = (
        "Create or update your step-by-step execution plan. "
        "Call this FIRST on multi-step tasks, and update step statuses as you finish them."
    )
    args_schema = UpdatePlanArgs

    def __init__(self, state: AgentState):
        self.state = state

    def execute(self, steps: list[dict]) -> str:
        # validated_args.model_dump() passes `steps` as a list of clean dicts
        self.state.plan = steps
        summary = "\n".join(
            f"  {s['id']}. [{s['status'].upper()}] {s['task']}" for s in steps
        )
        return f"Plan successfully updated:\n{summary}"
