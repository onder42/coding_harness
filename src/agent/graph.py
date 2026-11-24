from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.agent.agent_state import AgentState
from src.tools.base import BaseTool


# --- Structured Schema for 3B Planner ONLY ---
class TaskItem(BaseModel):
    task_id: int = Field(..., description="Sequential task number starting at 1.")
    title: str = Field(..., description="Short name of the task.")
    instructions: str = Field(
        ...,
        description="Specific, ordered steps for the worker to execute in this task (e.g., Step 1: ..., Step 2: ...).",
    )


class MasterPlan(BaseModel):
    tasks: List[TaskItem] = Field(
        ..., description="2 to 4 sequential tasks to complete the entire project."
    )


class AgentHarness:
    def __init__(
        self,
        llm: BaseChatModel,
        planner_llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_turns_per_task: int = 6,
    ):
        self.tools_registry: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        # Bind ONLY execution tools (bash, read, write) to the 2B model
        self.llm = llm.bind_tools([tool.to_json_schema() for tool in tools])
        self.planner_llm = planner_llm.with_structured_output(MasterPlan)
        self.system_prompt = system_prompt
        self.max_turns_per_task = max_turns_per_task

    def create_initial_plan(self, state: AgentState) -> AgentState:
        """Runs ONCE using qwen2.5:3b to create the master task list."""
        print("\n--- [PLANNER 3B] Generating Master Task List ---")
        state.user_goal = next(
            (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
            "",
        )

        planner_messages = [
            SystemMessage(
                content=(
                    "You are a senior technical lead. Break the user's request into 2 to 4 "
                    "self-contained Tasks. Group tightly coupled actions into a single Task. "
                    "Always instruct the worker to read existing source files before writing or fixing tests. "
                    "Do not write code. Keep instructions concrete with exact file names."
                )
            ),
            HumanMessage(content=f"Goal:\n{state.user_goal}"),
        ]


        master_plan: MasterPlan = self.planner_llm.invoke(planner_messages)
        state.plan = [
            {
                "task_id": t.task_id,
                "title": t.title,
                "instructions": t.instructions,
                "done": False,
            }
            for t in master_plan.tasks
        ]

        for t in state.plan:
            print(f"  [Task {t['task_id']}] {t['title']}: {t['instructions']}")
        return state

    def _prepare_task_context(self, state: AgentState, current_task: dict) -> None:
        """Wipes noisy tool outputs from prior tasks and gives the 2B model a clean slate."""
        history_block = (
            "\n".join(f"- {s}" for s in state.completed_summaries)
            if state.completed_summaries
            else "None yet."
        )

        task_prompt = (
            f"OVERALL GOAL: {state.user_goal}\n\n"
            f"COMPLETED SO FAR:\n{history_block}\n\n"
            f"YOUR CURRENT ASSIGNMENT (Task {current_task['task_id']} - {current_task['title']}):\n"
            f"{current_task['instructions']}\n\n"
            "Execute all steps in your current assignment using your tools. "
            "When finished, stop calling tools and write a 2-sentence summary of what you created or verified."
        )

        # Reset turns and messages so the 2B model stays fast and focused
        state.turns = 0
        state.messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=task_prompt),
        ]

    def agent_think_node(self, state: AgentState) -> AgentState:
        print(f"\n--- [EXECUTOR 2B] Thinking (Turn {state.turns + 1}) ---")
        response = self.llm.invoke(state.messages)
        state.add_message(response)
        state.turns += 1
        return state

    def execute_tools_node(self, state: AgentState) -> AgentState:
        print("--- [HARNESS] Executing Tools ---")
        last_message = state.messages[-1]
        for tool_call in getattr(last_message, "tool_calls", []):
            fn_name = tool_call["name"]
            args = tool_call["args"]
            print(f"[Harness Intercept] '{fn_name}' -> {args}")

            if fn_name in self.tools_registry:
                result = self.tools_registry[fn_name].validate_and_execute(**args)
            else:
                result = f"Error: Tool '{fn_name}' is not registered."

            result_str = str(result)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "\n... [TRUNCATED]"

            state.add_message(
                ToolMessage(name=fn_name, content=result_str, tool_call_id=tool_call["id"])
            )
        return state

    def run(self, state: AgentState) -> AgentState:
        # 1. Generate the master plan ONCE with the 3B model
        if not state.plan:
            state = self.create_initial_plan(state)

        # 2. Loop through each Task deterministically in Python
        for task in state.plan:
            if task["done"]:
                continue

            print(f"\n==================================================")
            print(f"STARTING TASK {task['task_id']}: {task['title']}")
            print(f"==================================================")

            # Clean up old messages and inject fresh context for this task
            self._prepare_task_context(state, task)

            for _ in range(self.max_turns_per_task):
                state = self.agent_think_node(state)
                last_msg = state.messages[-1]
                for m in state.messages:
                    print(f"\n{m}")

                if getattr(last_msg, "tool_calls", None):
                    state = self.execute_tools_node(state)
                else:
                    # 2B model finished the task and returned its summary!
                    summary = last_msg.content.strip()
                    task["done"] = True
                    state.completed_summaries.append(
                        f"Task {task['task_id']} ({task['title']}): {summary}"
                    )
                    print(f"\n[Task {task['task_id']} Complete] Summary: {summary}")
                    break
            else:
                print(
                    f"\n[Harness Warning] Task {task['task_id']} hit max turns ({self.max_turns_per_task})."
                )
                break

        print("\n=================== ALL TASKS COMPLETE ===================")
        for s in state.completed_summaries:
            print(f"✓ {s}")
        return state
