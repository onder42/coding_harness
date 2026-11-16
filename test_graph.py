from typing import Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_state import AgentState
from src.tools.base import BaseTool


class AgentHarness:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_turns: int = 10,
    ):
        self.tools_registry: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        tools_schema = [tool.to_json_schema() for tool in tools]

        self.llm = llm.bind_tools(tools_schema)
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def agent_think_node(self, state: AgentState) -> AgentState:
        """Node: Processes chat history + live plan status and requests next steps."""
        print(f"\n--- [NODE] Agent Thinking (Turn {state.turns + 1}) ---")

        if not any(isinstance(m, SystemMessage) for m in state.messages):
            state.messages.insert(0, SystemMessage(content=self.system_prompt))

        # Build a temporary prompt view so the live plan is always fresh at the end
        prompt_messages = list(state.messages)
        if state.plan:
            plan_lines = "\n".join(
                f"- Step {s['id']} [{s['status'].upper()}]: {s['task']}"
                for s in state.plan
            )
            prompt_messages.append(
                SystemMessage(
                    content=(
                        f"CURRENT PLAN CHECKLIST:\n{plan_lines}\n\n"
                        "Execute the next pending or in-progress step, then call `update_plan` to mark it completed."
                    )
                )
            )

        response = self.llm.invoke(prompt_messages)
        state.add_message(response)
        state.turns += 1
        return state

    def execute_tools_node(self, state: AgentState) -> AgentState:
        """Node: Executes requested tools and appends ToolMessage observations."""
        print("--- [NODE] Harness Executing Tools ---")
        last_message = state.messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        for tool_call in tool_calls:
            function_name = tool_call["name"]
            arguments = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"[Harness Intercept] Invoking '{function_name}' with: {arguments}")

            if function_name in self.tools_registry:
                result = self.tools_registry[function_name].validate_and_execute(**arguments)
            else:
                result = f"Error: Tool '{function_name}' is not registered."

            result_str = str(result)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "\n... [TRUNCATED BY HARNESS SAFETY LAYER]"

            observation_msg = ToolMessage(
                name=function_name,
                content=result_str,
                tool_call_id=tool_call_id,
            )
            state.add_message(observation_msg)

        return state

    def conditional_router(self, state: AgentState) -> str:
        """Router: Routes to tools, enforces plan completion, or exits to human."""
        last_message = state.messages[-1]

        # 1. If the model requested tools, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("-> Router Decision: Tool calls detected. Routing to ACT.")
            return "call_tools"

        # 2. Guardrail: Block early exit if plan steps remain unfinished
        unfinished = [s for s in state.plan if s.get("status") != "completed"]
        if unfinished:
            next_task = unfinished[0]
            print(
                f"-> Router Decision: Unfinished plan step {next_task['id']} detected. Nudging agent."
            )
            state.add_message(
                HumanMessage(
                    content=(
                        f"You stopped early, but Step {next_task['id']} ('{next_task['task']}') "
                        "is still not completed. Complete the remaining steps and update your plan."
                    )
                )
            )
            return "continue"

        print("-> Router Decision: All plan steps complete. Routing to END.")
        return "end"

    def run(self, state: AgentState) -> AgentState:
        """Orchestrates structural ReAct + Plan sequences using the injected state."""
        for _ in range(self.max_turns):
            state = self.agent_think_node(state)

            next_step = self.conditional_router(state)

            if next_step == "call_tools":
                state = self.execute_tools_node(state)
            elif next_step == "continue":
                # Skip tool execution and let the agent think again with the nudge
                continue
            elif next_step == "end":
                print("\n=================== FINAL HARNESS RESPONSE ===================")
                print(state.messages[-1].content)
                print("==============================================================")
                break
        else:
            print(
                f"\n[Harness Warning] Loop interrupted automatically. Safe turn cap limit reached ({self.max_turns})."
            )

        return state
