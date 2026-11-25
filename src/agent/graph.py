from typing import Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
import time
from src.agent.agent_state import AgentState
from src.tools.base import BaseTool


RETRY_CODES = ("503", "504", "429", "UNAVAILABLE", "DEADLINE_EXCEEDED")

class AgentHarness:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_turns: int = 15,
    ):
        self.tools_registry: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        tools_schema = [tool.to_json_schema() for tool in tools]
        
        # Bind schemas to the injected model instance
        self.llm = llm.bind_tools(tools_schema)
        self.system_prompt = system_prompt
        self.max_turns = max_turns


    def agent_think_node(self, state: AgentState) -> AgentState:
        print(f"\n--- [NODE] Agent Thinking (Turn {state.turns + 1}) ---")
        for attempt in range(5):
            try:
                response = self.llm.invoke(state.messages)
                state.add_message(response)
                state.turns += 1
                return state
            except Exception as e:
                if any(code in str(e) for code in RETRY_CODES):
                    wait = 3 * (attempt + 1)
                    print(f"[Harness] Transient API error (attempt {attempt + 1}/5). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Gemini API remained unavailable after 5 retries.")



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

            # Context Pruning Guardrail (prevents context overflow on small local models)
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
        """Router: Evaluates message objects to route execution paths dynamically."""
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("-> Router Decision: Tool calls detected. Routing to ACT.")
            return "call_tools"

        print("-> Router Decision: Final execution target achieved. Routing to END.")
        return "end"

    def run(self, state: AgentState) -> AgentState:
        """Orchestrates structural ReAct sequences using the injected state."""
        for _ in range(self.max_turns):
            # Step A: Think (Agent Node)
            state = self.agent_think_node(state)
            for m in state.messages:
                print(f"\n{m}")

            # Step B: Decide Route (Conditional Router)
            next_step = self.conditional_router(state)

            if next_step == "call_tools":
                # Step C: Act & Observe (Tool Execution Node)
                state = self.execute_tools_node(state)
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
