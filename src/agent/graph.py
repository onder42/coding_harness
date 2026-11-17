from typing import Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_state import AgentState
from src.tools.base import BaseTool


class AgentHarness:
    def __init__(
        self,
        llm: BaseChatModel,
        planner_llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_turns: int = 8,
    ):
        self.tools_registry: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        tools_schema = [tool.to_json_schema() for tool in tools]

        # Bind schemas ONLY to the executor model
        self.llm = llm.bind_tools(tools_schema)
        self.planner_llm = planner_llm
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def plan_node(self, state: AgentState) -> AgentState:
        """Node: Generates a clean numbered plan from the user's prompt using qwen2.5:3b."""
        print("\n--- [NODE] Agent Making Plan ---")
        user_goal = state.messages[-1].content

        planner_messages = [
            SystemMessage(
                content=(
                    "You are a simple planner. Break the user's request into a "
                    "short, numbered list of simple steps (1., 2., 3., etc.), keep the number of steps below 8."
                    "The executor AI only has three tools: `run_bash`, `read_file`, and `write_file` :run_bash is a persistant terminal."
                    "Map each step to a specific action (e.g., mkdir/cd, write_file, read_file, run python script) "
                    "and preserve exact file names, libraries, and model names from the prompt. "
                    "Do not write code or explanations. Output ONLY the numbered checklist."
                )
            ),
            HumanMessage(content=user_goal),
        ]


        plan_response = self.planner_llm.invoke(planner_messages)
        plan_text = plan_response.content.strip()

        state.plan = [
            {"step": line.strip(), "done": False}
            for line in plan_text.splitlines()
            if line.strip()
        ]

        print("Generated Plan:")
        for item in state.plan:
            print(f"  {item['step']}")

        state.add_message(
            HumanMessage(
                content=f"Here is your step-by-step execution plan:\n{plan_text}\n\nStart with Step 1."
            )
        )
        return state


    def agent_think_node(self, state: AgentState) -> AgentState:
        """Node: Processes compounding chat history and requests next steps from the model."""
        print(f"\n--- [NODE] Agent Thinking (Turn {state.turns + 1}) ---")

        if not any(isinstance(m, SystemMessage) for m in state.messages):
            state.messages.insert(0, SystemMessage(content=self.system_prompt))

        response = self.llm.invoke(state.messages)
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
        # Step 0: Run Planner once before the ReAct loop starts
        if not state.plan:
            state = self.plan_node(state)

        for _ in range(self.max_turns):
            # Step A: Think (Agent Node)
            state = self.agent_think_node(state)
            for m in state["messages"]:print(f"\n{m}")

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
