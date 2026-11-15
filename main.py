import os
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from src.tools.base import BaseTool
from src.tools.file_system import ReadFileTool, WriteFileTool
from src.tools.terminal import BashTool, PersistentBashSandbox

# 1. Scope Lockdown: Enforce a dedicated sandbox folder
WORKSPACE_ROOT = "/home/chris/Desktop/coding_harness"
os.makedirs(WORKSPACE_ROOT, exist_ok=True)

# 2. Initialize Shared Sandbox & Tools
sandbox = PersistentBashSandbox(initial_dir=WORKSPACE_ROOT)
active_tools: list[BaseTool] = [
    BashTool(sandbox),
    ReadFileTool(sandbox, WORKSPACE_ROOT),
    WriteFileTool(sandbox, WORKSPACE_ROOT),
]

# 3. Auto-generate registry and schemas from BaseTool instances
tools_registry: Dict[str, BaseTool] = {tool.name: tool for tool in active_tools}
tools_schema = [tool.to_json_schema() for tool in active_tools]

# 4. Bind auto-generated schemas to local Qwen model
llm = ChatOllama(model="qwen3.5:2b", temperature=0).bind_tools(tools_schema)


def agent_think_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Processes compounding chat history and requests next steps from the model."""
    print(f"\n--- [NODE] Agent Thinking (Turn {state['turns'] + 1}) ---")

    if len(state["messages"]) == 1:
        system_msg = SystemMessage(
            content=(
                "You are a secure local coding agent harness equipped with a persistent shell "
                "and a dedicated file writer. Use `run_bash` for navigating, inspecting, and running scripts. "
                "Use `write_file` whenever creating or updating files. Directory changes (cd) persist across turns."
            )
        )
        state["messages"].insert(0, system_msg)

    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["turns"] += 1
    return state


def execute_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [NODE] Harness Executing Tools ---")
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    for tool_call in tool_calls:
        function_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"[Harness Intercept] Invoking '{function_name}' with: {arguments}")

        if function_name in tools_registry:
            result = tools_registry[function_name].validate_and_execute(**arguments)
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
        state["messages"].append(observation_msg)

    return state


def conditional_router(state: Dict[str, Any]) -> str:
    """Router: Evaluates message objects to route execution paths dynamically."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("-> Router Decision: Tool calls detected. Routing to ACT.")
        return "call_tools"

    print("-> Router Decision: Final execution target achieved. Routing to END.")
    return "end"


def run_agent_harness(user_query: str, max_turns: int = 5):
    """Orchestrates structural ReAct sequences without hidden framework constraints."""
    state = {"messages": [HumanMessage(content=user_query)], "turns": 0}

    for _ in range(max_turns):
        # Step A: Think (Agent Node)
        state = agent_think_node(state)
        for m in state["messages"]:
            print(f"\n{m}")

        # Step B: Decide Route (Conditional Router)
        next_step = conditional_router(state)

        if next_step == "call_tools":
            # Step C: Act & Observe (Tool Execution Node)
            state = execute_tools_node(state)
        elif next_step == "end":
            print("\n=================== FINAL HARNESS RESPONSE ===================")
            print(state["messages"][-1].content)
            print("==============================================================")
            break

    if state["turns"] >= max_turns:
        print(
            f"\n[Harness Warning] Loop interrupted automatically. Safe turn cap limit reached ({max_turns})."
        )


if __name__ == "__main__":
    try:
        run_agent_harness(
            "cd to '/home/chris/Desktop/coding_harness/test_project'. I would like pyton file that contains a Class for a  RAG system that will take a document and chunch it. use the content and chunk it to paragraph, sentens, word. "
        )
    finally:
        sandbox.close()
        print("\n[Harness] Background shell terminated cleanly.")
