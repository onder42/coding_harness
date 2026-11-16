import os
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.agent.agent_state import AgentState
from src.agent.graph import AgentHarness
from src.tools.base import BaseTool
from src.tools.file_system import ReadFileTool, WriteFileTool
from src.tools.terminal import BashTool, PersistentBashSandbox

SYSTEM_PROMPT = (
    "You are a secure local coding agent harness equipped with a persistent shell "
    "and a dedicated file writer. Use `run_bash` for navigating, inspecting, and running scripts. "
    "Use `write_file` whenever creating or updating files. Directory changes (cd) persist across turns."
)



if __name__ == "__main__":

    # 1. Scope Lockdown: Enforce a dedicated sandbox folder
    workspace_root = "/home/chris/Desktop/coding_harness"
    os.makedirs(workspace_root, exist_ok=True)

    # 2. Initialize Shared Sandbox & Tools
    sandbox = PersistentBashSandbox(initial_dir=workspace_root)
    active_tools: list[BaseTool] = [
        BashTool(sandbox),
        ReadFileTool(sandbox, workspace_root),
        WriteFileTool(sandbox, workspace_root),
    ]

    # 3. Initialize LLM & Inject Dependencies into the Graph Harness
    planner_llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    executor_llm = ChatOllama(model="qwen3.5:2b", temperature=0)

    harness = AgentHarness(
        planner_llm=planner_llm,
        executor_llm=executor_llm,
        tools=active_tools,
        system_prompt=SYSTEM_PROMPT,
        max_turns=10,
    )


    user_prompt = (
        "create a  test_project/ directory cd into the test project and create a python files that will run a local Ollama Model as part of a simple Rag." 
        "after writing each file inspect the code re-run it to verify all tests pass."
    )

    initial_state = AgentState(messages=[HumanMessage(content=user_prompt)])

    try:
        harness.run(initial_state)
    finally:
        sandbox.close()
        print("\n[Harness] Background shell terminated cleanly.")

