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

    # 3. Initialize Executor & Planner LLMs
    base_llm = ChatOllama(model="qwen3.5:2b", temperature=0)
    planner_llm = ChatOllama(model="qwen2.5:3b", temperature=0)

    # 4. Inject Dependencies into the Graph Harness
    harness = AgentHarness(
        llm=base_llm,
        planner_llm=planner_llm,
        tools=active_tools,
        system_prompt=SYSTEM_PROMPT,
        max_turns=8,
    )

    user_prompt = (
        "explore the test_project directory. This is the creation of a Document Object and chunking system for RAG."
        "Make sure the system is working properly. "
    )

    initial_state = AgentState(messages=[HumanMessage(content=user_prompt)])

    try:
        harness.run(initial_state)
    finally:
        sandbox.close()
        print("\n[Harness] Background shell terminated cleanly.")
