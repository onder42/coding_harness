# main.py
import os
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from src.agent.agent_state import AgentState
from src.agent.graph import AgentHarness
from src.tools.base import BaseTool
from src.tools.file_system import ReadFileTool, WriteFileTool
from src.tools.terminal import BashTool, PersistentBashSandbox

workspace_root = "/home/chris/Desktop/coding_harness"

SYSTEM_PROMPT = (
    f"You are a local coding worker. Your current working directory is already {workspace_root}. "
    "Run commands directly using relative paths like `python3 test_project/test_task_manager.py`. "
    "Do not run `cd` before commands. "
    "When all steps in your assignment are finished, stop calling tools and write a brief 2-sentence summary."
)


if __name__ == "__main__":
    # 1. Scope Lockdown: Enforce a dedicated sandbox folder
    os.makedirs(workspace_root, exist_ok=True)

    # 2. Initialize Shared Sandbox & Tools
    sandbox = PersistentBashSandbox(initial_dir=workspace_root)
    active_tools: list[BaseTool] = [
        BashTool(sandbox),
        ReadFileTool(sandbox, workspace_root),
        WriteFileTool(sandbox, workspace_root),
    ]

    # Single 3B model for both Planner and Executor -> ZERO RAM swapping
    shared_llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    harness = AgentHarness(
        llm=shared_llm,
        planner_llm=shared_llm,
        tools=active_tools,
        system_prompt=SYSTEM_PROMPT,
        max_turns_per_task=10,
    )


    try:
        while True:
            user_prompt = input("\nEnter Prompt (or 'q' to quit): ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in {"q", "quit", "end"}:
                break

            # Fresh AgentState dumps all prior LLM messages on every prompt
            initial_state = AgentState(messages=[HumanMessage(content=user_prompt)])
            harness.run(initial_state)

    except KeyboardInterrupt:
        print("\n[Harness] Interrupted by user.")
    finally:
        sandbox.close()
        print("\n[Harness] Background shell terminated cleanly.")
