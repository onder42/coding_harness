import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.agent_state import AgentState
from src.agent.graph import AgentHarness
from src.tools.base import BaseTool
from src.tools.file_system import ReadFileTool, WriteFileTool
from src.tools.terminal import BashTool, PersistentBashSandbox

load_dotenv()

workspace_root = "/home/chris/Desktop/coding_harness"

SYSTEM_PROMPT = (
    f"You are a focused local coding worker running inside `{workspace_root}`. "
    "Use ONLY relative paths (e.g., `test_project/test_task_manager.py`). "
    "NEVER use `/testbed` or `/workspace`—those directories do not exist on this machine. "
    "Execute all steps in your current assignment using `run_bash`, `read_file`, and `write_file`. "
    "Directory changes (`cd`) persist across turns. When all steps in your assignment are finished, "
    "stop calling tools and write a brief 2-sentence summary of what you built or verified."
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

    base_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    timeout=60,
    max_retries=2,
    )



    harness = AgentHarness(
        llm=base_llm,
        tools=active_tools,
        system_prompt=SYSTEM_PROMPT,
        max_turns=15,
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
