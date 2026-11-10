import os
import subprocess
from typing import Dict, Any, List, Callable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama


class PersistentBashSandbox:
    def __init__(self):
        # Start a long-running, interactive bash session in the background
        self.process = subprocess.Popen(
            ['bash', '--noprofile', '--norc'], # Clean environment without user profile noise
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout so we capture errors
            text=True,
            bufsize=1
        )
        # Use a unique marker to signal when a command is finished executing
        self.sentinel = "---COMMAND_FINISHED_SENTINEL---"

    def execute_command(self, command: str) -> str:
        # 1. Clean the command and append our marker
        command = command.strip()
        full_input = f"{command}\necho '{self.sentinel}'\n"
        
        # 2. Write the command directly to the live shell's stdin
        self.process.stdin.write(full_input)
        self.process.stdin.flush()

        # 3. Read stdout line-by-line until our marker appears
        output_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            if self.sentinel in line:
                break
            output_lines.append(line)

        return "".join(output_lines)

    def close(self):
        self.process.terminate()

# Initialize the persistent sandbox background instance globally
sandbox = PersistentBashSandbox()


def run_bash(command: str) -> str:
    """Executes a bash shell script string inside the persistent background session environment."""
    # Harness Permission Gateway
    confirm = input(f"\n[APPROVAL REQUIRED] Run terminal command `{command}`? (y/n): ")
    if confirm.lower() != "y":
        return "User denied permission to execute this shell command."
    
    print("[Harness] Executing in background shell...")
    return sandbox.execute_command(command)

# Map string keys to functional execution pathways
tools_registry: Dict[str, Callable] = {
    "run_bash": run_bash
}

# Declarative JSON Schema optimized specifically for ChatOllama's model parsing engine
tools_schema = [
    {
        "name": "run_bash",
        "description": "Run a command in the persistent background shell.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The explicit bash shell script string to execute."}
            },
            "required": ["command"]
        }
    }
]

# Initialize local Qwen model natively equipped with the capability registry schemas
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools_schema)


def agent_think_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Processes compounding chat history and requests next steps from the model."""
    print(f"\n--- [NODE] Agent Thinking (Turn {state['turns'] + 1}) ---")
    
    if len(state["messages"]) == 1:
        system_msg = HumanMessage(
            content="System Context: You are a secure local coding agent harness equipped with a persistent shell. "
                    "Use your tools to achieve the goal. Note that directory changes (cd) are retained across turns."
        )
        state["messages"].insert(0, system_msg)

    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["turns"] += 1
    return state

def execute_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Parses structural tool arguments, passes them to the environment, and maps responses."""
    print("--- [NODE] Harness Executing Tools ---")
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        arguments = tool_call["args"] 
        tool_call_id = tool_call["id"] # Critical binding identifier for state mapping consistency
        
        print(f"[Harness Intercept] Invoking tool '{function_name}' with parameters: {arguments}")
        
        if function_name in tools_registry:
            result = tools_registry[function_name](**arguments)
        else:
            result = f"Error: Tool name '{function_name}' is not registered in this sandbox architecture."
            
        print(f"[Output]:\n{result}")
        
        if len(str(result)) > 8000:
            result = str(result)[:8000] + "\n... [TRUNCATED BY HARNESS SAFETY LAYER]"
            
        # Secure strict relational binding with ToolMessage mapping to clear upstream confusion
        observation_msg = ToolMessage(
            name=function_name,
            content=str(result),
            tool_call_id=tool_call_id
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


def run_agent_harness(user_query: str, max_turns: int = 3):
    """Orchestrates structural ReAct sequences without hidden framework constraints."""
    state = {
        "messages": [HumanMessage(content=user_query)],
        "turns": 0
    }
    
    for _ in range(max_turns):
        # Step A: Think (Agent Node)
        state = agent_think_node(state)
        for m in state['messages']:print(m)
        
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
        print(f"\n[Harness Warning] Loop interrupted automatically. Safe turn cap limit reached ({max_turns}).")


if __name__ == "__main__":
    run_agent_harness("Create a directory called 'test_project', move into it, create an empty file named 'app.py', and verify its contents.")
