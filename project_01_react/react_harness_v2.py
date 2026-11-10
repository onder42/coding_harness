import os
import subprocess
from typing import Dict, Any, List, Callable
# FIX: Import ToolMessage to correctly map IDs back to the model context window
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

# =====================================================================
# 1. HARNESS TOOLS CONFIGURATION (The Agent's Capability Registry)
# =====================================================================

def read_file(path: str) -> str:
    """Reads the text contents of a target local workspace file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def run_bash(command: str) -> str:
    """Executes a command inside the terminal with an explicit safety gate."""
    # Harness Permission Gateway
    confirm = input(f"\n[APPROVAL REQUIRED] Run terminal command `{command}`? (y/n): ")
    if confirm.lower() != "y":
        return "User denied permission to execute this shell command."
    
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

# Map tool string names directly to the underlying Python functions
tools_registry: Dict[str, Callable] = {
    "read_file": read_file,
    "run_bash": run_bash
}

# Declarative JSON Schemas formatted specifically for ChatOllama's tool parser
tools_schema = [
    {
        "name": "read_file",
        "description": "Read the text contents of a local file safely.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The exact relative or absolute file path."}
            },
            "required": ["path"]
        }
    },
    {
        "name": "run_bash",
        "description": "Execute a safe command line operation inside the terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The explicit bash shell script string to execute."}
            },
            "required": ["command"]
        }
    }
]

# Initialize local Qwen engine natively bundled with the declarative capability schemas
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools_schema)

# =====================================================================
# 2. STATE PIPELINE NODES (Exposed Core Graph Processing Logic)
# =====================================================================

def agent_think_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Processes context history and asks the LLM for its next step."""
    print(f"\n--- [NODE] Agent Thinking (Turn {state['turns'] + 1}) ---")
    
    # Inject an explicit system prompt context if this is the first turn
    if len(state["messages"]) == 1:
        system_msg = HumanMessage(
            content="System Context: You are a secure local coding agent harness. Use tools to inspect or fix things. If a command is denied, try an alternative approach or ask for clarifying guidance."
        )
        state["messages"].insert(0, system_msg)

    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["turns"] += 1
    return state

def execute_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Intercepts tool calls, processes execution blocks, and formats inputs."""
    print("--- [NODE] Harness Executing Tools ---")
    last_message = state["messages"][-1]
    
    # Grab ChatOllama's standardized, pre-parsed top-level tool block dictionary array
    tool_calls = getattr(last_message, "tool_calls", [])
    
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        arguments = tool_call["args"] 
        tool_call_id = tool_call["id"] # FIX: Extract unique tool call ID string reference
        
        print(f"[Harness Intercept] Invoking tool '{function_name}' with arguments: {arguments}")
        
        if function_name in tools_registry:
            result = tools_registry[function_name](**arguments)
        else:
            result = f"Error: Tool name '{function_name}' is not registered in this harness sandbox."
            
        # Context Pruner Safeguard: Basic string slicing to prevent model window exhaustion
        if len(str(result)) > 8000:
            result = str(result)[:8000] + "\n... [TRUNCATED BY HARNESS SAFETY LAYER]"
            
        # FIX: Append structural ToolMessage bound directly to the matching upstream ID
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
    
    # Assess if valid tools require background handling before finishing up the request
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("-> Router Decision: Tool calls detected. Routing to ACT.")
        return "call_tools"
        
    print("-> Router Decision: Final execution target achieved. Routing to END.")
    return "end"

# =====================================================================
# 3. MANUAL GRAPH EXECUTION LOOP (The Harness Engine Core)
# =====================================================================

def run_agent_harness(user_query: str, max_turns: int = 5):
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

# =====================================================================
# 4. INITIALIZATION RUN
# =====================================================================
if __name__ == "__main__":
    run_agent_harness("create a directory in this main project folder then add a python file with hello world program")
