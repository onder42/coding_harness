import json
from typing import List, Dict, Any, Callable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# 1. Define the raw tool functions
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

tools_registry: Dict[str, Callable] = {
    "add": add,
    "multiply": multiply
}

# 2. Simplified Schema format that ChatOllama explicitly looks for
tools_schema = [
    {
        "name": "add",
        "description": "Adds two integers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "multiply",
        "description": "Multiplies two integers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }
]

# Initialize ChatOllama natively with tool bindings
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools_schema)

# 3. Custom Nodes and Graph Components

def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the LLM reasoning step and updates history."""
    print("\n--- [NODE] Agent Thinking ---")
    messages = state["messages"]
    response = llm.invoke(messages)
    state["messages"].append(response)
    return state


def from_scratch_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parses tool arguments, executes the local math, and formats the output."""
    print("--- [NODE] Executing Tools ---")
    last_message = state["messages"][-1]
    
    # FIX: Use ChatOllama's standardized top-level property (.tool_calls)
    tool_calls = getattr(last_message, "tool_calls", [])
    
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        arguments = tool_call["args"]  # Already a clean python dict! No json.loads required.
        
        print(f"Action: Invoking tool '{function_name}' with args {arguments}")
        
        if function_name in tools_registry:
            result = tools_registry[function_name](**arguments)
        else:
            result = f"Error: Tool {function_name} not found."
            
        print(f"Observation: Result is {result}")
        
        # Append the context. The loop will pass this whole string to the agent next turn!
        observation_msg = HumanMessage(
            content=f"Observation from tool '{function_name}': The result is {result}. Now continue evaluating."
        )
        state["messages"].append(observation_msg)
        
    return state


def conditional_router(state: Dict[str, Any]) -> str:
    """Evaluates the state using ChatOllama's standard property structure."""
    last_message = state["messages"][-1]
    
    # Check if the model left behind valid tool calls in the standardized property
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("-> Router Decision: Tool calls detected. Routing to TOOL NODE.")
        return "call_tools"
    
    print("-> Router Decision: Final answer reached. Routing to END.")
    return "end"


# 4. The Orchestration Loop (Manual Graph Cycle)
def run_agent(user_query: str):
    state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    max_iterations = 5
    for _ in range(max_iterations):
        # Step 1: Think (Agent Node)
        state = agent_node(state)
        
        # Step 2: Decide route (Conditional Node)
        next_step = conditional_router(state)
        
        if next_step == "call_tools":
            # Step 3: Act & Observe (Tool Node)
            state = from_scratch_tool_node(state)
            # NO NEED FOR EXTRA INVOKE LOGIC HERE: 
            # The loop automatically circles back up to Step 1 on the next iteration!
        elif next_step == "end":
            print("\n--- Final Agent Response ---")
            print(state["messages"][-1].content)
            break

# Run the query
run_agent("What is 3 plus 5? Take that result and multiply it by 10.")
