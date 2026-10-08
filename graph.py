from langgraph.checkpoint.memory import MemorySaver
from groq import BadRequestError
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END

from agent_setup import model_react
from state import AgentState
from tools import tools_by_name


def call_model(state: AgentState):
    """THINK: invoke the model, retrying if it produces a malformed tool call."""
    last_error = None
    for attempt in range(3):
        try:
            response = model_react.invoke({"scratch_pad": state["messages"]})
            return {"messages": [response]}
        except BadRequestError as e:
            last_error = e
            print(f"[retry {attempt + 1}/3] invalid tool call from model, trying again...")
    raise last_error

def tool_node(state: AgentState):
    """ACT + OBSERVE: run every tool call from the model's last message."""
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=tc["name"],
                tool_call_id=tc["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState):
    """Route: more tools needed, or are we done?"""
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge("tools", "agent")  # after tools, always go back to the model
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
workflow.set_entry_point("agent")

graph = workflow.compile()
graph_with_memory = workflow.compile(checkpointer=MemorySaver())