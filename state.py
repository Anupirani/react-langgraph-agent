from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent: just the running list of messages."""
    # add_messages is a reducer: it appends new messages instead of overwriting the list
    messages: Annotated[Sequence[BaseMessage], add_messages]


if __name__ == "__main__":
    state: AgentState = {"messages": []}
    state["messages"] = add_messages(state["messages"], [HumanMessage(content="Hi")])
    state["messages"] = add_messages(state["messages"], [HumanMessage(content="Weather in NYC?")])
    print(len(state["messages"]), "messages in state")
    for m in state["messages"]:
        print(type(m).__name__, "->", m.content)