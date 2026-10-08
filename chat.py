from langchain_core.messages import HumanMessage
from graph import graph_with_memory

# Same thread_id = same conversation. Change it to start a fresh one.
config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": 12}

print("Chat with the agent. Type 'quit' to exit.")
while True:
    user = input("\nYou: ").strip()
    if user.lower() in ("quit", "exit"):
        break
    if not user:
        continue
    result = graph_with_memory.invoke(
        {"messages": [HumanMessage(content=user)]}, config=config
    )
    print("\nAgent:", result["messages"][-1].content)