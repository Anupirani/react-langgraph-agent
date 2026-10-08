import sys
from langchain_core.messages import HumanMessage
from graph import graph

query = " ".join(sys.argv[1:]) or "What is 15% of 2480, plus 37 squared?"
inputs = {"messages": [HumanMessage(content=query)]}

for step in graph.stream(inputs, stream_mode="values", config={"recursion_limit": 12}):
    step["messages"][-1].pretty_print()