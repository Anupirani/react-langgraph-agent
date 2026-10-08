from langchain_core.messages import HumanMessage
from graph import graph


def print_stream(stream):
    for s in stream:
        s["messages"][-1].pretty_print()


if __name__ == "__main__":
    # Print the graph structure as Mermaid text (paste into mermaid.live to see it)
    print(graph.get_graph().draw_mermaid())

    inputs = {"messages": [HumanMessage(
        content="What's the weather like in Zurich, and what should I wear based on the temperature?"
    )]}
    # recursion_limit is the graph's version of our MAX_STEPS guardrail
    print_stream(graph.stream(inputs, stream_mode="values", config={"recursion_limit": 12}))