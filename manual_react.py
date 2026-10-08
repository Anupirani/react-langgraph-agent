import json
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from agent_setup import model_react
from state import AgentState
from tools import tools_by_name

query = "What's the weather like in Zurich, and what should I wear based on the temperature?"
state: AgentState = {"messages": [HumanMessage(content=query)]}

MAX_STEPS = 6  # safety cap so a confused model can't loop forever

for step in range(1, MAX_STEPS + 1):
    # THINK: the model reads the full history and decides what to do
    response = model_react.invoke({"scratch_pad": state["messages"]})
    state["messages"] = add_messages(state["messages"], [response])

    print(f"\n=== Step {step}: THINK ===")
    print("Model said:", response.content or "(no text, only tool calls)")

    # No tool calls means the model is ready to give its final answer
    if not response.tool_calls:
        print("\n>>> FINAL ANSWER reached, loop ends.")
        print(response.content)
        break

    # ACT + OBSERVE: run every tool the model asked for
    for tc in response.tool_calls:
        print(f"ACT: {tc['name']}({tc['args']})")
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        print("OBSERVE:", str(result)[:300])

        state["messages"] = add_messages(
            state["messages"],
            [ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=tc["name"],
                tool_call_id=tc["id"],  # links this result to the model's request
            )],
        )
else:
    # Runs only if the loop finished WITHOUT hitting `break`
    print("\n!!! Hit MAX_STEPS without a final answer")