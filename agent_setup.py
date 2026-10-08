from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from tools import tools

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
model = ChatGroq(model=MODEL_NAME, temperature=0)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities
3. Provide clear, helpful responses based on your reasoning and any tool results

For any arithmetic, call the calculator tool. Never do math in your head.

For weather questions:
- Call get_weather ONCE with the city name. Do not use search_tool for current weather.
- Then call recommend_clothing ONCE with a short description built from the result
  (e.g. "Clear sky, 14°C").
- Then give a final answer. Only state facts that appear in tool results, and
  mention the observation time.

For news questions:
- Call get_news ONCE with a short topic. Do not use search_tool for news.
- Title the answer "Recent news about <topic>". Never use the words "latest" or "today".
- Give 3 to 5 bullets. End each bullet with (source, date, age_label), copied
  exactly from the tool result. Never write a date that is not in the tool result.
- Add one short closing line saying dates are approximate.
- If the tool returns an error or nothing, say you could not find recent coverage
  instead of guessing.

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad"),
])

# The prompt feeds a model that knows about our tools
model_react = chat_prompt | model.bind_tools(tools)


if __name__ == "__main__":
    query = "What's the weather like in Zurich, and what should I wear based on the temperature?"
    response = model_react.invoke({"scratch_pad": [HumanMessage(content=query)]})
    print("CONTENT:", repr(response.content))
    print("TOOL CALLS:", response.tool_calls)