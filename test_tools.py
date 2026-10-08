from tools import search_tool, recommend_clothing, tools_by_name

print(search_tool.invoke("What's the weather like in Zurich today?"))
print(recommend_clothing.invoke("Light rain, 12°C"))
print(list(tools_by_name))