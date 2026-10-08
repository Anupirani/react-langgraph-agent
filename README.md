# ReAct Agent with LangGraph

A tool-using AI agent that reasons, calls tools, observes results, and repeats until it can answer (the ReAct pattern). Built with LangGraph and Groq's free tier, so no paid API is needed.

Based on the IBM Skills Network lab "Build Reasoning and Acting AI Agents with LangGraph", with these changes:
- Groq (`openai/gpt-oss-20b`) instead of OpenAI
- A live weather tool (Open-Meteo) instead of web search for weather
- A safe calculator (AST-based, no `eval`)
- A news tool that filters out stale articles
- Conversation memory and an automated test suite

## Tools
| Tool | Purpose |
|---|---|
| `get_weather` | Current weather from Open-Meteo (no key needed) |
| `recommend_clothing` | Clothing advice from conditions and temperature |
| `calculator` | Arithmetic via a restricted parser; rejects anything else |
| `get_news` | Recent news (DuckDuckGo), newest first, with age labels |
| `search_tool` | General web search (DuckDuckGo) |

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # then put your free Groq key in .env
```
Get a free key at https://console.groq.com.

## Run
```bash
python ask.py "What is the current temperature in Zurich in Fahrenheit?"
python chat.py                   # multi-turn chat with memory
python evals.py                  # test suite
```

## Project structure
- `tools.py`: tool definitions and registry
- `agent_setup.py`: model, system prompt, tool binding
- `state.py`, `graph.py`: LangGraph state and the agent/tools loop
- `manual_react.py`: the same loop written by hand, for learning
- `ask.py`, `chat.py`, `run_agent.py`: ways to run the agent
- `evals.py`: calculator unit tests and agent behaviour tests

## What I learned
- A noisy search tool made the agent present a March weather page as today's weather. A dedicated API tool fixed it, which was more reliable than prompt changes.
- Small models sometimes emit malformed tool calls, so the graph retries.
- News dates from the free source are approximate, so the tool filters by age and the prompt requires copying dates verbatim.

## Limitations
- Free-tier rate limits apply, and a 20B model can occasionally misbehave.
- News and search results come from a free source and vary in quality.
- Memory in `chat.py` is in RAM only.