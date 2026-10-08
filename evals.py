import re
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage

from graph import graph
from tools import calculator


def norm(text: str) -> str:
    """Lowercase and strip spaces/commas so '1,741' and '1 741' both match '1741'."""
    return re.sub(r"[\s,\u202f\u00a0]", "", str(text).lower())


def run_agent(question: str):
    out = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 12},
    )
    msgs = out["messages"]
    called = [tc["name"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls]
    return called, msgs[-1].content


# ---------- Part 1: deterministic tests of the calculator (no LLM involved) ----------
CALC_CASES = [
    ("0.15 * 2480 + 37 ** 2", False),                   # valid
    ("(13.8 * 9/5) + 32", False),                        # valid
    ("37 ^ 2", True),                                    # XOR is not allowed
    ("__import__('os').system('echo hacked')", True),    # injection must be rejected
    ("2 ** 1000", True),                                 # exponent cap
    ("1 / 0", True),                                     # clean error
]

# ---------- Part 2: agent behaviour tests (these call the LLM) ----------
AGENT_CASES = [
    {
        "name": "arithmetic uses the calculator",
        "q": "What is 15% of 2480, plus 37 squared?",
        "must_call": ["calculator"],
        "answer_has_any": ["1741"],
    },
    {
        "name": "weather uses live tool, not web search",
        "q": "What's the weather in Zurich and what should I wear?",
        "must_call": ["get_weather", "recommend_clothing"],
        "must_not_call": ["search_tool", "get_news"],
    },
    {
        "name": "weather in Fahrenheit chains two tools",
        "q": "What is the current temperature in Zurich in Fahrenheit?",
        "must_call": ["get_weather", "calculator"],
    },
    {
        "name": "nonsense news topic is not invented",
        "q": "Summarize the latest news about zxqvbn flurbleton",
        "must_call": ["get_news"],
        "answer_has_any": ["couldn'tfind", "couldnotfind", "unabletofind",
                           "norecent", "noresults", "noarticles", "notfind"],
    },
    {
        "name": "plain chat needs no tools",
        "q": "Say hello in one short sentence.",
        "must_not_call": ["get_weather", "search_tool", "get_news", "calculator",
                          "recommend_clothing"],
    },
 {
        "name": "news answer states ages and avoids 'latest'",
        "q": "Summarize the latest news about Switzerland",
        "must_call": ["get_news"],
        "answer_has_any": ["daysold", "daysago", "lessthan1dayold"],
        "answer_lacks": ["latest"],
    },
]


def main():
    failures = 0

    print("== Calculator unit tests ==")
    for expr, should_error in CALC_CASES:
        result = calculator.invoke({"expression": expr})
        got_error = isinstance(result, dict) and "error" in result
        ok = got_error == should_error
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {expr!r} -> {result}")

    print("\n== Agent behaviour tests ==")
    for case in AGENT_CASES:
        problems = []
        try:
            called, answer = run_agent(case["q"])
            for t in case.get("must_call", []):
                if t not in called:
                    problems.append(f"expected tool call: {t}")
            for t in case.get("must_not_call", []):
                if t in called:
                    problems.append(f"unexpected tool call: {t}")
            if "answer_has_any" in case:
                if not any(norm(p) in norm(answer) for p in case["answer_has_any"]):
                    problems.append(f"answer missing any of {case['answer_has_any']}")
                    for bad in case.get("answer_lacks", []):
                     if norm(bad) in norm(answer):
                      problems.append(f"answer should not contain: {bad!r}")        
                    
        except Exception as e:
            called, answer = [], ""
            problems.append(f"crashed: {type(e).__name__}: {str(e)[:150]}")

        failures += 1 if problems else 0
        print(f"{'FAIL' if problems else 'PASS'}  {case['name']}")
        print(f"      tools called: {called}")
        for p in problems:
            print(f"      problem: {p}")
        time.sleep(2)  # be gentle with the free-tier rate limit

    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()