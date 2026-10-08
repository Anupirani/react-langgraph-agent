import ast
import operator
import re
from datetime import datetime, timezone

import requests
from ddgs import DDGS
from langchain_core.tools import tool

# ---------------------------------------------------------------- weather

# Open-Meteo returns numeric weather codes; map them to words the model can use
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


@tool
def get_weather(city: str):
    """
    Get the CURRENT weather for a city using live data (Open-Meteo).
    Always use this tool for questions about current weather or temperature.

    :param city: City name, e.g. "Zurich"
    :return: Current temperature in °C, conditions, wind and precipitation
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return {"error": f"City not found: {city}"}
        place = geo["results"][0]

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        cur = wx["current"]
        return {
            "location": f"{place['name']}, {place.get('country', '')}",
            "observed_at": cur["time"],
            "temperature_c": cur["temperature_2m"],
            "feels_like_c": cur["apparent_temperature"],
            "conditions": WEATHER_CODES.get(cur["weather_code"], "Unknown"),
            "precipitation_mm": cur["precipitation"],
            "wind_kmh": cur["wind_speed_10m"],
        }
    except Exception as e:
        # Return the error as data so the agent can see it and adapt
        return {"error": f"Weather lookup failed: {e}"}


@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on a weather description.
    Include the temperature with its unit, e.g. "Light rain, 9°C".

    :param weather: Weather description with temperature (e.g., "Clear sky, 14°C")
    :return: Clothing advice suited to the weather
    """
    w = weather.lower()

    if "snow" in w or "freezing" in w:
        return "Wear a heavy coat, gloves, and boots."
    if any(k in w for k in ("rain", "drizzle", "shower", "thunder", "wet")):
        return "Bring a raincoat and waterproof shoes."

    # Parse a temperature like "14°C" or "57 F" and convert to Celsius
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([cf])\b", w)
    if m:
        temp = float(m.group(1))
        if m.group(2) == "f":
            temp = (temp - 32) * 5 / 9
        if temp <= 0:
            return "Wear a heavy coat, gloves, and boots."
        if temp < 10:
            return "Wear a warm jacket or sweater."
        if temp < 18:
            return "A light jacket should be fine."
        if temp < 27:
            return "A t-shirt or light long sleeves is fine."
        return "T-shirt, shorts, and sunscreen recommended."

    if "hot" in w:
        return "T-shirt, shorts, and sunscreen recommended."
    if "cold" in w:
        return "Wear a warm jacket or sweater."
    return "A light jacket should be fine."


# ---------------------------------------------------------------- search

@tool
def search_tool(query: str):
    """
    Search the web for general information and facts.
    Do NOT use this for current weather (use get_weather) or recent news (use get_news).

    :param query: The search query string
    :return: A list of search results with title, url and snippet
    """
    try:
        results = DDGS().text(query, max_results=5)
        return [
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
            if "bing.com/aclick" not in (r.get("href") or "")
        ]
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]


# ---------------------------------------------------------------- calculator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Evaluate a parsed arithmetic expression. Anything else is rejected."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** % // are allowed")


@tool
def calculator(expression: str):
    """
    Evaluate a basic arithmetic expression exactly. Use this for ANY calculation
    instead of doing math yourself. Supports + - * / ** % // and parentheses.
    Use ** for powers (never ^).
    Write percentages as decimals, e.g. 15% of 2480 -> "0.15 * 2480".

    :param expression: An arithmetic expression, e.g. "(15.6 * 9/5) + 32"
    :return: The numeric result, or an error message
    """
    if len(expression) > 200:
        return {"error": "Expression too long"}
    try:
        tree = ast.parse(expression, mode="eval")
        return {"expression": expression, "result": _safe_eval(tree)}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate: {e}"}


# ---------------------------------------------------------------- news

def _age_in_days(date_str):
    """Best-effort age of an article in days; None if the date can't be parsed."""
    if not date_str:
        return None
    s = date_str.strip()
    m = re.match(r"(\d+)\s*(minute|hour|day|week|month)s?\s+ago", s, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days_per = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30}
        return round(n * days_per[unit], 1)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 86400, 1)
    except ValueError:
        return None


@tool
def get_news(topic: str, max_age_days: int = 14):
    """
    Fetch recent news about a topic. Returns only articles from the last
    max_age_days days (default 14), newest first, each with its source, date
    and age label. Dates come from a search engine and are approximate.
    Articles whose date cannot be determined are left out.

    :param topic: The news topic, e.g. "Swiss National Bank"
    :param max_age_days: Only include articles at most this many days old
    :return: A list of articles, or an error if nothing recent was found
    """
    try:
        results = DDGS().news(topic, max_results=10, timelimit="m")
        articles = []
        for r in results:
            age = _age_in_days(r.get("date"))
            if age is None or age > max_age_days:
                continue
            articles.append({
                "date": (r.get("date") or "")[:10],
                                "age_label": "less than 1 day old" if age < 1 else (
                    "1 day old" if round(age) == 1 else f"{round(age)} days old"),
                "age_days": age,
                "source": r.get("source"),
                "title": r.get("title"),
                "snippet": r.get("body"),
                "url": r.get("url"),
            })
        if not articles:
            return [{"error": f"No articles from the last {max_age_days} days for '{topic}'."}]
        articles.sort(key=lambda a: a["age_days"])
        return articles[:5]
    except Exception as e:
        return [{"error": f"News lookup failed: {e}"}]


# ---------------------------------------------------------------- registry

tools = [get_weather, search_tool, recommend_clothing, calculator, get_news]
tools_by_name = {t.name: t for t in tools}