import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

# ===== Step 1: REAL functions =====

def web_search(query):
    """Calls Tavily's real search API and returns a short summary of top results."""
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": tavily_key, "query": query, "max_results": 3},
    )
    data = response.json()
    if "results" not in data:
        return f"Search failed: {data}"
    # Combine top results into a compact string the model can reason over
    summaries = [f"{r['title']}: {r['content'][:200]}" for r in data["results"]]
    return "\n\n".join(summaries)

def get_weather(city):
    """Fake weather lookup - kept simple since today's focus is the search tool."""
    fake_data = {"Tokyo": "18°C, clear skies", "Paris": "12°C, rainy", "Delhi": "28°C, sunny"}
    return fake_data.get(city, "Weather data not available for this city")

TOOL_IMPLS = {
    "web_search": web_search,
    "get_weather": get_weather,
}

# ===== Step 2: Tool descriptions =====

tools = [
    {
        "function_declarations": [
            {
                "name": "web_search",
                "description": "Search the web for current, real-world information. Use this for facts you don't already know, like current events, people, or places.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get the current weather for a specific city. Only works for: Tokyo, Paris, Delhi.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"],
                },
            },
        ]
    }
]

# ===== Step 3: A question forcing search -> reasoning -> second tool =====

history = [
    {
        "role": "user",
        "parts": [
            {
                "text": (
                    "Search the web to find out what the capital of Japan is. "
                    "Then tell me the weather in that city."
                )
            }
        ],
    }
]

# ===== Step 4: Same loop as Day 29 =====

MAX_TURNS = 6
for turn in range(MAX_TURNS):
    response = requests.post(url, json={"contents": history, "tools": tools})
    data = response.json()

    if "error" in data:
        print("❌ API Error:", data["error"]["message"])
        break

    parts = data["candidates"][0]["content"]["parts"]
    function_response_parts = []

    for part in parts:
        if "functionCall" not in part:
            continue
        call = part["functionCall"]
        name = call["name"]
        args = call.get("args") or {}
        print(f"🔧 Model wants to call: {name} with args {args}")

        fn = TOOL_IMPLS.get(name)
        result = fn(**args) if fn else f"Unknown tool: {name}"
        print(f"   ↳ result: {result}\n")

        function_response_parts.append(
            {"functionResponse": {"name": name, "response": {"result": result}}}
        )

    if not function_response_parts:
        final_text = "".join(p.get("text", "") for p in parts)
        print("🤖 Final answer:", final_text.strip() or "No response")
        break

    history.append({"role": "model", "parts": parts})
    history.append({"role": "user", "parts": function_response_parts})
else:
    print("⚠️ Hit MAX_TURNS without a final answer.")