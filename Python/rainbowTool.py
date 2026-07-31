import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

def web_search(query):
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": tavily_key, "query": query, "max_results": 3},
    )
    data = response.json()
    if "error" in data:
        return f"Search failed: {data['error']}"
    if "results" not in data:
        return f"Search failed: unexpected response shape {data}"
    summaries = [f"{r['title']}: {r['content'][:200]}" for r in data["results"]]
    return "\n\n".join(summaries)

def get_weather(city):
    fake_data = {"Tokyo": "18°C, clear skies", "Paris": "12°C, rainy", "Delhi": "28°C, sunny"}
    return fake_data.get(city, "Weather data not available for this city")

TOOL_IMPLS = {"web_search": web_search, "get_weather": get_weather}

tools = [
    {
        "function_declarations": [
            {
                "name": "web_search",
                "description": "Search the web for current, real-world information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather for: Tokyo, Paris, Delhi.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ]
    }
]

history = [
    {"role": "user", "parts": [{"text": "Search the web to find out what the capital of Japan is. Then tell me the weather in that city."}]}
]

# ===== Guardrail 2: structured log, not just print statements =====
execution_log = []

# ===== Guardrail 3: track the LAST call's (name, args) to detect exact repeats =====
last_call_signature = None

MAX_TURNS = 6
for turn in range(MAX_TURNS):
    response = requests.post(url, json={"contents": history, "tools": tools})
    data = response.json()

    if "error" in data:
        execution_log.append({"turn": turn + 1, "type": "api_error", "detail": data["error"]["message"]})
        print("❌ API Error:", data["error"]["message"])
        break

    parts = data["candidates"][0]["content"]["parts"]
    function_response_parts = []
    stuck_in_loop = False

    for part in parts:
        if "functionCall" not in part:
            continue
        call = part["functionCall"]
        name = call["name"]
        args = call.get("args") or {}

        # Guardrail 3: same tool + same args as last time = stuck, stop
        current_signature = (name, json.dumps(args, sort_keys=True))
        if current_signature == last_call_signature:
            execution_log.append({"turn": turn + 1, "type": "stuck_detected", "tool": name, "args": args})
            print(f"⚠️ Agent repeated the exact same call ({name}, {args}) — stopping to avoid a wasted loop.")
            stuck_in_loop = True
            break
        last_call_signature = current_signature

        fn = TOOL_IMPLS.get(name)
        try:
            result = fn(**args) if fn else f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool execution error: {e}"

        # Guardrail 2: record the full step, not just print it
        execution_log.append({
            "turn": turn + 1,
            "type": "tool_call",
            "tool": name,
            "args": args,
            "result": result,
        })
        print(f"🔧 Turn {turn+1} — {name}({args}) → {result[:100]}")

        function_response_parts.append(
            {"functionResponse": {"name": name, "response": {"result": result}}}
        )

    if stuck_in_loop:
        break

    if not function_response_parts:
        final_text = "".join(p.get("text", "") for p in parts)
        execution_log.append({"turn": turn + 1, "type": "final_answer", "text": final_text.strip()})
        print("🤖 Final answer:", final_text.strip() or "No response")
        break

    history.append({"role": "model", "parts": parts})
    history.append({"role": "user", "parts": function_response_parts})
else:
    execution_log.append({"turn": MAX_TURNS, "type": "max_turns_hit"})
    print("⚠️ Hit MAX_TURNS without a final answer.")

# Show the full structured trace at the end
print("\n=== EXECUTION LOG ===")
for entry in execution_log:
    print(entry)