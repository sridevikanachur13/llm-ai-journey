import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"


# Step 1: The REAL functions that actually run
def get_rainbow_colors():
    """Returns the 7 colors of the rainbow, in order."""
    return ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]


def make_dress(color):
    """Turns a color into a dress description."""
    return f"dress made of {color} color"


TOOL_IMPLS = {
    "get_rainbow_colors": get_rainbow_colors,
    "make_dress": make_dress,
}


# Step 2: Describe the tools to the model in its required format
tools = [
    {
        "function_declarations": [
            {
                "name": "get_rainbow_colors",
                "description": (
                    "Get the ordered list of the seven colors of the rainbow. "
                    "Takes no arguments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "make_dress",
                "description": (
                    "Make a dress out of a single color. Must be called with a color "
                    "that came from get_rainbow_colors."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "The color of the dress, e.g. 'blue'",
                        }
                    },
                    "required": ["color"],
                },
            },
        ]
    }
]


# Step 3: Ask a question that forces the model to chain the two tools
history = [
    {
        "role": "user",
        "parts": [
            {
                "text": (
                    "Get the colors of the rainbow, take the 5th color in that list, "
                    "and make a dress out of it. Tell me what the dress is."
                )
            }
        ],
    }
]


# Step 4: Loop until the model stops asking for tools
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
        print(f"   ↳ result: {result}")

        function_response_parts.append(
            {
                "functionResponse": {
                    "name": name,
                    "response": {"result": result},
                }
            }
        )

    # No tool calls left -> this turn is the final natural-language answer
    if not function_response_parts:
        final_text = "".join(p.get("text", "") for p in parts)
        print("🤖 Final answer:", final_text.strip() or "No response")
        break

    # Feed the tool output back so the model can decide what to do next
    history.append({"role": "model", "parts": parts})
    history.append({"role": "user", "parts": function_response_parts})
else:
    print("⚠️ Hit MAX_TURNS without a final answer.")