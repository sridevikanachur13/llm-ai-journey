import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

# Step 1: The REAL function that actually runs (fake data for now)
def get_weather(city):
    fake_data = {
        "Mumbai": "32°C, humid",
        "Delhi": "38°C, sunny",
        "Bangalore": "24°C, rainy"
    }
    return fake_data.get(city, "Unknown city")

def get_time(city): 
    fake_data = {
        "Mumbai": "10:00 AM",
        "Delhi": "11:00 AM",
        "Bangalore": "12:00 AM"
    }
    return fake_data.get(city, "Unknown time")

# Step 2: Describe this tool to the model in its required format
tools = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                }
            },
             {
                "name": "get_time",
                "description": "Get the current time for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                }
            }
        ]
    }
]

# Step 3: Ask a question that requires the tool
history = [
    {"role": "user", "parts": [{"text": "What's the weather like in Bangalore right now? What is the time in Bangalore?"}]}
]

payload = {"contents": history, "tools": tools}
response = requests.post(url, json=payload)
data = response.json()
if "error" in data:
    print("❌ API Error:", data["error"]["message"])
    exit() 
print("Model response:", json.dumps(data, indent=2))
# Step 4: Check if the model wants to call a function
parts = data["candidates"][0]["content"]["parts"]

function_response_parts = []

for part in parts:
    if "functionCall" in part:
        call = part["functionCall"]
        print(f"🔧 Model wants to call: {call['name']} with args {call['args']}")

        if call["name"] == "get_weather":
            result = get_weather(call["args"]["city"])
        elif call["name"] == "get_time":
            result = get_time(call["args"]["city"])
        else:
            result = "Unknown tool"

        # each function call gets its OWN separate response, matched by name
        function_response_parts.append({
            "functionResponse": {
                "name": call["name"],
                "response": {"result": result}
            }
        })

if function_response_parts:
    history.append({"role": "model", "parts": parts})
    history.append({"role": "user", "parts": function_response_parts})

    final_response = requests.post(url, json={"contents": history, "tools": tools})
    final_data = final_response.json()
    final_text = final_data["candidates"][0]["content"]["parts"][0]["text"]
    print("🤖 Final answer:", final_text)
else:
    print("🤖", parts[0].get("text", "No response"))