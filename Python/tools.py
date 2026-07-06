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
            }
        ]
    }
]

# Step 3: Ask a question that requires the tool
history = [
    {"role": "user", "parts": [{"text": "What's the weather like in Bangalore right now?"}]}
]

payload = {"contents": history, "tools": tools}
response = requests.post(url, json=payload)
data = response.json()
print("Model response:", json.dumps(data, indent=2))
# Step 4: Check if the model wants to call a function
candidate = data["candidates"][0]["content"]["parts"][0]

if "functionCall" in candidate:
    call = candidate["functionCall"]
    print(f"🔧 Model wants to call: {call['name']} with args {call['args']}")

    # Step 5: YOUR code actually runs the function
    if call["name"] == "get_weather":
        result = get_weather(call["args"]["city"])

    # Step 6: Send the result back to the model
    history.append({"role": "model", "parts": [{"functionCall": call}]})
    history.append({
        "role": "user",
        "parts": [{"functionResponse": {"name": call["name"], "response": {"result": result}}}]
    })

    final_response = requests.post(url, json={"contents": history, "tools": tools})
    final_data = final_response.json()
    final_text = final_data["candidates"][0]["content"]["parts"][0]["text"]
    print("🤖 Final answer:", final_text)
else:
    print("🤖", candidate.get("text", "No response"))