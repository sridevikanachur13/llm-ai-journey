import "dotenv/config";

async function askGemini(conversationHistory) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ contents: conversationHistory }),
    },
  );
  const data = await response.json();
  return data.candidates[0].content.parts[0].text;
}

// This array is our "memory" - just like a messages state in React
let history = [];

// Turn 1
history.push({
  role: "user",
  parts: [{ text: "My name is Rahul and I'm a React developer." }],
});
let reply1 = await askGemini(history);
console.log("AI:", reply1);
history.push({ role: "model", parts: [{ text: reply1 }] });

// Turn 2 - notice we're NOT reminding it of the name, it should remember
history.push({
  role: "user",
  parts: [{ text: "What's my name and what do I do?" }],
});
let reply2 = await askGemini(history);
console.log("AI:", reply2);
