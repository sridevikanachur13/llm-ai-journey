import "dotenv/config";

async function askGemini(conversationHistory) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        contents: conversationHistory,
      }),
    },
  );
  const data = await response.json();
  return data.candidates[0].content.parts[0].text;
}

let reply;
let history = [];
// 1. Role assignment
history.push({
  role: "user",
  parts: [{ text: "You are a strict code reviewer. Review this: const x = 1" }],
});
reply = await askGemini(history);
console.log("AI:", reply);

// 2. Few-shot
history.push({
  role: "user",
  parts: [{ text: "Convert to past tense.\nRun -> Ran\nEat -> Ate\nGo ->" }],
});
reply = await askGemini(history);
console.log("AI:", reply);

// 3. Explicit format
history.push({
  role: "user",
  parts: [
    { text: "List 3 React hooks. Respond ONLY as JSON array of strings." },
  ],
});
reply = await askGemini(history);
console.log("AI:", reply);

// 4. Step by step
history.push({
  role: "user",
  parts: [
    { text: "Is 847 a prime number? Think step by step before answering." },
  ],
});
reply = await askGemini(history);
console.log("AI:", reply);

// 5. Constraints
history.push({
  role: "user",
  parts: [{ text: "Explain useEffect in exactly one sentence." }],
});
reply = await askGemini(history);
console.log("AI:", reply);
