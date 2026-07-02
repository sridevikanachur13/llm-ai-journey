import "dotenv/config";

async function askGeminiWithPersona(conversationHistory, systemInstruction) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemInstruction }] },
        contents: conversationHistory,
      }),
    },
  );
  const data = await response.json();
  return data.candidates[0].content.parts[0].text;
}

let history2 = [{ role: "user", parts: [{ text: "How do I center a div?" }] }];

let persona =
  "You are a sarcastic senior developer who makes fun of basic CSS questions but still gives the correct answer.";

let reply = await askGeminiWithPersona(history2, persona);
console.log(reply);
