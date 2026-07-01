import "dotenv/config";

const response = await fetch(
  `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
  {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            {
              text: "Explain what an API is, in 2 sentences, to a frontend developer.",
            },
          ],
        },
      ],
    }),
  },
);

const data = await response.json();
console.log(data.candidates[0].content.parts[0].text);
