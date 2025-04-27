import ollama from "npm:ollama"
import embeddings from "./embeddings.json" with { type: "json" }
import { cosine } from "./utils.ts"

const state = {
  step: "start",
  messages: [
    {
      role: "system",
      content:
        "You are an expert answering questions based only on the following document: ",
    },
    {
      role: "user",
      content: "How to use drizzle?",
    },
  ],
}

const start = async () => {
  const embedding = await ollama.embed({
    model: "mxbai-embed-large",
    input: state.messages[0].content,
  })

  let max = 0
  let context = ""

  for (const { embedding: doc, text } of embeddings) {
    const sim = cosine(embedding.embeddings[0], doc)

    if (sim > max) {
      max = sim
      context = text
    }
  }

  state.messages[0].content += context
  state.step = "generate"
}

const generate = async () => {
  const response = await ollama.chat({
    model: "llama3.2",
    messages: state.messages,
  })

  state.messages.push({
    role: "assistant",
    content: response.message.content,
  })

  state.step = "end"
}

while (state.step !== "end") {
  if (state.step === "start") {
    await start()
  } else if (state.step === "generate") {
    await generate()
  }
}

console.log(state.messages[state.messages.length - 1].content)
