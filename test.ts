import ollama from "npm:ollama"
import embeddings from "./embeddings.json" with { type: "json" }
import { cosine } from "./utils.ts"

const state = {
  step: "start",
  context: "",
  messages: [{ role: "user", content: "How to use drizzle?" }],
}

const start = async () => {
  const embedding = await ollama.embed({
    model: "mxbai-embed-large",
    input: state.messages[0].content,
  })

  let max = 0
  let doc = ""

  for (const { embedding: e, source } of embeddings) {
    const sim = cosine(embedding.embeddings[0], e)
    console.log(sim)

    if (sim > max) {
      max = sim
      doc = source
    }
  }

  state.context = doc
  state.step = "generate"
}

const generate = async () => {
  // const response = await ollama.chat({
  //   model: "llama3.2",
  //   messages: state.messages,
  // })
}

while (state.step !== "end") {
  if (state.step === "start") {
    await start()
  } else if (state.step === "generate") {
    await generate()
  }
}

console.log(state)
