import { load } from "npm:cheerio"
import ollama from "npm:ollama"

const urls = [
  "https://deno.com/blog/not-using-npm-specifiers-doing-it-wrong",
  "https://deno.com/blog/build-database-app-drizzle",
  "https://deno.com/blog/v2.1",
]

const embeddings = await Promise.all(
  urls.map(async (url) => {
    const res = await fetch(url)
    const html = await res.text()

    const $ = load(html)
    const content = $("body").text()

    const { embeddings } = await ollama.embed({
      model: "mxbai-embed-large",
      input: content,
    })

    return { embedding: embeddings[0], text: content, source: url }
  }),
)

await Deno.writeTextFile("embeddings.json", JSON.stringify(embeddings))
