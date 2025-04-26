import { ChatOllama } from "npm:@langchain/ollama"

const model = new ChatOllama({ model: "deepseek-r1:1.5b" })

import { StructuredOutputParser } from "npm:@langchain/core/output_parsers"
import { ChatPromptTemplate } from "npm:@langchain/core/prompts"
import { RunnableSequence } from "npm:@langchain/core/runnables"
import { z } from "npm:zod"

const zodSchema = z.object({
  answer: z.string().describe("answer to the user's question"),
  source: z
    .string()
    .describe(
      "source used to answer the user's question, should be a website."
    ),
})

const parser = StructuredOutputParser.fromZodSchema(zodSchema)

const chain = RunnableSequence.from([
  ChatPromptTemplate.fromTemplate(
    "Answer the users question as best as possible.\n{format_instructions}\n{question}"
  ),
  model,
  parser,
])

const response = await chain.invoke({
  question: "What is a Deno?",
  format_instructions: parser.getFormatInstructions(),
})

console.log(response)

import { OllamaEmbeddings } from "npm:@langchain/ollama"

const embeddings = new OllamaEmbeddings({
  model: "mxbai-embed-large",
})

import { CheerioWebBaseLoader } from "npm:@langchain/community/document_loaders/web/cheerio"
import "npm:cheerio"

const urls = [
  "https://deno.com/blog/not-using-npm-specifiers-doing-it-wrong",
  "https://deno.com/blog/v2.1",
  "https://deno.com/blog/build-database-app-drizzle",
]

const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url).load())
)
const docsList = docs.flat()

import { RecursiveCharacterTextSplitter } from "npm:@langchain/textsplitters"

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
})
const allSplits = await splitter.splitDocuments(docsList)
console.log(`Split blog posts into ${allSplits.length} sub-documents.`)

import { MemoryVectorStore } from "npm:langchain/vectorstores/memory"

const vectorStore = await MemoryVectorStore.fromDocuments(allSplits, embeddings)

const retriever = vectorStore.asRetriever()

import { BaseMessage } from "npm:@langchain/core/messages"
import { Annotation } from "npm:@langchain/langgraph"

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
})

import { ToolNode } from "npm:@langchain/langgraph/prebuilt"
import { createRetrieverTool } from "npm:langchain/tools/retriever"

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Deno from various blog posts.",
})
const tools = [tool]

const toolNode = new ToolNode<typeof GraphState.State>(tools)

import { isAIMessage, isToolMessage } from "npm:@langchain/core/messages"
import { END } from "npm:@langchain/langgraph"

function shouldRetrieve(state: typeof GraphState.State): string {
  console.log("---DECIDE TO RETRIEVE---")
  const { messages } = state
  const lastMessage = messages[messages.length - 1]

  if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
    console.log("---DECISION: RETRIEVE---")
    return "retrieve"
  }

  return END
}

async function gradeDocuments(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GET RELEVANCE---")

  const tool = {
    name: "give_relevance_score",
    description: "Give a relevance score to the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  }

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  
  -------

  {context} 
  
  -------

  Here is the user question: {question}

  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`
  )

  const model = new ChatOllama({
    model: "llama3.2:3b",
    temperature: 0,
  }).bindTools([tool])

  const { messages } = state
  const firstMessage = messages[0]
  const lastMessage = messages[messages.length - 1]

  const chain = prompt.pipe(model)

  const score = await chain.invoke({
    question: firstMessage.content as string,
    context: lastMessage.content as string,
  })

  return {
    messages: [score],
  }
}

function checkRelevance(state: typeof GraphState.State): "yes" | "no" {
  console.log("---CHECK RELEVANCE---")

  const { messages } = state
  const lastMessage = messages[messages.length - 1]
  if (!isAIMessage(lastMessage)) {
    throw new Error(
      "The 'checkRelevance' node requires the most recent message to be an AI message."
    )
  }

  const { tool_calls: toolCalls } = lastMessage
  if (!toolCalls || !toolCalls.length) {
    throw new Error(
      "The 'checkRelevance' node requires the most recent message to contain tool calls."
    )
  }

  if (toolCalls[0].args.binaryScore === "yes") {
    console.log("---DECISION: DOCS RELEVANT---")
    return "yes"
  }
  console.log("---DECISION: DOCS NOT RELEVANT---")
  return "no"
}

async function agent(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CALL AGENT---")

  const { messages } = state
  const filteredMessages = messages.filter((message) => {
    if (isAIMessage(message) && message.tool_calls?.length) {
      return message.tool_calls[0].name !== "give_relevance_score"
    }
    return true
  })

  const model = new ChatOllama({
    model: "llama3.2:3b",
    temperature: 0,
    streaming: true,
  }).bindTools(tools)

  const response = await model.invoke(filteredMessages)
  return {
    messages: [response],
  }
}

async function rewrite(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---TRANSFORM QUERY---")

  const { messages } = state
  const question = messages[0].content as string
  const prompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:

-------

{question} 

-------

Formulate an improved question:`
  )

  // Grader
  const model = new ChatOllama({
    model: "deepseek-r1:1.5b",
    temperature: 0,
    streaming: true,
  })
  const response = await prompt.pipe(model).invoke({ question })
  return {
    messages: [response],
  }
}

async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---")

  const { messages } = state
  const question = messages[0].content as string
  // Extract the most recent ToolMessage
  const lastToolMessage = messages
    .slice()
    .reverse()
    .find((msg) => isToolMessage(msg))
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history")
  }

  const context = lastToolMessage.content as string

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Here is the initial question:

-------

{question}

-------

Here is the context that you should use to answer the question:

-------

{context}

-------

Answer:`
  )

  const llm = new ChatOllama({
    model: "deepseek-r1:1.5b",
    temperature: 0,
    streaming: true,
  })

  const ragChain = prompt.pipe(llm)

  const response = await ragChain.invoke({
    context,
    question,
  })

  return {
    messages: [response],
  }
}

import { StateGraph } from "npm:@langchain/langgraph"

const workflow = new StateGraph(GraphState)
  .addNode("agent", agent)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)

import { START } from "npm:@langchain/langgraph"

// Call agent node to decide to retrieve or not
workflow.addEdge(START, "agent")

// Decide whether to retrieve
workflow.addConditionalEdges(
  "agent",
  // Assess agent decision
  shouldRetrieve
)

workflow.addEdge("retrieve", "gradeDocuments")

// Edges taken after the `action` node is called.
workflow.addConditionalEdges(
  "gradeDocuments",
  // Assess agent decision
  checkRelevance,
  {
    // Call tool node
    yes: "generate",
    no: "rewrite", // placeholder
  }
)

workflow.addEdge("generate", END)
workflow.addEdge("rewrite", "agent")

// Compile
const app = workflow.compile()

import { HumanMessage } from "npm:@langchain/core/messages"

const inputs = {
  messages: [new HumanMessage("What are some new features of Deno 2.1?")],
}

let finalState
for await (const output of await app.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    console.log(`${key} -->`)
    finalState = value
  }
}

const lastMessage = finalState.messages[finalState.messages.length - 1]
const content = lastMessage.content
  .replace("<think>", "<details><summary>Thinking...</summary>")
  .replace("</think>", "</details>")

console.log(content)
