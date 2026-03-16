import Fastify from "fastify";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const PORT = Number(process.env.PORT) || 8081;
const OPENAI_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is required");
  process.exit(1);
}

const app = Fastify({ logger: true });

// ReasoningRequest shape (subset of contract — full schema in agent-provider-contracts)
interface ReasoningRequest {
  actorId: string;
  input: string;
  mode?: "ask" | "plan" | "reflect";
  requestId?: string;
  actorContext?: {
    actorType?: string;
    displayName?: string;
    activeGoals?: string[];
    metadata?: Record<string, unknown>;
  };
}

const SYSTEM_PROMPTS: Record<string, string> = {
  ask: "You are a helpful assistant. Answer the user's question clearly and concisely.",
  plan: "You are a planning assistant. Break the user's goal into a clear, ordered set of steps. Think step by step.",
  reflect: "You are a reflective assistant. Review the user's input and provide thoughtful observations, identifying strengths, risks, and improvements.",
};

function buildSystemPrompt(request: ReasoningRequest): string {
  const base = SYSTEM_PROMPTS[request.mode ?? "ask"] ?? SYSTEM_PROMPTS.ask;

  const ctx = request.actorContext;
  if (!ctx) return base;

  const parts: string[] = [base];

  if (ctx.displayName) {
    parts.push(`You are speaking with ${ctx.displayName}.`);
  }
  if (ctx.actorType) {
    parts.push(`Actor type: ${ctx.actorType}.`);
  }
  if (ctx.activeGoals && ctx.activeGoals.length > 0) {
    parts.push(`Actor's active goals:\n${ctx.activeGoals.map((g) => `- ${g}`).join("\n")}`);
  }

  return parts.join("\n\n");
}

app.get("/health", async (_request, reply) => {
  return reply.send({ ok: true, service: "reference-provider-langchain", model: OPENAI_MODEL });
});

app.post<{ Body: ReasoningRequest }>("/reasoning", async (request, reply) => {
  const body = request.body;

  if (!body.actorId || typeof body.actorId !== "string") {
    return reply.status(400).send({ error: "actorId is required" });
  }
  if (!body.input || typeof body.input !== "string") {
    return reply.status(400).send({ error: "input is required" });
  }

  const systemPrompt = buildSystemPrompt(body);

  const model = new ChatOpenAI({
    model: OPENAI_MODEL,
    temperature: 0.7,
  });

  const start = Date.now();
  try {
    const response = await model.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(body.input),
    ]);

    const text = typeof response.content === "string"
      ? response.content
      : JSON.stringify(response.content);

    const durationMs = Date.now() - start;

    return reply.send({
      text,
      requestId: body.requestId,
      providerMetadata: {
        provider: "langchain",
        model: OPENAI_MODEL,
        mode: body.mode ?? "ask",
        durationMs,
      },
    });
  } catch (err) {
    request.log.error({ err }, "LangChain inference failed");
    return reply.status(502).send({
      error: "Inference failed",
      detail: err instanceof Error ? err.message : String(err),
    });
  }
});

async function main() {
  try {
    await app.listen({ port: PORT, host: "0.0.0.0" });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

main();
