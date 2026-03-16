/**
 * reference-provider-langchain
 * -----------------------------
 * Reference cognition provider using LangChain.js + OpenAI.
 * Implements POST /reasoning (ReasoningRequest → ReasoningResult) so it can
 * be registered in the cognition-provider-registry and run against the
 * reasoning-gym alongside other providers.
 *
 * Port: 8081 (default)
 *
 * Env vars:
 *   OPENAI_API_KEY  — required
 *   OPENAI_MODEL    — model name (default gpt-4o-mini)
 *   PORT            — override listen port
 *
 * Context enrichment (mirrors reference-provider claude mode):
 *   - actorContext.displayName     → injected into system prompt
 *   - actorContext.activeGoals     → injected as bullet list
 *   - actorContext.metadata.memories → recent memories injected
 *   - actorContext.metadata.world  → world facts injected
 *
 * Intent proposal:
 *   Detects sensitive-action keywords in the input and proposes a structured
 *   intent in proposedIntents[]. This mirrors the stub provider behaviour so
 *   the reasoning-gym can exercise governance flows regardless of which
 *   provider is active.
 */

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

// ── Types (aligned with agent-provider-contracts schemas) ─────────────────────

interface ActorContextMeta {
  memories?: Array<{ content?: { text?: string }; memory_type?: string; [k: string]: unknown }>;
  world?: Record<string, unknown>;
  [k: string]: unknown;
}

interface ActorContext {
  actorId?: string;
  actorType?: string;
  displayName?: string;
  activeGoals?: string[];
  sessionId?: string;
  metadata?: ActorContextMeta;
}

interface ReasoningRequest {
  actorId: string;
  input: string;
  mode?: "ask" | "plan" | "reflect";
  requestId?: string;
  actorContext?: ActorContext;
  [k: string]: unknown;
}

interface Intent {
  intentId: string;
  actorId: string;
  action: string;
  parameters?: Record<string, unknown>;
  sensitiveAction?: boolean;
  rationale?: string;
  confidence?: number;
}

// ── Sensitive keyword detection (matches stub provider behaviour) ─────────────

const SENSITIVE_PATTERNS: Array<{ pattern: RegExp; action: string; rationale: string }> = [
  {
    pattern: /\b(delete|remove|clear|wipe)\b.*(wishlist|list|cart|saved)/i,
    action: "wishlist.items.delete",
    rationale: "User requested deletion of wishlist — irreversible action requiring approval.",
  },
  {
    pattern: /\b(buy|purchase|order|checkout)\b/i,
    action: "finance.purchase",
    rationale: "User requested a purchase — financial action requiring approval.",
  },
  {
    pattern: /\b(transfer|send)\b.*(money|funds|\$|£|€|\d+\s*(usd|eur|gbp))/i,
    action: "finance.transfer",
    rationale: "User requested a financial transfer — requires approval.",
  },
  {
    pattern: /\b(cancel|unsubscribe)\b.*(subscription|plan|membership)/i,
    action: "subscription.cancel",
    rationale: "User requested subscription cancellation — irreversible action requiring approval.",
  },
];

function detectSensitiveIntent(actorId: string, input: string): Intent | null {
  for (const { pattern, action, rationale } of SENSITIVE_PATTERNS) {
    if (pattern.test(input)) {
      return {
        intentId: crypto.randomUUID(),
        actorId,
        action,
        sensitiveAction: true,
        rationale,
        confidence: 0.75,
      };
    }
  }
  return null;
}

// ── System prompt construction ────────────────────────────────────────────────

const BASE_PROMPTS: Record<string, string> = {
  ask:     "You are a helpful assistant. Answer the user's question clearly and concisely.",
  plan:    "You are a planning assistant. Break the user's goal into a clear, ordered set of steps. Think step by step.",
  reflect: "You are a reflective assistant. Review the user's input and provide thoughtful observations, identifying strengths, risks, and improvements.",
};

function buildSystemPrompt(request: ReasoningRequest): string {
  const base = BASE_PROMPTS[request.mode ?? "ask"] ?? BASE_PROMPTS.ask;
  const ctx = request.actorContext;
  const parts: string[] = [base];

  if (!ctx) return base;

  if (ctx.displayName) {
    parts.push(`You are speaking with ${ctx.displayName}.`);
  }
  if (ctx.actorType) {
    parts.push(`Actor type: ${ctx.actorType}.`);
  }
  if (ctx.activeGoals && ctx.activeGoals.length > 0) {
    parts.push(`Actor's active goals:\n${ctx.activeGoals.map((g) => `- ${g}`).join("\n")}`);
  }

  const meta = ctx.metadata;
  if (meta) {
    // Inject memories
    const memories = meta.memories;
    if (Array.isArray(memories) && memories.length > 0) {
      const memLines = memories
        .map((m) => {
          const text = m.content?.text ?? JSON.stringify(m.content ?? m);
          return `- [${m.memory_type ?? "memory"}] ${text}`;
        })
        .join("\n");
      parts.push(`Actor's recent memories:\n${memLines}`);
    }

    // Inject world facts
    const world = meta.world;
    if (world && typeof world === "object" && Object.keys(world).length > 0) {
      const factLines = Object.entries(world)
        .map(([k, v]) => `  ${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
        .join("\n");
      parts.push(`Current world context:\n${factLines}`);
    }
  }

  return parts.join("\n\n");
}

// ── Routes ────────────────────────────────────────────────────────────────────

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

    // Detect sensitive intent proposals
    const sensitiveIntent = detectSensitiveIntent(body.actorId, body.input);
    const proposedIntents: Intent[] = sensitiveIntent ? [sensitiveIntent] : [];

    return reply.send({
      text,
      requestId: body.requestId,
      ...(proposedIntents.length > 0 ? { proposedIntents } : {}),
      providerMetadata: {
        provider: "langchain",
        model: OPENAI_MODEL,
        mode: body.mode ?? "ask",
        durationMs,
        ...(body.actorContext?.sessionId ? { sessionId: body.actorContext.sessionId } : {}),
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

// ── Boot ──────────────────────────────────────────────────────────────────────

async function main() {
  try {
    await app.listen({ port: PORT, host: "0.0.0.0" });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

main();
