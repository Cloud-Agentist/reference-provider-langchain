/**
 * reference-provider-langchain
 * -----------------------------
 * Reference cognition provider using LangChain.js + OpenAI.
 * Accepts PerceptionFrame-based requests with time-sliced multimodal sensory data.
 * Returns motor commands (move, speak, gesture, act) via OpenAI function calling.
 *
 * This is a REQUEST-RESPONSE provider — the platform calls it on a heartbeat.
 *
 * Port: 8081 (default)
 *
 * Env vars:
 *   OPENAI_API_KEY  — required
 *   OPENAI_MODEL    — model name (default gpt-4o-mini)
 *   PORT            — override listen port
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

// ── Types ────────────────────────────────────────────────────────────────────

interface SensoryChannel {
  facultyId: string;
  modality: string;
  payload: { format: string; data: string };
  sources?: Array<{ sourceId: string; distance: number; bearing: number }>;
}

interface SensorySlice {
  sliceId: string;
  actorId: string;
  capturedAt: string;
  durationMs: number;
  channels: SensoryChannel[];
}

interface PerceptionFrame {
  frameId: string;
  actorId: string;
  capturedAt: string;
  slices: SensorySlice[];
  memoryContext?: unknown[];
  selfState?: {
    status?: string;
    currentGoals?: string[];
    pendingActions?: string[];
    faculties?: string[];
  };
  attentionHints?: string[];
}

interface MotorCommand {
  commandType: "move" | "speak" | "gesture" | "act";
  actorId: string;
  move?: { target?: { position?: { x: number; y: number; z: number }; targetActorId?: string; area?: string }; speed?: string };
  speak?: { content?: string; volume?: string; targetActorId?: string };
  gesture?: { type?: string; targetActorId?: string };
  act?: { action?: string; parameters?: Record<string, unknown>; rationale?: string };
}

interface ReasoningRequestBody {
  actorId: string;
  perceptionFrame?: PerceptionFrame;
  input?: string;
  mode?: "ask" | "plan" | "reflect" | "react";
  requestId?: string;
  actorContext?: {
    actorId?: string;
    actorType?: string;
    displayName?: string;
    activeGoals?: string[];
    sessionId?: string;
    metadata?: {
      memories?: Array<{ content?: { text?: string }; memory_type?: string }>;
      world?: Record<string, unknown>;
    };
  };
  availableCapabilities?: Array<{ action: string; sensitivityLevel: string; description: string }>;
}

// ── Perception to prompt ─────────────────────────────────────────────────────

function perceptionToPrompt(frame: PerceptionFrame, directInput?: string): string {
  const parts: string[] = [];

  if (frame.selfState) {
    const s = frame.selfState;
    const lines: string[] = [];
    if (s.status) lines.push(`Status: ${s.status}`);
    if (s.faculties?.length) lines.push(`Faculties: ${s.faculties.join(", ")}`);
    if (s.currentGoals?.length) lines.push(`Goals:\n${s.currentGoals.map(g => `  - ${g}`).join("\n")}`);
    if (lines.length > 0) parts.push(`## Your State\n${lines.join("\n")}`);
  }

  if (frame.attentionHints?.length) {
    parts.push(`## Attention\n${frame.attentionHints.map(h => `- ${h}`).join("\n")}`);
  }

  if (frame.slices.length > 0) {
    parts.push(`## Sensory Input (${frame.slices.length} slices)`);
    for (const slice of frame.slices) {
      const time = new Date(slice.capturedAt).toLocaleTimeString();
      const channelDescs: string[] = [];
      for (const ch of slice.channels) {
        try {
          const data = ch.payload.format === "json" ? JSON.parse(ch.payload.data) : ch.payload.data;
          channelDescs.push(`[${ch.modality}] ${typeof data === "object" ? JSON.stringify(data) : data}`);
        } catch {
          channelDescs.push(`[${ch.modality}] (data)`);
        }
      }
      if (channelDescs.length > 0) parts.push(`### ${time}\n${channelDescs.join("\n")}`);
    }
  }

  if (frame.memoryContext?.length) {
    const memLines = frame.memoryContext.map((m: unknown) => {
      const mem = m as Record<string, unknown>;
      const content = mem.content as Record<string, unknown> | undefined;
      return `- ${content?.text ?? JSON.stringify(content ?? mem)}`;
    });
    parts.push(`## Memories\n${memLines.join("\n")}`);
  }

  if (directInput) {
    parts.push(`## Direct Input\nThe user says: "${directInput}"`);
  }

  return parts.join("\n\n");
}

// ── System prompt ────────────────────────────────────────────────────────────

const BASE_SYSTEM_PROMPT =
  "You are an embodied agent in a 3D virtual world. You perceive the world through " +
  "time-sliced sensory data and can act through motor commands.\n\n" +
  "When you want to act, describe your intended actions in your response using " +
  "this exact format (one per line):\n" +
  "  [MOVE] area=commons\n" +
  "  [SPEAK] content=Hello there! volume=normal\n" +
  "  [GESTURE] type=wave\n" +
  "  [ACT] action=calendar.event.create parameters={\"title\":\"Meeting\"} rationale=User requested a meeting\n\n" +
  "Always provide natural language text alongside any action commands.\n\n";

const MODE_PROMPTS: Record<string, string> = {
  ask: "The user is asking a direct question. Answer concisely.",
  plan: "Produce a step-by-step action plan.",
  reflect: "Reflect on the situation with observations and recommendations.",
  react: "React autonomously to your sensory input. If nothing interesting, observe briefly.",
};

function buildSystemPrompt(body: ReasoningRequestBody): string {
  const mode = body.mode ?? "ask";
  let prompt = BASE_SYSTEM_PROMPT + (MODE_PROMPTS[mode] ?? MODE_PROMPTS.ask);

  const ctx = body.actorContext;
  if (ctx?.displayName) prompt += `\n\nYou are speaking with ${ctx.displayName}.`;
  if (ctx?.activeGoals?.length) prompt += `\n\nGoals:\n${ctx.activeGoals.map(g => `- ${g}`).join("\n")}`;

  if (ctx?.metadata?.memories?.length) {
    const memLines = ctx.metadata.memories.map(m => `- [${m.memory_type ?? "memory"}] ${m.content?.text ?? JSON.stringify(m.content)}`);
    prompt += `\n\nMemories:\n${memLines.join("\n")}`;
  }

  if (body.availableCapabilities?.length) {
    const capLines = body.availableCapabilities.map(c => `- ${c.action} [${c.sensitivityLevel}]: ${c.description}`);
    prompt += `\n\nAvailable actions:\n${capLines.join("\n")}`;
  }

  return prompt;
}

// ── Parse motor commands from text ───────────────────────────────────────────

function parseMotorCommands(text: string, actorId: string): { commands: MotorCommand[]; cleanText: string } {
  const commands: MotorCommand[] = [];
  const lines = text.split("\n");
  const cleanLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();

    const moveMatch = trimmed.match(/^\[MOVE\]\s*(.*)/i);
    if (moveMatch) {
      const params = parseParams(moveMatch[1]);
      commands.push({
        commandType: "move",
        actorId,
        move: {
          target: {
            ...(params.area ? { area: params.area } : {}),
            ...(params.targetActorId ? { targetActorId: params.targetActorId } : {}),
          },
          speed: params.speed ?? "walk",
        },
      });
      continue;
    }

    const speakMatch = trimmed.match(/^\[SPEAK\]\s*(.*)/i);
    if (speakMatch) {
      const params = parseParams(speakMatch[1]);
      commands.push({
        commandType: "speak",
        actorId,
        speak: {
          content: params.content ?? "",
          volume: params.volume ?? "normal",
          ...(params.targetActorId ? { targetActorId: params.targetActorId } : {}),
        },
      });
      continue;
    }

    const gestureMatch = trimmed.match(/^\[GESTURE\]\s*(.*)/i);
    if (gestureMatch) {
      const params = parseParams(gestureMatch[1]);
      commands.push({
        commandType: "gesture",
        actorId,
        gesture: { type: params.type ?? "idle" },
      });
      continue;
    }

    const actMatch = trimmed.match(/^\[ACT\]\s*(.*)/i);
    if (actMatch) {
      const params = parseParams(actMatch[1]);
      let parsedParams: Record<string, unknown> = {};
      if (params.parameters) {
        try { parsedParams = JSON.parse(params.parameters); } catch { /* ignore */ }
      }
      commands.push({
        commandType: "act",
        actorId,
        act: {
          action: params.action ?? "",
          parameters: parsedParams,
          rationale: params.rationale ?? "",
        },
      });
      continue;
    }

    cleanLines.push(line);
  }

  return { commands, cleanText: cleanLines.join("\n").trim() };
}

function parseParams(paramStr: string): Record<string, string> {
  const params: Record<string, string> = {};
  // Match key=value pairs, where value can be a JSON object {...} or unquoted string
  const regex = /(\w+)=(\{[^}]*\}|[^\s]+)/g;
  let match;
  while ((match = regex.exec(paramStr)) !== null) {
    params[match[1]] = match[2];
  }
  return params;
}

// ── Routes ───────────────────────────────────────────────────────────────────

app.get("/health", async (_request, reply) => {
  return reply.send({ ok: true, service: "reference-provider-langchain", model: OPENAI_MODEL, interactionPattern: "request-response" });
});

app.post<{ Body: ReasoningRequestBody }>("/reasoning", async (request, reply) => {
  const body = request.body;

  if (!body.actorId) {
    return reply.status(400).send({ error: "actorId is required" });
  }
  if (!body.input && !body.perceptionFrame) {
    return reply.status(400).send({ error: "input or perceptionFrame is required" });
  }

  // Build user message from perception frame and/or direct input
  let userMessage: string;
  if (body.perceptionFrame) {
    userMessage = perceptionToPrompt(body.perceptionFrame, body.input);
  } else {
    userMessage = body.input!;
  }

  const systemPrompt = buildSystemPrompt(body);

  const model = new ChatOpenAI({ model: OPENAI_MODEL, temperature: 0.7 });

  const start = Date.now();
  try {
    const response = await model.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userMessage),
    ]);

    const rawText = typeof response.content === "string"
      ? response.content
      : JSON.stringify(response.content);

    const durationMs = Date.now() - start;

    // Parse motor commands from text output
    const { commands, cleanText } = parseMotorCommands(rawText, body.actorId);

    return reply.send({
      text: cleanText,
      requestId: body.requestId,
      ...(commands.length > 0 ? { motorCommands: commands } : {}),
      providerMetadata: {
        provider: "langchain",
        model: OPENAI_MODEL,
        mode: body.mode ?? "ask",
        durationMs,
        motorCommandCount: commands.length,
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

// ── Boot ─────────────────────────────────────────────────────────────────────

async function main() {
  try {
    await app.listen({ port: PORT, host: "0.0.0.0" });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

main();
