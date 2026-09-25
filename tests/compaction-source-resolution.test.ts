import { expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ChatGptAccountSafety } from "../src/adapters/chatgpt-web/account-safety";
import type { BrowserTurn } from "../src/adapters/chatgpt-web/browser-worker";
import { rememberCompactionContinuation } from "../src/adapters/chatgpt-web/compaction-continuation";
import { extractChatGptCompactionSourceRevision } from "../src/adapters/chatgpt-web/environment";
import { chatGptWebExecutionNamespace, createChatGptWebAdapter } from "../src/adapters/chatgpt-web/index";
import { deferred } from "../src/adapters/chatgpt-web/runtime-lifecycle";
import { callTurnBroker, TurnBroker } from "../src/adapters/chatgpt-web/turn-broker";
import {
  ChatGptTextFeed,
  ChatGptTraceFeed,
  ChatGptTurnSessions,
  chatGptCompactionNativeTurnExecutionKey,
  chatGptCompactionSourceExecutionKey,
  chatGptConversationKey,
  chatGptTurnExecutionKey,
  chatGptTurnSessions,
} from "../src/adapters/chatgpt-web/turn-execution";
import { defaultBrokerEndpoint } from "../src/config";
import { encodeCompactionSummary } from "../src/responses/compaction";
import type { AdapterEvent, CodexParsedRequest, CodexProviderConfig } from "../src/types";

function register(sessions: ChatGptTurnSessions, key: string, conversationKey: string, active: boolean) {
  const browser = deferred<string>();
  if (!active) browser.resolve("settled answer");
  return sessions.getOrCreate(key, () => ({
    mode: "read-only",
    browser: browser.promise,
    trace: new ChatGptTraceFeed(),
    text: new ChatGptTextFeed(),
    conversationKey,
    cancel: () => browser.resolve("cancelled"),
  }));
}

test("a compaction replaces the browser turn still running in its conversation or native turn", async () => {
  const sessions = new ChatGptTurnSessions();
  const keys = { history: "history", nativeTurn: "native-turn", conversation: "conversation" };
  try {
    expect(sessions.compactionSourceKey(keys)).toBe("history");
    // Before a new turn samples, the settled answer of the history's instruction is the source.
    await register(sessions, "history", "conversation", false).browserOutcome;
    expect(sessions.compactionSourceKey(keys)).toBe("history");
    // A browser turn running elsewhere is not this compaction's conversation.
    register(sessions, "elsewhere", "other-conversation", true);
    expect(sessions.compactionSourceKey(keys)).toBe("history");
    // Mid-turn, the running browser turn is the source whichever turn its instruction came from.
    const live = register(sessions, "live", "conversation", true);
    expect(sessions.compactionSourceKey(keys)).toBe("live");
    live.cancel();
    await live.browserOutcome;
    register(sessions, "native-turn", "rotated-conversation", true);
    expect(sessions.compactionSourceKey(keys)).toBe("native-turn");
  } finally {
    sessions.clear();
  }
});

const THREAD = "thread_compaction_continuation";
const ORIGINAL_TURN = "turn_original_instruction";
const LIVE_TURN = "turn_continued_after_compaction";
const EARLIER_CHECKPOINT = "Earlier checkpoint of the original task.";

/** A turn that already compacted once: its latest instruction belongs to an older native turn. */
function continuedTurn(compaction: boolean): CodexParsedRequest {
  return {
    modelId: "gpt-5.6-sol",
    stream: true,
    options: { reasoning: "high" },
    context: { messages: [{ role: "user", content: "Original task", timestamp: 1 }] },
    ...(compaction ? { _compactionRequest: true } : {}),
    _rawBody: {
      prompt_cache_key: THREAD,
      client_metadata: { "x-codex-turn-metadata": JSON.stringify({ thread_id: THREAD, turn_id: LIVE_TURN }) },
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Original task" }],
          internal_chat_message_metadata_passthrough: { turn_id: ORIGINAL_TURN },
        },
        { type: "compaction", encrypted_content: encodeCompactionSummary(EARLIER_CHECKPOINT) },
      ],
    },
  };
}

test("a second mid-turn compaction hands off from the live conversation instead of a fresh chat", async () => {
  const root = mkdtempSync(join(process.platform === "win32" ? tmpdir() : "/tmp", "cgw-continued-compact-"));
  const socketPath = defaultBrokerEndpoint(root);
  const provider: CodexProviderConfig = {
    adapter: "chatgpt-web",
    baseUrl: `browser://continued-compact-${process.pid}-${Date.now()}`,
    chatgptWeb: {
      useEnhancedWebSessionMode: true,
      browserHost: "launcher",
      browserHostDescriptorPath: join(root, "launcher.json"),
      brokerSocketPath: socketPath,
      localToolsEnabled: true,
      solAvailable: true,
      extraHighAvailable: true,
      proAvailable: true,
    },
  };
  const namespace = chatGptWebExecutionNamespace(provider);
  const compact = continuedTurn(true);
  // The earlier compaction bound the original instruction to this continuing native turn.
  rememberCompactionContinuation(compact, { threadId: THREAD, turnId: LIVE_TURN },
    [extractChatGptCompactionSourceRevision(compact)], EARLIER_CHECKPOINT);
  const live = continuedTurn(false);
  const liveKey = `${namespace}:${chatGptTurnExecutionKey(live)}`;
  const liveConversation = chatGptConversationKey(live, namespace)!;
  // The history's instruction keys no live session; the compaction's own native turn does.
  expect(`${namespace}:${chatGptCompactionSourceExecutionKey(compact)}`).not.toBe(liveKey);
  expect(`${namespace}:${chatGptCompactionNativeTurnExecutionKey(compact)}`).toBe(liveKey);
  const liveBrowser = deferred<string>();
  const source = chatGptTurnSessions.getOrCreate(liveKey, () => ({
    mode: "read-only",
    browser: liveBrowser.promise,
    trace: new ChatGptTraceFeed(),
    text: new ChatGptTextFeed(),
    usageInput: live,
    nativeIdentity: { threadId: THREAD, turnId: LIVE_TURN },
    conversationKey: liveConversation,
    cancel: () => liveBrowser.resolve("cancelled"),
  }));
  const turns: BrowserTurn[] = [];
  const worker = {
    async run(turn: BrowserTurn) {
      turns.push(turn);
      const prepared = await turn.prepare();
      try {
        if (!turn.requireRetainedConversation) return "Fresh-chat checkpoint.";
        await callTurnBroker(socketPath, {
          method: "submit_compaction_handoff",
          token: /turn_token (control_\w+)/.exec(prepared.text)![1]!,
          handoffId: /handoff_id (handoff_\w+)/.exec(prepared.text)![1]!,
          summary: "Checkpoint written by the live conversation.",
        });
        return "Checkpoint submitted.";
      } finally { prepared.release(); }
    },
  };
  const warnings: string[] = [];
  const warn = spyOn(console, "warn").mockImplementation(message => { warnings.push(String(message)); });
  const events: AdapterEvent[] = [];
  let running: Promise<void> | undefined;
  try {
    running = createChatGptWebAdapter(provider, {
      worker: worker as never,
      accountSafety: new ChatGptAccountSafety(join(root, "account-safety.json")),
    }).runTurn!(compact, { headers: new Headers() }, event => events.push(event));
    await Bun.sleep(10);
    expect(turns).toHaveLength(0);
    liveBrowser.resolve("Live answer.");
    await running;

    expect(turns).toHaveLength(1);
    expect(turns[0]!.requireRetainedConversation).toBeTrue();
    expect(turns[0]!.conversationKey).toBe(liveConversation);
    expect(warnings.filter(line => line.includes("retained compaction"))).toEqual([]);
    expect(events.some(event => event.type === "text_delta"
      && event.text.includes("Checkpoint written by the live conversation."))).toBeTrue();
    expect(events.at(-1)).toMatchObject({ type: "done", endTurn: true });
    expect(source.conversationKey()).toBeUndefined();
  } finally {
    warn.mockRestore();
    liveBrowser.resolve("cleanup");
    await running?.catch(() => {});
    chatGptTurnSessions.clear();
    await TurnBroker.forSocket(socketPath).close();
    rmSync(root, { recursive: true, force: true });
  }
});
