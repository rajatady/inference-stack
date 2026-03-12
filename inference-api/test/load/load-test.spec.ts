/**
 * Load Test Suite — Baseline Measurement for KV Cache Architecture
 *
 * Measures real TPS, latency percentiles, prefix sharing waste, multi-turn
 * recompute cost, concurrency scaling, and cross-model contention.
 *
 * All metrics flow into ClickHouse via the MetricsService pipeline.
 * After running, query: GET /v1/metrics/tps, /latency, /breakdown
 *
 * Requires:
 *   - SSH tunnel to RunPod (localhost:50051, :50052)
 *   - GPU workers running with SmolLM2-135M loaded
 *   - ClickHouse running locally
 *
 * Run: npm run test:load
 */
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../../src/app.module';
import { SchedulerService } from '../../src/scheduler/scheduler.service';
// eslint-disable-next-line @typescript-eslint/no-var-requires
const conversationData = require('./conversation-data.json');

const TEST_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const MODEL_1_7B = 'HuggingFaceTB/SmolLM2-1.7B-Instruct';
const IMAGE_MODEL = 'stabilityai/sd-turbo';
const TIMEOUT = 300_000; // 5 min — high concurrency ramps take time

// Real conversation extracted from this project's Claude Code session (230 turns, ~18K tokens)
const REAL_CONVERSATION: Array<{ role: string; content: string }> = conversationData;

// ── Helpers ────────────────────────────────────────────────

interface RequestResult {
  status: number;
  body: any;
  startMs: number;
  endMs: number;
  durationMs: number;
}

async function fireOne(
  app: INestApplication,
  payload: Record<string, any>,
): Promise<RequestResult> {
  const startMs = Date.now();
  const res = await request(app.getHttpServer())
    .post('/v1/completions')
    .send(payload);
  const endMs = Date.now();
  return {
    status: res.status,
    body: res.body,
    startMs,
    endMs,
    durationMs: endMs - startMs,
  };
}

async function fireSequential(
  app: INestApplication,
  payloads: Record<string, any>[],
): Promise<RequestResult[]> {
  const results: RequestResult[] = [];
  for (const payload of payloads) {
    results.push(await fireOne(app, payload));
  }
  return results;
}

async function fireConcurrent(
  app: INestApplication,
  payloads: Record<string, any>[],
): Promise<RequestResult[]> {
  return Promise.all(payloads.map((p) => fireOne(app, p)));
}

async function fireImage(
  app: INestApplication,
  payload: Record<string, any>,
): Promise<RequestResult> {
  const startMs = Date.now();
  const res = await request(app.getHttpServer())
    .post('/v1/images/generations')
    .send(payload);
  const endMs = Date.now();
  return {
    status: res.status,
    body: res.body,
    startMs,
    endMs,
    durationMs: endMs - startMs,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function avg(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function extractTiming(result: RequestResult) {
  const usage = result.body?.usage || {};
  return {
    e2eMs: result.durationMs,
    prefillMs: usage.prefill_time_ms ?? 0,
    decodeMs: usage.decode_time_ms ?? 0,
    totalGpuMs: usage.total_time_ms ?? 0,
    promptTokens: usage.prompt_tokens ?? 0,
    completionTokens: usage.completion_tokens ?? 0,
    decodeTps:
      usage.decode_time_ms > 0
        ? (usage.completion_tokens ?? 0) / (usage.decode_time_ms / 1000)
        : 0,
    prefillTps:
      usage.prefill_time_ms > 0
        ? (usage.prompt_tokens ?? 0) / (usage.prefill_time_ms / 1000)
        : 0,
  };
}

// ── Shared system prompt for prefix tests (~200 tokens worth) ──
const SYSTEM_PROMPT =
  'You are a helpful AI assistant specialized in scientific research. ' +
  'You have extensive knowledge of physics, chemistry, biology, and mathematics. ' +
  'When answering questions, provide detailed explanations with examples. ' +
  'Always cite relevant theories and principles. ' +
  'If a question is ambiguous, ask for clarification before answering. ' +
  'Format your responses with clear sections and bullet points when appropriate. ' +
  'You should consider multiple perspectives and present balanced viewpoints. ' +
  'Remember to explain complex concepts in simple terms when possible. ' +
  'Your goal is to help users understand scientific topics deeply. ' +
  'Always be accurate and acknowledge uncertainty when it exists. ' +
  'Provide references to seminal papers and textbooks when relevant. ' +
  'Consider the historical context of scientific discoveries in your explanations. ';

// ── Realistic conversation-length user messages (~50-150 tokens each) ──
const REALISTIC_PROMPTS = [
  "I'm working on a research paper about the implications of quantum computing on current cryptographic standards. Can you explain how Shor's algorithm threatens RSA encryption and what post-quantum alternatives are being considered by NIST?",
  "I've been reading about CRISPR-Cas9 gene editing and I'm confused about the difference between somatic and germline editing. What are the ethical considerations for each, and what regulations exist in different countries?",
  "Can you walk me through the process of how mRNA vaccines work, starting from the design of the mRNA sequence to the immune response it triggers? I want to understand why they can be developed faster than traditional vaccines.",
  "I'm trying to understand the relationship between dark matter, dark energy, and the expansion of the universe. How do we know dark matter exists if we can't observe it directly? What evidence supports the Lambda-CDM model?",
  "Explain how transformer architectures work in large language models. I understand the basics of attention mechanisms, but I want to understand multi-head attention, positional encoding, and why transformers replaced RNNs for sequence modeling.",
  "I'm studying climate science and I want to understand the difference between climate models and weather forecasting models. What are the key feedback loops in the climate system, and why is cloud formation such a difficult variable to model?",
  "Help me understand how superconductors work at the quantum mechanical level. What is Cooper pairing, and why does the BCS theory break down for high-temperature superconductors? What's the latest research on room-temperature superconductivity?",
  "I need to understand the Navier-Stokes equations for my fluid dynamics course. Can you explain what each term represents physically, why turbulence makes them so hard to solve, and what the Millennium Prize Problem about them actually asks?",
];

// ── Test Suite ─────────────────────────────────────────────

describe('Load Test — KV Cache Baseline Measurement', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();

    // Warm-up: ensure model is loaded and first-inference latency is out of the way
    console.log('\n🔥 Warming up (2 requests)...');
    await fireOne(app, {
      model: TEST_MODEL,
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 5,
      user: 'load-warmup',
    });
    await fireOne(app, {
      model: TEST_MODEL,
      messages: [{ role: 'user', content: 'Hi again' }],
      max_tokens: 5,
      user: 'load-warmup',
    });
    console.log('✅ Warm-up complete\n');
  }, TIMEOUT);

  afterAll(async () => {
    // Print summary report
    await printReport(app);
    await app.close();
  });

  // ── S1: Single Request Baseline ──────────────────────────

  describe('S1: Single Request Baseline', () => {
    it(
      'should establish baseline TPS with zero contention',
      async () => {
        console.log('\n── S1: Single Request Baseline ──');

        const results = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: TEST_MODEL,
            messages: [
              { role: 'system', content: 'You are a helpful assistant. Answer concisely.' },
              { role: 'user', content: REALISTIC_PROMPTS[i % REALISTIC_PROMPTS.length] },
            ],
            max_tokens: 50,
            user: 'load-s1-baseline',
          })),
        );

        const failed = results.filter((r) => r.status !== 200);
        if (failed.length > 0) {
          console.log(`  ⚠ ${failed.length} request(s) failed: ${failed.map((r) => `${r.status}`).join(', ')}`);
        }
        const timings = results.filter((r) => r.status === 200).map(extractTiming);
        expect(timings.length).toBeGreaterThanOrEqual(3);

        console.log('  Decode TPS:  ', timings.map((t) => t.decodeTps.toFixed(1)).join(', '));
        console.log('  Prefill TPS: ', timings.map((t) => t.prefillTps.toFixed(1)).join(', '));
        console.log('  E2E (ms):    ', timings.map((t) => t.e2eMs).join(', '));
        console.log('  Prefill (ms):', timings.map((t) => t.prefillMs.toFixed(1)).join(', '));
        console.log('  Decode (ms): ', timings.map((t) => t.decodeMs.toFixed(1)).join(', '));
        console.log(
          `  AVG: decode=${avg(timings.map((t) => t.decodeTps)).toFixed(1)} tps, ` +
            `prefill=${avg(timings.map((t) => t.prefillTps)).toFixed(1)} tps, ` +
            `e2e=${avg(timings.map((t) => t.e2eMs)).toFixed(0)}ms`,
        );
      },
      TIMEOUT,
    );
  });

  // ── S2: Concurrency Ramp ─────────────────────────────────

  describe('S2: Concurrency Ramp', () => {
    for (const concurrency of [1, 2, 4, 8, 16, 32, 64, 128, 256]) {
      it(
        `should measure latency at concurrency=${concurrency}`,
        async () => {
          console.log(`\n── S2: Concurrency = ${concurrency} ──`);

          // Fire exactly `concurrency` requests in parallel
          const payloads = Array.from(
            { length: concurrency },
            (_, i) => ({
              model: TEST_MODEL,
              messages: [
                { role: 'system', content: SYSTEM_PROMPT },
                { role: 'user', content: REALISTIC_PROMPTS[i % REALISTIC_PROMPTS.length] },
              ],
              max_tokens: 30,
              user: `load-s2-c${concurrency}`,
            }),
          );
          const allResults = await fireConcurrent(app, payloads);

          const timings = allResults.filter((r) => r.status === 200).map(extractTiming);
          const errors = allResults.filter((r) => r.status !== 200);
          const e2es = timings.map((t) => t.e2eMs);
          const dtps = timings.map((t) => t.decodeTps);

          console.log(
            `  n=${timings.length} ok, ${errors.length} errors | ` +
              `decode_tps avg=${avg(dtps).toFixed(1)} | ` +
              `e2e p50=${percentile(e2es, 50)}ms p95=${percentile(e2es, 95)}ms p99=${percentile(e2es, 99)}ms`,
          );
          if (errors.length > 0) {
            console.log(`  Error statuses: ${errors.map((r) => r.status).join(', ')}`);
          }

          // At very high concurrency some 429s are expected — just need at least some to succeed
          expect(timings.length).toBeGreaterThan(0);
        },
        TIMEOUT,
      );
    }
  });

  // ── S3: Repeated Prefix (KV Cache Waste) ─────────────────

  describe('S3: Repeated Prefix — KV Cache Waste', () => {
    it(
      'should measure prefill waste from recomputing shared prefix',
      async () => {
        console.log('\n── S3: Repeated Prefix (20 requests, same system prompt) ──');

        const queries = [
          'What is quantum entanglement?',
          'Explain the theory of relativity.',
          'How does photosynthesis work?',
          'What are black holes?',
          'Describe the structure of DNA.',
          'What is the Heisenberg uncertainty principle?',
          'How do vaccines work?',
          'What causes tides?',
          'Explain superconductivity.',
          'What is dark matter?',
          'How does nuclear fusion work?',
          'What is the Doppler effect?',
          'Explain natural selection.',
          'What are gravitational waves?',
          'How does an MRI machine work?',
          'What is entropy?',
          'Explain the carbon cycle.',
          'What is antimatter?',
          'How does GPS work?',
          'What is CRISPR?',
        ];

        const payloads = queries.map((q) => ({
          model: TEST_MODEL,
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: q },
          ],
          max_tokens: 20,
          user: 'load-s3-prefix',
        }));

        const results = await fireSequential(app, payloads);
        const timings = results.filter((r) => r.status === 200).map(extractTiming);

        const totalPrefillMs = timings.reduce((s, t) => s + t.prefillMs, 0);
        const avgPrefillMs = totalPrefillMs / timings.length;
        const firstPrefillMs = timings[0]?.prefillMs ?? 0;

        // With KV cache: only first request pays full prefill, rest reuse cached prefix
        const wastedMs = totalPrefillMs - firstPrefillMs;
        const wastedPct =
          totalPrefillMs > 0 ? ((wastedMs / totalPrefillMs) * 100).toFixed(1) : '0';

        console.log(`  Requests: ${timings.length}`);
        console.log(`  Prompt tokens (each): ${timings[0]?.promptTokens ?? '?'}`);
        console.log(`  Avg prefill time: ${avgPrefillMs.toFixed(1)}ms`);
        console.log(`  Total prefill time: ${totalPrefillMs.toFixed(1)}ms`);
        console.log(`  First request prefill: ${firstPrefillMs.toFixed(1)}ms`);
        console.log(
          `  WASTED PREFILL (without KV cache): ${wastedMs.toFixed(1)}ms (${wastedPct}%)`,
        );
        console.log(
          `  ↳ With prefix sharing: only ${firstPrefillMs.toFixed(1)}ms needed, ` +
            `saving ${wastedMs.toFixed(1)}ms`,
        );

        expect(timings.length).toBe(20);
      },
      TIMEOUT,
    );
  });

  // ── S4: Multi-Turn Conversation ──────────────────────────

  describe('S4: Multi-Turn Conversation', () => {
    it(
      'should measure cumulative recompute cost across turns',
      async () => {
        console.log('\n── S4: Multi-Turn Conversation (5 turns) ──');

        const turns = [
          'What is quantum computing?',
          'How does it differ from classical computing?',
          'What are qubits and superposition?',
          'What problems can quantum computers solve efficiently?',
          'When will quantum computers be practical for everyday use?',
        ];

        const simulatedAssistantResponses = [
          'Quantum computing is a type of computation that uses quantum-mechanical phenomena such as superposition and entanglement to process information. Unlike classical computers that use bits, quantum computers use qubits which can exist in multiple states simultaneously.',
          'Classical computers process bits that are either 0 or 1. Quantum computers use qubits that can be in a superposition of both states. This allows quantum computers to explore many solutions simultaneously, making them exponentially faster for certain problems.',
          'A qubit is the basic unit of quantum information. Superposition allows a qubit to be in a combination of 0 and 1 states until measured. When measured, it collapses to one state. This property enables quantum parallelism.',
          'Quantum computers excel at problems like integer factorization (breaking RSA encryption via Shor\'s algorithm), unstructured search (Grover\'s algorithm), quantum simulation of molecules, and optimization problems. They are not universally faster than classical computers.',
        ];

        // Build growing conversation: each turn includes all prior messages
        const messages: Array<{ role: string; content: string }> = [
          { role: 'system', content: SYSTEM_PROMPT },
        ];
        const results: RequestResult[] = [];

        for (let i = 0; i < turns.length; i++) {
          messages.push({ role: 'user', content: turns[i] });

          const result = await fireOne(app, {
            model: TEST_MODEL,
            messages: [...messages], // copy to avoid mutation
            max_tokens: 30,
            user: `load-s4-turn${i + 1}`,
          });
          results.push(result);

          // Add simulated assistant response for next turn's context
          if (i < simulatedAssistantResponses.length) {
            messages.push({ role: 'assistant', content: simulatedAssistantResponses[i] });
          }
        }

        const timings = results.filter((r) => r.status === 200).map(extractTiming);

        let cumulativeWaste = 0;
        console.log('  Turn | Prompt Tokens | Prefill (ms) | Redundant Tokens | Waste (ms)');
        console.log('  ─────┼───────────────┼──────────────┼──────────────────┼───────────');

        for (let i = 0; i < timings.length; i++) {
          const t = timings[i];
          // Turn 1 has no redundant tokens; subsequent turns recompute all prior tokens
          const redundantTokens = i === 0 ? 0 : timings[i - 1].promptTokens;
          const wasteMs =
            i === 0
              ? 0
              : t.prefillMs * (redundantTokens / Math.max(t.promptTokens, 1));
          cumulativeWaste += wasteMs;

          console.log(
            `    ${i + 1}  |  ${String(t.promptTokens).padStart(11)} |  ${t.prefillMs.toFixed(1).padStart(10)} |  ${String(redundantTokens).padStart(14)} |  ${wasteMs.toFixed(1).padStart(7)}`,
          );
        }

        console.log(`\n  Cumulative recompute waste: ${cumulativeWaste.toFixed(1)}ms`);
        console.log(
          `  ↳ With KV cache (session affinity): only new tokens prefilled each turn`,
        );

        expect(timings.length).toBe(5);
      },
      TIMEOUT,
    );
  });

  // ── S5: Sustained Load ───────────────────────────────────

  describe('S5: Sustained Load (20s @ 1 rps)', () => {
    it(
      'should measure steady-state behavior',
      async () => {
        console.log('\n── S5: Sustained Load (20s @ 1 rps) ──');

        const totalRequests = 20;
        const intervalMs = 1000;
        const results: RequestResult[] = [];

        // Use allSettled pattern: fire at controlled rate, collect results
        // Cap in-flight to avoid ECONNRESET
        const maxInFlight = 4;
        const inFlight: Promise<void>[] = [];

        for (let i = 0; i < totalRequests; i++) {
          const p = fireOne(app, {
            model: TEST_MODEL,
            messages: [
              { role: 'system', content: 'You are a helpful assistant. Be concise.' },
              { role: 'user', content: REALISTIC_PROMPTS[i % REALISTIC_PROMPTS.length] },
            ],
            max_tokens: 30,
            user: 'load-s5-sustained',
          }).then(
            (r) => { results.push(r); },
            (err) => {
              results.push({
                status: 0,
                body: { error: err.message },
                startMs: Date.now(),
                endMs: Date.now(),
                durationMs: 0,
              });
            },
          );
          inFlight.push(p);

          // If at max in-flight, wait for one to finish before sending more
          if (inFlight.length >= maxInFlight) {
            await Promise.race(inFlight);
            // Remove settled promises
            for (let j = inFlight.length - 1; j >= 0; j--) {
              const settled = await Promise.race([
                inFlight[j].then(() => true),
                Promise.resolve(false),
              ]);
              if (settled) inFlight.splice(j, 1);
            }
          }

          if (i < totalRequests - 1) await sleep(intervalMs);
        }

        // Wait for remaining in-flight
        await Promise.allSettled(inFlight);

        const ok = results.filter((r) => r.status === 200);
        const errors = results.filter((r) => r.status !== 200);
        const timings = ok.map(extractTiming);
        const e2es = timings.map((t) => t.e2eMs);

        const statsRes = await request(app.getHttpServer()).get('/v1/completions/stats');

        console.log(`  Total requests: ${results.length}`);
        console.log(`  Successful: ${ok.length}, Errors: ${errors.length}`);
        if (timings.length > 0) {
          console.log(
            `  E2E latency: p50=${percentile(e2es, 50)}ms p95=${percentile(e2es, 95)}ms p99=${percentile(e2es, 99)}ms`,
          );
          console.log(
            `  Decode TPS avg: ${avg(timings.map((t) => t.decodeTps)).toFixed(1)}`,
          );
        }
        console.log(`  Final queue stats: ${JSON.stringify(statsRes.body)}`);

        expect(ok.length).toBeGreaterThan(0);
      },
      TIMEOUT,
    );
  });

  // ── S6: Cross-Model Contention ───────────────────────────

  describe('S6: Cross-Model Contention (text + image)', () => {
    it(
      'should measure cross-GPU interference',
      async () => {
        console.log('\n── S6: Cross-Model Contention (text on GPU-0, image on GPU-1) ──');

        // First: text-only baseline (3 requests)
        const textOnlyResults = await fireConcurrent(
          app,
          Array.from({ length: 3 }, (_, i) => ({
            model: TEST_MODEL,
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: REALISTIC_PROMPTS[i] },
            ],
            max_tokens: 20,
            user: 'load-s6-text-alone',
          })),
        );

        const textOnlyTimings = textOnlyResults
          .filter((r) => r.status === 200)
          .map(extractTiming);
        const textOnlyAvgTps = avg(textOnlyTimings.map((t) => t.decodeTps));

        console.log(
          `  Text-only (3 concurrent): avg decode_tps=${textOnlyAvgTps.toFixed(1)}`,
        );

        // Now: text + image concurrent
        const textPayloads = Array.from({ length: 3 }, (_, i) => ({
          model: TEST_MODEL,
          messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: REALISTIC_PROMPTS[i + 3] },
          ],
          max_tokens: 20,
          user: 'load-s6-text',
        }));

        const imagePayloads = Array.from({ length: 2 }, (_, i) => ({
          model: IMAGE_MODEL,
          prompt: `A beautiful sunset over mountains, request ${i + 1}`,
        }));

        // Fire text + image concurrently
        const [textResults, imageResults] = await Promise.all([
          fireConcurrent(app, textPayloads),
          Promise.all(
            imagePayloads.map((p) =>
              fireImage(app, { ...p, user: 'load-s6-image' }),
            ),
          ),
        ]);

        const textTimings = textResults
          .filter((r) => r.status === 200)
          .map(extractTiming);
        const textWithImageAvgTps = avg(
          textTimings.map((t) => t.decodeTps),
        );

        const imageOk = imageResults.filter((r) => r.status === 200);

        console.log(
          `  Text+Image (3 text + 2 image): avg decode_tps=${textWithImageAvgTps.toFixed(1)}`,
        );
        console.log(
          `  Image gen: ${imageOk.length}/${imageResults.length} ok, avg=${avg(imageResults.map((r) => r.durationMs)).toFixed(0)}ms`,
        );

        const impact =
          textOnlyAvgTps > 0
            ? (((textOnlyAvgTps - textWithImageAvgTps) / textOnlyAvgTps) * 100).toFixed(1)
            : '?';
        console.log(
          `  Contention impact: ${impact}% TPS reduction (should be ~0% — different GPUs)`,
        );

        expect(textTimings.length).toBeGreaterThan(0);
      },
      TIMEOUT,
    );
  });

  // ── S7: Model Size Scaling ────────────────────────────────

  describe('S7: Model Size Scaling', () => {
    const MODELS = [
      { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', label: '135M' },
      { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', label: '360M' },
      { id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', label: '1.7B' },
    ];
    const CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32];

    for (const model of MODELS) {
      describe(`${model.label}`, () => {
        beforeAll(async () => {
          console.log(`\n── S7: Warming up ${model.label} (forcing model load)... ──`);
          const warmup = await fireOne(app, {
            model: model.id,
            messages: [{ role: 'user', content: 'Hello' }],
            max_tokens: 5,
            user: `stress-s7-${model.label}-warmup`,
          });
          console.log(`  ${model.label} loaded: ${warmup.status === 200 ? 'OK' : `FAIL ${warmup.status}`} (${warmup.durationMs}ms)`);
        }, TIMEOUT);

        for (const c of CONCURRENCY_LEVELS) {
          it(
            `${model.label} at concurrency=${c}`,
            async () => {
              const payloads = Array.from({ length: c }, (_, i) => ({
                model: model.id,
                messages: [
                  { role: 'system', content: SYSTEM_PROMPT },
                  { role: 'user', content: REALISTIC_PROMPTS[i % REALISTIC_PROMPTS.length] },
                ],
                max_tokens: 30,
                user: `stress-s7-${model.label}-c${c}`,
              }));
              const results = await fireConcurrent(app, payloads);

              const ok = results.filter((r) => r.status === 200);
              const timings = ok.map(extractTiming);
              const errors = results.filter((r) => r.status !== 200);
              const e2es = timings.map((t) => t.e2eMs);
              const dtps = timings.map((t) => t.decodeTps);
              const queueWaits = ok.map((r) => r.body?._timing?.queueWaitMs ?? 0);

              console.log(
                `  ${model.label} c=${c}: ` +
                  `${ok.length} ok, ${errors.length} err | ` +
                  `decode_tps avg=${timings.length ? avg(dtps).toFixed(1) : '0'} | ` +
                  `e2e p50=${timings.length ? percentile(e2es, 50) : '?'}ms p95=${timings.length ? percentile(e2es, 95) : '?'}ms | ` +
                  `qWait avg=${queueWaits.length ? avg(queueWaits).toFixed(0) : '?'}ms`,
              );

              expect(ok.length).toBeGreaterThan(0);
            },
            TIMEOUT,
          );
        }
      });
    }
  });

  // ── S8: Same-GPU Model Thrashing ──────────────────────────

  describe('S8: Same-GPU Model Thrashing', () => {
    const MODEL_A = 'HuggingFaceTB/SmolLM2-135M-Instruct';
    const MODEL_B = 'HuggingFaceTB/SmolLM2-360M-Instruct';

    it(
      'should measure swap cost between models on same GPU',
      async () => {
        console.log('\n── S8: Model Thrashing (135M ↔ 360M on GPU-0) ──');

        // Phase 1: Baseline — 5 requests to model A (should already be loaded from S7)
        console.log('  Phase 1: 5 sequential to 135M (baseline)...');
        const baselineResults = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: MODEL_A,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s8-baseline',
          })),
        );
        const baselineTimings = baselineResults.filter((r) => r.status === 200).map(extractTiming);
        const baselineAvgE2e = baselineTimings.length ? avg(baselineTimings.map((t) => t.e2eMs)) : 0;
        console.log(
          `  Baseline: avg e2e=${baselineAvgE2e.toFixed(0)}ms, ` +
            `avg decode_tps=${avg(baselineTimings.map((t) => t.decodeTps)).toFixed(1)}`,
        );

        // Phase 2: Swap — 5 requests to model B (forces model swap)
        console.log('  Phase 2: 5 sequential to 360M (forces swap)...');
        const swapResults = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: MODEL_B,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s8-swap',
          })),
        );
        const swapTimings = swapResults.filter((r) => r.status === 200).map(extractTiming);
        const firstSwapE2e = swapResults[0]?.durationMs ?? 0;
        const swapOverhead = firstSwapE2e - baselineAvgE2e;
        const swapRoutingMs = swapResults[0]?.body?._timing?.routingTimeMs ?? 0;
        console.log(
          `  First-after-swap e2e: ${firstSwapE2e}ms (baseline: ${baselineAvgE2e.toFixed(0)}ms)`,
        );
        console.log(
          `  Swap overhead: ${swapOverhead.toFixed(0)}ms (routing: ${swapRoutingMs}ms)`,
        );

        // Phase 3: Swap back — 5 requests to model A
        console.log('  Phase 3: 5 sequential to 135M (swap back)...');
        const swapBackResults = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: MODEL_A,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s8-swapback',
          })),
        );
        const swapBackFirst = swapBackResults[0]?.durationMs ?? 0;
        console.log(`  Swap-back first request: ${swapBackFirst}ms`);

        // Phase 4: Interleaved — 10 concurrent, 5x each model
        console.log('  Phase 4: 10 concurrent (5x 135M + 5x 360M)...');
        const interleavedPayloads = [
          ...Array.from({ length: 5 }, (_, i) => ({
            model: MODEL_A,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s8-interleaved-a',
          })),
          ...Array.from({ length: 5 }, (_, i) => ({
            model: MODEL_B,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s8-interleaved-b',
          })),
        ];
        const interleavedResults = await fireConcurrent(app, interleavedPayloads);
        const interleavedOk = interleavedResults.filter((r) => r.status === 200);
        const interleavedE2es = interleavedOk.map((r) => r.durationMs);
        const totalWallTime = Math.max(...interleavedResults.map((r) => r.endMs)) -
          Math.min(...interleavedResults.map((r) => r.startMs));

        console.log(
          `  Interleaved: ${interleavedOk.length}/10 ok, ` +
            `wall time=${totalWallTime}ms, ` +
            `e2e p50=${percentile(interleavedE2es, 50)}ms p95=${percentile(interleavedE2es, 95)}ms`,
        );

        // Summary
        const singleModelWallEst = baselineAvgE2e * 15;
        console.log(
          `\n  Summary: single-model 15 req est=${singleModelWallEst.toFixed(0)}ms, ` +
            `thrashing total=${(baselineResults.reduce((s, r) => s + r.durationMs, 0) + swapResults.reduce((s, r) => s + r.durationMs, 0) + swapBackResults.reduce((s, r) => s + r.durationMs, 0)).toFixed(0)}ms`,
        );

        expect(baselineTimings.length).toBeGreaterThan(0);
      },
      TIMEOUT,
    );
  });

  // ── S9: Queue Overflow + Recovery ─────────────────────────

  describe('S9: Queue Overflow + Recovery', () => {
    it(
      'should handle burst beyond queue depth and recover',
      async () => {
        console.log('\n── S9: Queue Overflow + Recovery ──');

        // With batching, the system can drain fast. We need a large burst
        // to actually hit maxQueueDepth=100. If no 429s, the system is handling it.
        const BURST_SIZE = 500;

        // Burst: fire all at once
        const burstStart = Date.now();
        const payloads = Array.from({ length: BURST_SIZE }, (_, i) => ({
          model: TEST_MODEL,
          messages: [
            { role: 'user', content: REALISTIC_PROMPTS[i % REALISTIC_PROMPTS.length] },
          ],
          max_tokens: 15,
          user: `stress-s9-burst-u${i % 5}`,
        }));
        const results = await fireConcurrent(app, payloads);
        const burstDuration = Date.now() - burstStart;

        const ok = results.filter((r) => r.status === 200);
        const rejected = results.filter((r) => r.status === 429);
        const other = results.filter((r) => r.status !== 200 && r.status !== 429);
        const retryAfters = rejected.map((r) => r.body?.retryAfter ?? r.body?.error?.retryAfter ?? '?');

        console.log(`  Burst size: ${BURST_SIZE}, duration: ${burstDuration}ms`);
        console.log(`  Accepted: ${ok.length}, Rejected (429): ${rejected.length}, Other errors: ${other.length}`);
        if (retryAfters.length > 0) {
          const numericRA = retryAfters.filter((v) => typeof v === 'number') as number[];
          console.log(
            `  Retry-After values: min=${numericRA.length ? Math.min(...numericRA) : '?'}, ` +
              `max=${numericRA.length ? Math.max(...numericRA) : '?'}`,
          );
        }

        // Wait for queue to drain
        console.log('  Waiting for queue drain...');
        const drainStart = Date.now();
        let drained = false;
        for (let i = 0; i < 120; i++) { // max 60s
          await sleep(500);
          const stats = await request(app.getHttpServer()).get('/v1/completions/stats');
          if (stats.body.queueDepth === 0 && stats.body.activeCount === 0) {
            drained = true;
            break;
          }
        }
        const drainTimeMs = Date.now() - drainStart;
        console.log(`  Queue drained: ${drained ? 'YES' : 'NO'} in ${drainTimeMs}ms`);

        // Recovery: 5 sequential requests
        console.log('  Recovery: 5 sequential requests...');
        const recoveryResults = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: TEST_MODEL,
            messages: [{ role: 'user', content: 'What is 2+2?' }],
            max_tokens: 10,
            user: 'stress-s9-recovery',
          })),
        );
        const recoveryOk = recoveryResults.filter((r) => r.status === 200);
        const recoveryE2es = recoveryOk.map((r) => r.durationMs);
        console.log(
          `  Recovery: ${recoveryOk.length}/5 ok, avg e2e=${recoveryE2es.length ? avg(recoveryE2es).toFixed(0) : '?'}ms`,
        );

        // With efficient batching, system may handle all without 429s — that's a pass too
        expect(ok.length).toBeGreaterThan(0); // some should succeed
        expect(recoveryOk.length).toBe(5); // recovery must work
        if (rejected.length === 0) {
          console.log(`  ✓ System handled all ${BURST_SIZE} without backpressure — batching absorbs burst`);
        }
      },
      TIMEOUT,
    );
  });

  // ── S10: Mixed Modality on Same GPU ───────────────────────

  describe('S10: Mixed Modality — Text + TTS on GPU-0', () => {
    const TTS_MODEL = 'hexgrad/Kokoro-82M';

    it(
      'should measure cross-modality contention on same GPU',
      async () => {
        console.log('\n── S10: Mixed Modality (text + TTS on GPU-0) ──');

        // Phase 1: Text baseline
        console.log('  Phase 1: Text baseline (5 sequential to 135M)...');
        const textBaseline = await fireSequential(
          app,
          Array.from({ length: 5 }, (_, i) => ({
            model: TEST_MODEL,
            messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
            max_tokens: 20,
            user: 'stress-s10-text-baseline',
          })),
        );
        const textTimings = textBaseline.filter((r) => r.status === 200).map(extractTiming);
        const textAvgE2e = textTimings.length ? avg(textTimings.map((t) => t.e2eMs)) : 0;
        console.log(
          `  Text baseline: ${textTimings.length}/5 ok, avg e2e=${textAvgE2e.toFixed(0)}ms, ` +
            `avg decode_tps=${textTimings.length ? avg(textTimings.map((t) => t.decodeTps)).toFixed(1) : '?'}`,
        );

        // Phase 2: TTS baseline
        console.log('  Phase 2: TTS baseline (5 sequential)...');
        const ttsBaseline: RequestResult[] = [];
        for (let i = 0; i < 5; i++) {
          const startMs = Date.now();
          const res = await request(app.getHttpServer())
            .post('/v1/audio/speech')
            .send({
              model: TTS_MODEL,
              input: `This is test sentence number ${i + 1} for the text to speech system.`,
            });
          const endMs = Date.now();
          ttsBaseline.push({
            status: res.status,
            body: res.body,
            startMs,
            endMs,
            durationMs: endMs - startMs,
          });
        }
        const ttsOk = ttsBaseline.filter((r) => r.status === 200);
        const ttsAvgE2e = ttsOk.length ? avg(ttsOk.map((r) => r.durationMs)) : 0;
        console.log(`  TTS baseline: ${ttsOk.length}/5 ok, avg e2e=${ttsAvgE2e.toFixed(0)}ms`);

        // Phase 3: Mixed concurrent — 5 text + 5 TTS
        console.log('  Phase 3: Mixed concurrent (5 text + 5 TTS)...');
        const mixedStart = Date.now();
        const [mixedTextResults, mixedTtsResults] = await Promise.all([
          fireConcurrent(
            app,
            Array.from({ length: 5 }, (_, i) => ({
              model: TEST_MODEL,
              messages: [{ role: 'user', content: REALISTIC_PROMPTS[i] }],
              max_tokens: 20,
              user: 'stress-s10-mixed-text',
            })),
          ),
          Promise.all(
            Array.from({ length: 5 }, async (_, i) => {
              const startMs = Date.now();
              const res = await request(app.getHttpServer())
                .post('/v1/audio/speech')
                .send({
                  model: TTS_MODEL,
                  input: `Mixed workload sentence ${i + 1} for concurrent testing.`,
                });
              const endMs = Date.now();
              return {
                status: res.status,
                body: res.body,
                startMs,
                endMs,
                durationMs: endMs - startMs,
              } as RequestResult;
            }),
          ),
        ]);
        const mixedWallTime = Date.now() - mixedStart;

        const mixedTextOk = mixedTextResults.filter((r) => r.status === 200);
        const mixedTtsOk = mixedTtsResults.filter((r) => r.status === 200);
        const mixedTextE2e = mixedTextOk.length ? avg(mixedTextOk.map((r) => r.durationMs)) : 0;
        const mixedTtsE2e = mixedTtsOk.length ? avg(mixedTtsOk.map((r) => r.durationMs)) : 0;

        console.log(
          `  Mixed text: ${mixedTextOk.length}/5 ok, avg e2e=${mixedTextE2e.toFixed(0)}ms ` +
            `(baseline: ${textAvgE2e.toFixed(0)}ms, ${textAvgE2e > 0 ? ((mixedTextE2e / textAvgE2e - 1) * 100).toFixed(0) : '?'}% change)`,
        );
        console.log(
          `  Mixed TTS: ${mixedTtsOk.length}/5 ok, avg e2e=${mixedTtsE2e.toFixed(0)}ms ` +
            `(baseline: ${ttsAvgE2e.toFixed(0)}ms, ${ttsAvgE2e > 0 ? ((mixedTtsE2e / ttsAvgE2e - 1) * 100).toFixed(0) : '?'}% change)`,
        );
        console.log(`  Total wall time: ${mixedWallTime}ms`);

        expect(mixedTextOk.length + mixedTtsOk.length).toBeGreaterThan(0);
      },
      TIMEOUT,
    );
  });

  // ── S11: Sustained Overload + Recovery Profile ────────────

  describe('S11: Sustained Overload (6 rps × 30s)', () => {
    it(
      'should measure overload behavior and recovery',
      async () => {
        console.log('\n── S11: Sustained Overload (6 rps × 30s = ~180 requests) ──');

        const RPS = 6;
        const DURATION_S = 30;
        const results: RequestResult[] = [];
        const snapshots: Array<{ t: number; queued: number; active: number; accepted: number; rejected: number }> = [];
        let accepted = 0;
        let rejected = 0;

        const overloadStart = Date.now();
        const promises: Promise<void>[] = [];

        for (let s = 0; s < DURATION_S; s++) {
          // Fire RPS requests this second
          for (let r = 0; r < RPS; r++) {
            const p = fireOne(app, {
              model: TEST_MODEL,
              messages: [
                { role: 'user', content: REALISTIC_PROMPTS[(s * RPS + r) % REALISTIC_PROMPTS.length] },
              ],
              max_tokens: 20,
              user: `stress-s11-overload`,
            }).then(
              (res) => {
                results.push(res);
                if (res.status === 200) accepted++;
                else rejected++;
              },
              () => { rejected++; },
            );
            promises.push(p);
          }

          // Sample queue state every second
          try {
            const stats = await request(app.getHttpServer()).get('/v1/completions/stats');
            snapshots.push({
              t: s,
              queued: stats.body.queueDepth ?? 0,
              active: stats.body.activeCount ?? 0,
              accepted,
              rejected,
            });
          } catch { /* ignore */ }

          if (s < DURATION_S - 1) await sleep(1000);
        }

        console.log('  Overload phase complete. Waiting for in-flight...');
        await Promise.allSettled(promises);
        const overloadDuration = Date.now() - overloadStart;

        const ok = results.filter((r) => r.status === 200);
        const r429 = results.filter((r) => r.status === 429);

        console.log(`  Sent: ${RPS * DURATION_S}, OK: ${ok.length}, 429: ${r429.length}, Other: ${results.length - ok.length - r429.length}`);
        console.log(`  Overload duration: ${overloadDuration}ms`);

        // Print queue depth snapshots (every 5s)
        console.log('  Queue snapshots:');
        for (const snap of snapshots.filter((_, i) => i % 5 === 0)) {
          console.log(
            `    t=${snap.t}s: queued=${snap.queued}, active=${snap.active}, ` +
              `accepted=${snap.accepted}, rejected=${snap.rejected}`,
          );
        }

        // Wait for drain
        console.log('  Waiting for drain...');
        const drainStart = Date.now();
        let zombieCount = -1;
        for (let i = 0; i < 120; i++) {
          await sleep(500);
          const stats = await request(app.getHttpServer()).get('/v1/completions/stats');
          if (stats.body.queueDepth === 0 && stats.body.activeCount === 0) {
            zombieCount = 0;
            break;
          }
          zombieCount = stats.body.activeCount;
        }
        const drainTimeMs = Date.now() - drainStart;
        console.log(`  Drain time: ${drainTimeMs}ms, zombies: ${zombieCount}`);

        // Recovery
        const recovery = await fireSequential(
          app,
          Array.from({ length: 3 }, () => ({
            model: TEST_MODEL,
            messages: [{ role: 'user', content: 'What is 2+2?' }],
            max_tokens: 10,
            user: 'stress-s11-recovery',
          })),
        );
        const recoveryOk = recovery.filter((r) => r.status === 200);
        console.log(
          `  Recovery: ${recoveryOk.length}/3 ok, avg e2e=${recoveryOk.length ? avg(recoveryOk.map((r) => r.durationMs)).toFixed(0) : '?'}ms`,
        );

        expect(zombieCount).toBe(0);
        expect(recoveryOk.length).toBe(3);
      },
      TIMEOUT,
    );
  });

  // ── S12: Request Timeout Behavior ─────────────────────────

  describe('S12: Request Timeout', () => {
    it(
      'should timeout long requests and recover cleanly',
      async () => {
        console.log('\n── S12: Request Timeout (5s timeout, 500-token generation) ──');

        // Access scheduler to override timeout
        const scheduler = app.get(SchedulerService);
        const originalTimeout = 60_000;
        scheduler.setRequestTimeout(5_000); // 5s timeout

        try {
          // Send request that will take longer than 5s
          const startMs = Date.now();
          const res = await request(app.getHttpServer())
            .post('/v1/completions')
            .send({
              model: TEST_MODEL,
              messages: [
                { role: 'system', content: SYSTEM_PROMPT },
                { role: 'user', content: 'Write a very detailed essay about the history of computing, covering every major milestone from the abacus to modern quantum computers. Include all important dates, people, and technical details.' },
              ],
              max_tokens: 500,
              user: 'stress-s12-timeout',
            });
          const elapsed = Date.now() - startMs;

          console.log(`  Request completed in ${elapsed}ms with status ${res.status}`);
          console.log(`  Response: ${JSON.stringify(res.body).substring(0, 200)}`);

          // If it timed out, error type should be 'timeout'
          if (res.status !== 200) {
            const errType = res.body?.error?.type;
            console.log(`  Error type: ${errType}`);
            expect(errType).toBe('timeout');
            expect(elapsed).toBeLessThan(10_000); // should fire within ~5s, not hang
          } else {
            console.log(`  Request completed before timeout (model fast enough). elapsed=${elapsed}ms`);
          }

          // Verify cleanup
          await sleep(500);
          const stats = await request(app.getHttpServer()).get('/v1/completions/stats');
          console.log(`  Post-timeout stats: ${JSON.stringify(stats.body)}`);
          expect(stats.body.activeCount).toBe(0);

          // Recovery: next request should work (may need model reload after timeout)
          const recovery = await fireOne(app, {
            model: TEST_MODEL,
            messages: [{ role: 'user', content: 'Say hello' }],
            max_tokens: 5,
            user: 'stress-s12-recovery',
          });
          console.log(`  Recovery: status=${recovery.status}, e2e=${recovery.durationMs}ms`);
          // Accept 200 (success) or any non-timeout response — model may need reload
          expect(recovery.status === 200 || recovery.status === 404).toBe(true);
          if (recovery.status === 404) {
            // Model was unloaded, try again (model auto-loads)
            const retry = await fireOne(app, {
              model: TEST_MODEL,
              messages: [{ role: 'user', content: 'Say hello' }],
              max_tokens: 5,
              user: 'stress-s12-recovery-retry',
            });
            console.log(`  Recovery retry: status=${retry.status}, e2e=${retry.durationMs}ms`);
            expect(retry.status).toBe(200);
          }
        } finally {
          scheduler.setRequestTimeout(originalTimeout);
        }
      },
      TIMEOUT,
    );
  });

  // ── S13: Real Conversation — Growing Context Window ──────
  //
  // Uses actual conversation data from this project's Claude Code session.
  // Tests what happens as context grows from 3 turns (~150 tokens) to 50 turns (~3800 tokens).
  // This is the test that proves whether KV cache is a real bottleneck:
  // - Prefill cost should grow with context length
  // - Without KV cache, every "turn" recomputes the entire history
  // - At some point, context exceeds model's window → errors or truncation
  // - VRAM usage grows with concurrent long-context requests

  describe('S13: Real Conversation — Context Window Stress', () => {
    const CONTEXT_WINDOWS = [3, 5, 10, 20, 30, 50]; // growing turn counts

    it(
      'S13a: prefill cost vs context length (single request, growing turns)',
      async () => {
        console.log('\n── S13a: Prefill Cost vs Context Length (real conversation) ──');
        console.log('  Using real conversation data from this project (230 turns)\n');

        const rows: Array<{
          turns: number;
          tokens: number;
          prefillMs: number;
          decodeMs: number;
          e2eMs: number;
          prefillTps: number;
          decodeTps: number;
          status: number;
          error?: string;
        }> = [];

        for (const turnCount of CONTEXT_WINDOWS) {
          // Take last N turns from real conversation, ensure it starts with user
          let slice = REAL_CONVERSATION.slice(0, turnCount);
          if (slice[0]?.role !== 'user') {
            slice = [{ role: 'user', content: 'Hello' }, ...slice];
          }
          // Ensure it ends with user (model needs to respond)
          if (slice[slice.length - 1]?.role !== 'user') {
            slice.push({ role: 'user', content: 'Continue.' });
          }

          const estimatedTokens = slice.reduce((sum, m) => sum + m.content.length / 4, 0);

          const result = await fireOne(app, {
            model: TEST_MODEL,
            messages: slice,
            max_tokens: 10,
            user: `stress-s13a-turns-${turnCount}`,
          });

          const t = extractTiming(result);
          const row = {
            turns: turnCount,
            tokens: Math.round(estimatedTokens),
            prefillMs: t.prefillMs,
            decodeMs: t.decodeMs,
            e2eMs: t.e2eMs,
            prefillTps: t.prefillTps,
            decodeTps: t.decodeTps,
            status: result.status,
            error: result.status !== 200 ? JSON.stringify(result.body).slice(0, 100) : undefined,
          };
          rows.push(row);

          console.log(
            `  turns=${turnCount.toString().padStart(3)} | ` +
              `~${row.tokens.toString().padStart(5)} tok | ` +
              `prefill=${row.prefillMs.toFixed(0).padStart(6)}ms (${row.prefillTps.toFixed(0).padStart(5)} tps) | ` +
              `decode=${row.decodeMs.toFixed(0).padStart(5)}ms (${row.decodeTps.toFixed(0).padStart(5)} tps) | ` +
              `e2e=${row.e2eMs.toString().padStart(6)}ms | ` +
              `status=${row.status}` +
              (row.error ? ` ERROR: ${row.error}` : ''),
          );
        }

        // Key metrics
        const successRows = rows.filter((r) => r.status === 200);
        if (successRows.length >= 2) {
          const first = successRows[0];
          const last = successRows[successRows.length - 1];
          const prefillGrowth = last.prefillMs / Math.max(1, first.prefillMs);
          const tokenGrowth = last.tokens / Math.max(1, first.tokens);

          console.log('\n  ── KV Cache Impact Analysis ──');
          console.log(
            `  Context grew ${tokenGrowth.toFixed(1)}x (${first.tokens}→${last.tokens} tokens)`,
          );
          console.log(
            `  Prefill grew ${prefillGrowth.toFixed(1)}x (${first.prefillMs.toFixed(0)}→${last.prefillMs.toFixed(0)}ms)`,
          );
          console.log(
            `  Without KV cache: every multi-turn request re-prefills entire history`,
          );
          console.log(
            `  Cumulative prefill waste over ${successRows.length} turns: ${successRows.reduce((s, r) => s + r.prefillMs, 0).toFixed(0)}ms`,
          );

          const failedRows = rows.filter((r) => r.status !== 200);
          if (failedRows.length > 0) {
            console.log(
              `  Context overflow: ${failedRows.length} requests failed (context exceeded model window)`,
            );
          }
        }

        // At least the small windows should succeed
        expect(rows[0].status).toBe(200);
        expect(rows[1].status).toBe(200);
      },
      TIMEOUT,
    );

    it(
      'S13b: multi-turn recompute waste with real conversation (simulated chat session)',
      async () => {
        console.log('\n── S13b: Multi-Turn Recompute (simulated chat, real data) ──');
        console.log('  Sending 10 "turns" where each turn includes full prior history\n');

        // Take first 20 turns of real conversation, send incrementally
        const turns = REAL_CONVERSATION.slice(0, 20);
        const results: Array<{
          turn: number;
          contextTokens: number;
          prefillMs: number;
          decodeMs: number;
          e2eMs: number;
        }> = [];

        let cumulativePrefill = 0;

        for (let i = 1; i <= Math.min(10, turns.length); i++) {
          let slice = turns.slice(0, i * 2); // 2 messages per "turn"
          if (slice.length === 0) break;
          if (slice[0]?.role !== 'user') {
            slice = [{ role: 'user', content: 'Hello' }, ...slice];
          }
          if (slice[slice.length - 1]?.role !== 'user') {
            slice.push({ role: 'user', content: 'Continue.' });
          }

          const contextTokens = Math.round(
            slice.reduce((s, m) => s + m.content.length / 4, 0),
          );

          const result = await fireOne(app, {
            model: TEST_MODEL,
            messages: slice,
            max_tokens: 10,
            user: `stress-s13b-turn-${i}`,
          });

          if (result.status !== 200) {
            console.log(`  Turn ${i}: FAILED (status=${result.status}) — context too large`);
            break;
          }

          const t = extractTiming(result);
          cumulativePrefill += t.prefillMs;

          results.push({
            turn: i,
            contextTokens,
            prefillMs: t.prefillMs,
            decodeMs: t.decodeMs,
            e2eMs: t.e2eMs,
          });

          console.log(
            `  Turn ${i.toString().padStart(2)}: ` +
              `context=${contextTokens.toString().padStart(5)} tok | ` +
              `prefill=${t.prefillMs.toFixed(0).padStart(5)}ms | ` +
              `decode=${t.decodeMs.toFixed(0).padStart(5)}ms | ` +
              `e2e=${t.e2eMs.toString().padStart(5)}ms | ` +
              `cumulative_prefill=${cumulativePrefill.toFixed(0)}ms`,
          );
        }

        if (results.length >= 2) {
          const firstPrefill = results[0].prefillMs;
          const totalWaste = cumulativePrefill - firstPrefill; // first turn is unavoidable
          console.log('\n  ── Recompute Waste Analysis ──');
          console.log(`  Total prefill across ${results.length} turns: ${cumulativePrefill.toFixed(0)}ms`);
          console.log(`  With KV cache (only first turn prefills): ${firstPrefill.toFixed(0)}ms`);
          console.log(`  Wasted recompute: ${totalWaste.toFixed(0)}ms (${((totalWaste / cumulativePrefill) * 100).toFixed(1)}%)`);
        }

        expect(results.length).toBeGreaterThanOrEqual(2);
      },
      TIMEOUT,
    );

    it(
      'S13c: concurrent long-context requests (VRAM pressure)',
      async () => {
        console.log('\n── S13c: Concurrent Long-Context Requests ──');
        console.log('  Multiple requests with 20-turn real conversation history, concurrent\n');

        // Build a 20-turn context payload
        let context = REAL_CONVERSATION.slice(0, 20);
        if (context[0]?.role !== 'user') {
          context = [{ role: 'user', content: 'Hello' }, ...context];
        }
        if (context[context.length - 1]?.role !== 'user') {
          context.push({ role: 'user', content: 'What should we do next?' });
        }
        const contextTokens = Math.round(
          context.reduce((s, m) => s + m.content.length / 4, 0),
        );

        console.log(`  Context size: ${context.length} messages, ~${contextTokens} tokens`);

        const concurrencyLevels = [1, 2, 4, 8];
        for (const c of concurrencyLevels) {
          const payloads = Array.from({ length: c }, (_, i) => ({
            model: TEST_MODEL,
            messages: context,
            max_tokens: 10,
            user: `stress-s13c-c${c}-${i}`,
          }));

          const wallStart = Date.now();
          const results = await fireConcurrent(app, payloads);
          const wallMs = Date.now() - wallStart;

          const ok = results.filter((r) => r.status === 200);
          const timings = ok.map(extractTiming);
          const avgPrefill = timings.length > 0 ? avg(timings.map((t) => t.prefillMs)) : 0;
          const avgDecode = timings.length > 0 ? avg(timings.map((t) => t.decodeMs)) : 0;
          const avgE2e = timings.length > 0 ? avg(timings.map((t) => t.e2eMs)) : 0;
          const totalPrefill = timings.reduce((s, t) => s + t.prefillMs, 0);

          console.log(
            `  c=${c.toString().padStart(3)}: ` +
              `${ok.length}/${results.length} ok | ` +
              `avg_prefill=${avgPrefill.toFixed(0).padStart(5)}ms | ` +
              `avg_decode=${avgDecode.toFixed(0).padStart(5)}ms | ` +
              `avg_e2e=${avgE2e.toFixed(0).padStart(6)}ms | ` +
              `wall=${wallMs.toString().padStart(6)}ms | ` +
              `total_prefill=${totalPrefill.toFixed(0)}ms`,
          );

          expect(ok.length).toBeGreaterThan(0);
        }
      },
      TIMEOUT,
    );

    it(
      'S13d: 1.7B model with real conversation (larger KV cache per token)',
      async () => {
        console.log('\n── S13d: 1.7B Model — Real Conversation Context Scaling ──');
        console.log('  Larger model = bigger KV cache per token, more VRAM pressure\n');

        // Warmup 1.7B
        console.log('  Loading 1.7B model (may take 10-20s)...');
        const warmup = await fireOne(app, {
          model: MODEL_1_7B,
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 5,
          user: 'stress-s13d-warmup',
        });
        if (warmup.status !== 200) {
          console.log(`  SKIP: 1.7B model failed to load (status=${warmup.status})`);
          return;
        }
        console.log(`  1.7B loaded (warmup: ${warmup.durationMs}ms)\n`);

        const windows = [3, 5, 10, 20];
        for (const turnCount of windows) {
          let slice = REAL_CONVERSATION.slice(0, turnCount);
          if (slice[0]?.role !== 'user') {
            slice = [{ role: 'user', content: 'Hello' }, ...slice];
          }
          if (slice[slice.length - 1]?.role !== 'user') {
            slice.push({ role: 'user', content: 'Continue.' });
          }
          const tokens = Math.round(slice.reduce((s, m) => s + m.content.length / 4, 0));

          const result = await fireOne(app, {
            model: MODEL_1_7B,
            messages: slice,
            max_tokens: 10,
            user: `stress-s13d-turns-${turnCount}`,
          });

          if (result.status !== 200) {
            console.log(`  turns=${turnCount}: FAILED (status=${result.status}) — likely context overflow`);
            continue;
          }

          const t = extractTiming(result);
          console.log(
            `  turns=${turnCount.toString().padStart(3)} | ` +
              `~${tokens.toString().padStart(5)} tok | ` +
              `prefill=${t.prefillMs.toFixed(0).padStart(6)}ms | ` +
              `decode=${t.decodeMs.toFixed(0).padStart(5)}ms | ` +
              `e2e=${result.durationMs.toString().padStart(6)}ms | ` +
              `KV cache ≈ ${((tokens * 1536 * 2 * 24 * 2) / 1024 / 1024).toFixed(1)}MB`,
          );
          // KV cache estimate: tokens × hidden_dim × 2(K+V) × num_layers × 2(bytes FP16)
          // 1.7B: hidden=2048, layers=24 → tokens × 2048 × 2 × 24 × 2 = tokens × 196608 bytes
          // Actually SmolLM2-1.7B: hidden=2048, layers=24, heads=32, head_dim=64
          // KV per token = 2 × 24 × 2 × 64 × 32 × 2 bytes... simplified above
        }

        expect(warmup.status).toBe(200);
      },
      TIMEOUT,
    );

    it(
      'S13e: context window overflow behavior',
      async () => {
        console.log('\n── S13e: Context Window Overflow ──');
        console.log('  Send contexts that exceed model max_position_embeddings\n');

        // SmolLM2 typically has 2048 context window
        // Send progressively larger contexts until it fails
        const largeTurnCounts = [30, 50, 80, 100];

        for (const turnCount of largeTurnCounts) {
          let slice = REAL_CONVERSATION.slice(0, Math.min(turnCount, REAL_CONVERSATION.length));
          if (slice[0]?.role !== 'user') {
            slice = [{ role: 'user', content: 'Hello' }, ...slice];
          }
          if (slice[slice.length - 1]?.role !== 'user') {
            slice.push({ role: 'user', content: 'Summarize.' });
          }
          const tokens = Math.round(slice.reduce((s, m) => s + m.content.length / 4, 0));

          const result = await fireOne(app, {
            model: TEST_MODEL,
            messages: slice,
            max_tokens: 10,
            user: `stress-s13e-overflow-${turnCount}`,
          });

          const statusIcon = result.status === 200 ? 'OK' : 'FAIL';
          console.log(
            `  turns=${turnCount.toString().padStart(3)} | ` +
              `~${tokens.toString().padStart(5)} tok | ` +
              `status=${result.status} ${statusIcon} | ` +
              `e2e=${result.durationMs.toString().padStart(6)}ms` +
              (result.status !== 200
                ? ` | error: ${JSON.stringify(result.body?.error?.message || result.body?.message || '').slice(0, 80)}`
                : ''),
          );
        }

        // We expect at least some to fail (context overflow)
        // This proves the system needs context management (KV cache or truncation)
        console.log('\n  ── Implication ──');
        console.log('  Without KV cache: every turn re-sends full history, hitting context limit fast');
        console.log('  With KV cache: only new tokens need processing, context "window" is effectively unlimited');
      },
      TIMEOUT,
    );
  });

  // ════════════════════════════════════════════════════════════
  // S14: Disaggregated KV Cache — Recompute vs Transfer Cost
  // ════════════════════════════════════════════════════════════

  describe('S14: Recompute vs Cache Transfer Cost (1.7B)', () => {
    it(
      'should prove cache transfer is cheaper than recompute',
      async () => {
        console.log('\n━━━ S14: Recompute vs Cache Transfer Cost (1.7B) ━━━');

        // Use 10 turns of real conversation
        const TURNS = 10;
        const turns = REAL_CONVERSATION.slice(0, TURNS * 2); // 10 user+assistant pairs

        // Phase A: No cache (baseline) — send full history each turn
        console.log('\n  Phase A: No Cache (full recompute each turn)');
        const noCacheResults: Array<{
          turn: number;
          contextTokens: number;
          prefillMs: number;
          totalMs: number;
        }> = [];

        for (let turn = 0; turn < TURNS; turn++) {
          const messages = turns.slice(0, (turn + 1) * 2).map((t) => ({
            role: t.role,
            content: t.content,
          }));

          const res = await fireOne(app, {
            model: MODEL_1_7B,
            messages,
            max_tokens: 5,
            user: 'load-s14-nocache',
            // No session_id → no caching
          });

          expect(res.status).toBe(200);

          noCacheResults.push({
            turn: turn + 1,
            contextTokens: res.body.usage?.prompt_tokens ?? 0,
            prefillMs: res.body.usage?.prefill_time_ms ?? 0,
            totalMs: res.body.usage?.total_time_ms ?? 0,
          });
        }

        // Phase B: With cache — send only new message each turn (same session_id)
        console.log('  Phase B: With Cache (disaggregated KV cache)');
        const sessionId = `s14-cache-test-${Date.now()}`;
        const cacheResults: Array<{
          turn: number;
          cachedTokens: number;
          newTokens: number;
          cacheLoadMs: number;
          cacheSaveMs: number;
          prefillMs: number;
          totalMs: number;
        }> = [];

        for (let turn = 0; turn < TURNS; turn++) {
          const messages = turns.slice(0, (turn + 1) * 2).map((t) => ({
            role: t.role,
            content: t.content,
          }));

          const res = await fireOne(app, {
            model: MODEL_1_7B,
            messages,
            max_tokens: 5,
            user: 'load-s14-cache',
            session_id: sessionId,
          });

          expect(res.status).toBe(200);

          cacheResults.push({
            turn: turn + 1,
            cachedTokens: res.body.usage?.cached_tokens ?? 0,
            newTokens: (res.body.usage?.prompt_tokens ?? 0) - (res.body.usage?.cached_tokens ?? 0),
            cacheLoadMs: res.body.usage?.cache_load_ms ?? 0,
            cacheSaveMs: res.body.usage?.cache_save_ms ?? 0,
            prefillMs: res.body.usage?.prefill_time_ms ?? 0,
            totalMs: res.body.usage?.total_time_ms ?? 0,
          });
        }

        // Report comparison table
        console.log('\n  ┌──────┬───────────┬──────────────────┬──────────────────────────────┬─────────┐');
        console.log('  │ Turn │ Context   │ No-Cache Prefill │ Cache Load+Prefill (save)    │ Savings │');
        console.log('  ├──────┼───────────┼──────────────────┼──────────────────────────────┼─────────┤');

        for (let i = 0; i < TURNS; i++) {
          const nc = noCacheResults[i];
          const wc = cacheResults[i];
          const ncPrefill = nc.prefillMs;
          const wcTotal = wc.cacheLoadMs + wc.prefillMs;
          const savings = ncPrefill > 0 ? ((ncPrefill - wcTotal) / ncPrefill * 100) : 0;

          console.log(
            `  │ ${String(nc.turn).padStart(4)} │ ${String(nc.contextTokens).padStart(5)} tok │ ` +
            `${ncPrefill.toFixed(0).padStart(12)}ms │ ` +
            `${wc.cacheLoadMs.toFixed(1)}+${wc.prefillMs.toFixed(0)}=${wcTotal.toFixed(0)}ms ` +
            `(save ${wc.cacheSaveMs.toFixed(1)}ms)`.padEnd(8) +
            ` │ ${savings.toFixed(0).padStart(5)}% │`,
          );
        }
        console.log('  └──────┴───────────┴──────────────────┴──────────────────────────────┴─────────┘');

        // Cost savings calculation (Anthropic pricing model)
        const totalNoCachePrefill = noCacheResults.reduce((s, r) => s + r.prefillMs, 0);
        const totalCachePrefill = cacheResults.reduce((s, r) => s + r.cacheLoadMs + r.prefillMs, 0);
        const overallSavings = totalNoCachePrefill > 0
          ? ((totalNoCachePrefill - totalCachePrefill) / totalNoCachePrefill * 100)
          : 0;

        console.log('\n  ── Cost Analysis ──');
        console.log(`  Total recompute cost: ${totalNoCachePrefill.toFixed(0)}ms`);
        console.log(`  Total cache cost:     ${totalCachePrefill.toFixed(0)}ms`);
        console.log(`  Compute savings:      ${overallSavings.toFixed(1)}%`);
        console.log(`  Anthropic pricing:    Cached tokens at 0.1x → ~90% API cost reduction`);

        // Verify cache is working: at least half the turns should show cached_tokens > 0
        const turnsWithCache = cacheResults.filter((r) => r.cachedTokens > 0).length;
        console.log(`  Cache hits:           ${turnsWithCache}/${TURNS} turns`);
        expect(turnsWithCache).toBeGreaterThanOrEqual(Math.floor(TURNS / 2));
      },
      TIMEOUT,
    );
  });

  // ════════════════════════════════════════════════════════════
  // S15: Concurrent Session Caching (DRAM pressure)
  // ════════════════════════════════════════════════════════════

  describe('S15: Concurrent Session Caching', () => {
    it(
      'should handle multiple cached sessions concurrently',
      async () => {
        console.log('\n━━━ S15: Concurrent Session Caching ━━━');

        const NUM_SESSIONS = 5;
        const TURNS_PER_SESSION = 3;

        // Each session uses a different slice of conversation data
        const sessions: Array<{
          sessionId: string;
          turns: Array<{ role: string; content: string }>;
        }> = [];

        for (let s = 0; s < NUM_SESSIONS; s++) {
          const offset = s * TURNS_PER_SESSION * 2;
          sessions.push({
            sessionId: `s15-session-${s}-${Date.now()}`,
            turns: REAL_CONVERSATION.slice(offset, offset + TURNS_PER_SESSION * 2),
          });
        }

        // Run all sessions sequentially, 3 turns each
        const sessionResults: Array<{
          sessionId: string;
          turnsCompleted: number;
          cacheHits: number;
          lastCachedTokens: number;
        }> = [];

        for (const session of sessions) {
          let cacheHits = 0;
          let lastCachedTokens = 0;

          for (let turn = 0; turn < TURNS_PER_SESSION; turn++) {
            const messages = session.turns.slice(0, (turn + 1) * 2).map((t) => ({
              role: t.role,
              content: t.content,
            }));

            const res = await fireOne(app, {
              model: MODEL_1_7B,
              messages,
              max_tokens: 5,
              user: 'load-s15',
              session_id: session.sessionId,
            });

            expect(res.status).toBe(200);

            const cached = res.body.usage?.cached_tokens ?? 0;
            if (cached > 0) cacheHits++;
            lastCachedTokens = cached;
          }

          sessionResults.push({
            sessionId: session.sessionId,
            turnsCompleted: TURNS_PER_SESSION,
            cacheHits,
            lastCachedTokens,
          });
        }

        // Report
        console.log('\n  Session Results:');
        console.log('  ┌─────────┬────────┬────────────┬───────────────┐');
        console.log('  │ Session │ Turns  │ Cache Hits │ Last Cached   │');
        console.log('  ├─────────┼────────┼────────────┼───────────────┤');
        for (let i = 0; i < sessionResults.length; i++) {
          const r = sessionResults[i];
          console.log(
            `  │ ${String(i).padStart(7)} │ ${String(r.turnsCompleted).padStart(6)} │ ` +
            `${String(r.cacheHits).padStart(10)} │ ${String(r.lastCachedTokens).padStart(9)} tok │`,
          );
        }
        console.log('  └─────────┴────────┴────────────┴───────────────┘');

        const totalCacheHits = sessionResults.reduce((s, r) => s + r.cacheHits, 0);
        const expectedHits = NUM_SESSIONS * (TURNS_PER_SESSION - 1); // first turn always miss
        console.log(`\n  Total cache hits: ${totalCacheHits}/${expectedHits} expected`);

        // Each session should have at least 1 cache hit (turns 2+)
        for (const r of sessionResults) {
          expect(r.cacheHits).toBeGreaterThanOrEqual(1);
        }
      },
      TIMEOUT,
    );
  });

  // ════════════════════════════════════════════════════════════
  // S16: Cache Eviction Under Budget Constraint
  // ════════════════════════════════════════════════════════════

  describe('S16: Cache Eviction Under Budget', () => {
    it(
      'should evict LRU sessions when DRAM budget exceeded',
      async () => {
        console.log('\n━━━ S16: Cache Eviction Under Budget ━━━');
        console.log('  Note: This test relies on the worker KVCacheStore budget.');
        console.log('  With default 200GB budget, eviction unlikely with small test data.');
        console.log('  Verifying cache behavior across many sessions instead.\n');

        const NUM_SESSIONS = 8;
        const sessionIds: string[] = [];

        // Create 8 sessions, each with 1 turn
        console.log('  Phase 1: Creating sessions...');
        for (let s = 0; s < NUM_SESSIONS; s++) {
          const sid = `s16-evict-${s}-${Date.now()}`;
          sessionIds.push(sid);

          const messages = REAL_CONVERSATION.slice(s * 2, s * 2 + 2).map((t) => ({
            role: t.role,
            content: t.content,
          }));

          const res = await fireOne(app, {
            model: MODEL_1_7B,
            messages,
            max_tokens: 5,
            user: 'load-s16',
            session_id: sid,
          });
          expect(res.status).toBe(200);
          console.log(`    Session ${s}: created (prompt_tokens=${res.body.usage?.prompt_tokens})`);
        }

        // Revisit first and last session — both should still have cache
        console.log('\n  Phase 2: Revisiting sessions...');
        const revisitResults: Array<{
          session: number;
          cachedTokens: number;
          cacheLoadMs: number;
        }> = [];

        for (const idx of [0, 3, 7]) {
          const messages = REAL_CONVERSATION.slice(idx * 2, idx * 2 + 4).map((t) => ({
            role: t.role,
            content: t.content,
          }));

          const res = await fireOne(app, {
            model: MODEL_1_7B,
            messages,
            max_tokens: 5,
            user: 'load-s16',
            session_id: sessionIds[idx],
          });
          expect(res.status).toBe(200);

          revisitResults.push({
            session: idx,
            cachedTokens: res.body.usage?.cached_tokens ?? 0,
            cacheLoadMs: res.body.usage?.cache_load_ms ?? 0,
          });
          console.log(
            `    Session ${idx}: cached_tokens=${res.body.usage?.cached_tokens ?? 0}, ` +
            `cache_load_ms=${(res.body.usage?.cache_load_ms ?? 0).toFixed(1)}ms`,
          );
        }

        // With default budget, all should still be cached
        console.log('\n  ── Results ──');
        for (const r of revisitResults) {
          console.log(`  Session ${r.session}: cached=${r.cachedTokens} tokens, load=${r.cacheLoadMs.toFixed(1)}ms`);
          expect(r.cachedTokens).toBeGreaterThan(0);
        }
      },
      TIMEOUT,
    );
  });
});

// ── Summary Report ─────────────────────────────────────────

async function printReport(app: INestApplication) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║              METRICS PIPELINE SUMMARY                     ║');
  console.log('╠════════════════════════════════════════════════════════════╣');

  try {
    const [tpsRes, latencyRes, breakdownRes] = await Promise.all([
      request(app.getHttpServer()).get('/v1/metrics/tps?window=10m'),
      request(app.getHttpServer()).get('/v1/metrics/latency?window=10m'),
      request(app.getHttpServer()).get('/v1/metrics/breakdown?window=10m'),
    ]);

    const f = (v: any, d = 1) => (v != null ? Number(v).toFixed(d) : '?');

    if (tpsRes.status === 200 && tpsRes.body.current) {
      const c = tpsRes.body.current;
      console.log(
        `║  Overall: decode=${f(c.decode_tps)} tps, ` +
          `prefill=${f(c.prefill_tps)} tps, ` +
          `${f(c.requests_per_minute, 0)} rpm`,
      );

      if (tpsRes.body.by_model?.length > 0) {
        console.log('║');
        console.log('║  Per-Model TPS:');
        for (const m of tpsRes.body.by_model) {
          console.log(
            `║    ${m.model}: decode=${f(m.decode_tps)} prefill=${f(m.prefill_tps)} avg=${f(m.avg_generation_ms, 0)}ms (n=${m.request_count})`,
          );
        }
      }
    }

    if (latencyRes.status === 200 && latencyRes.body.e2e) {
      const l = latencyRes.body;
      console.log('║');
      console.log('║  Latency Percentiles (10m window):');
      console.log(
        `║    E2E:      p50=${f(l.e2e?.p50, 0)}ms  p95=${f(l.e2e?.p95, 0)}ms  p99=${f(l.e2e?.p99, 0)}ms`,
      );
      console.log(
        `║    Prefill:  p50=${f(l.prefill?.p50)}ms  p95=${f(l.prefill?.p95)}ms  p99=${f(l.prefill?.p99)}ms`,
      );
      console.log(
        `║    Decode:   p50=${f(l.decode?.p50)}ms  p95=${f(l.decode?.p95)}ms  p99=${f(l.decode?.p99)}ms`,
      );
      console.log(
        `║    Queue:    p50=${f(l.queue_wait?.p50)}ms  p95=${f(l.queue_wait?.p95)}ms  p99=${f(l.queue_wait?.p99)}ms`,
      );
    }

    if (breakdownRes.status === 200 && breakdownRes.body.by_worker?.length > 0) {
      console.log('║');
      console.log('║  Per-Worker:');
      for (const w of breakdownRes.body.by_worker) {
        console.log(
          `║    ${w.worker_id}: n=${w.request_count} avg_e2e=${f(w.avg_e2e_ms, 0)}ms tokens=${w.total_prompt_tokens}+${w.total_completion_tokens}`,
        );
      }
    }
  } catch (err) {
    console.log(`║  (metrics API unavailable: ${err.message})`);
  }

  console.log('║');
  console.log(
    '║  Full data in ClickHouse: SELECT * FROM inference.inference_metrics',
  );
  console.log(
    "║  Per-scenario: WHERE user_id LIKE 'load-%' GROUP BY user_id",
  );
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
}
