// One progress bar for a whole image-stack run.
//
// The backend reports each stage on its own ("Analyzing image 12 of 110
// (16%)", "Streaming detail and selecting sharp source 40 of 104", ...), so
// a bar driven by the latest message filled up, emptied and filled again for
// every stage, and a scan solved a second time without some frames started
// over from the first stage. Here every message is placed in a fixed list of
// steps whose shares are rough estimates of where a 1:1 scan spends its time
// (枫鹰雉鸡 takes 37 minutes in all; the per-step shares are not measured),
// the bar only moves forward, and the remaining time is projected from the
// time spent so far.

export type StackStepId =
  'load' | 'match' | 'solve' | 'measure' | 'compose' | 'repair' | 'fuse' | 'relate' | 'composeStations' | 'finish';

/**
 * A scan (one or several captures, one frame per position) is composed by
 * the streaming compositor; a stack with several focus layers per camera
 * position fuses every station first, relates the stations and composes
 * them, which the backend announces with "Fusing capture station".
 */
export type StackPlan = 'scan' | 'layered';

export const STACK_PLANS: Record<StackPlan, ReadonlyArray<{ id: StackStepId; share: number }>> = {
  scan: [
    { id: 'load', share: 16 },
    { id: 'match', share: 4 },
    { id: 'solve', share: 4 },
    { id: 'measure', share: 12 },
    { id: 'compose', share: 44 },
    { id: 'repair', share: 14 },
    { id: 'finish', share: 6 },
  ],
  layered: [
    { id: 'load', share: 10 },
    { id: 'match', share: 4 },
    { id: 'solve', share: 4 },
    { id: 'fuse', share: 38 },
    { id: 'relate', share: 10 },
    { id: 'composeStations', share: 28 },
    { id: 'finish', share: 6 },
  ],
};

export interface StackProgressState {
  startedAt: number;
  plan: StackPlan;
  /** Bar position, 0..100, never decreasing. */
  value: number;
  step: StackStepId;
  /** "12 / 110" within the step, when the message counts. */
  count: { current: number; total: number } | null;
  /** The scan is being solved a second time without some frames. */
  resolving: boolean;
  /** Where the second solve's own plan starts on the bar, and its span. */
  offset: number;
  span: number;
}

export const initialStackProgress = (now: number): StackProgressState => ({
  startedAt: now,
  plan: 'scan',
  value: 0,
  step: 'load',
  count: null,
  resolving: false,
  offset: 0,
  span: 100,
});

interface ParsedMessage {
  step: StackStepId;
  fraction: number;
  count: { current: number; total: number } | null;
  restart?: boolean;
  plan?: StackPlan;
}

const countIn = (message: string) => {
  const match = message.match(/(\d+)\s+of\s+(\d+)/i);
  if (!match) return null;
  const current = Number(match[1]);
  const total = Number(match[2]);
  return total > 0 ? { current: Math.min(current, total), total } : null;
};

/** The step a backend message belongs to and how far into it it is. */
export function parseStackMessage(message: string): ParsedMessage | null {
  const text = message.toLowerCase();
  const count = countIn(message);
  const fraction = count ? count.current / count.total : 0;
  const at = (step: StackStepId, share = fraction): ParsedMessage => ({ step, fraction: share, count });

  if (text.startsWith('solving the scan again')) return { step: 'load', fraction: 0, count: null, restart: true };
  if (text.includes('fusing capture station')) return { ...at('fuse'), plan: 'layered' };
  if (text.includes('relating capture stations')) return { ...at('relate', 0.1), plan: 'layered' };
  if (text.includes('composing capture station')) return { ...at('composeStations'), plan: 'layered' };
  if (text.includes('creating preview')) return at('finish', 0.7);
  if (text.includes('finalizing')) return at('finish', 0.3);
  if (text.includes('repairing soft detail')) return at('repair');
  if (
    text.includes('streaming detail') ||
    text.includes('focus-stacking image') ||
    text.includes('stitching image') ||
    text.includes('composing focus tile') ||
    text.includes('refining and selecting detail')
  )
    return at('compose');
  if (text.includes('measuring the detail')) return at('measure');
  // Rendering starts by measuring every frame's detail, then composes.
  if (text.includes('warping and blending') || text.includes('framing or lens change')) return at('measure', 0);
  if (text.startsWith('matching') && text.includes('solved poses')) return at('solve', 0.7);
  if (text.includes('determining stitching order')) return at('solve', 0.1);
  if (text.includes('stitching order')) return at('solve', 0.4);
  if (text.includes('matching image overlaps')) return at('match');
  if (text.includes('finding image matches')) return at('match', 0);
  if (text.includes('analyzing image')) return at('load');
  if (text.includes('loading and preparing') || text.includes('starting')) return at('load', 0);
  return null;
}

/** Bar position of `fraction` through `step` in a plan spanning 0..100. */
function planValue(plan: StackPlan, step: StackStepId, fraction: number) {
  let before = 0;
  for (const entry of STACK_PLANS[plan]) {
    if (entry.id === step) return before + entry.share * Math.min(1, Math.max(0, fraction));
    before += entry.share;
  }
  return before;
}

export function advanceStackProgress(state: StackProgressState, message: string | null): StackProgressState {
  if (!message) return state;
  const parsed = parseStackMessage(message);
  if (!parsed) return state;
  const plan = parsed.plan ?? state.plan;
  // Inside a layered station the fuser counts that station's own frames
  // ("Focus-stacking image 2 of 4"); the station count says more.
  if (plan === 'layered' && !STACK_PLANS.layered.some((entry) => entry.id === parsed.step)) return state;
  let { offset, span, resolving } = state;
  if (parsed.restart) {
    // The second solve repeats every step in what is left of the bar.
    offset = state.value;
    span = 100 - state.value;
    resolving = true;
  }
  const value = Math.max(
    state.value,
    Math.min(99.5, offset + (span * planValue(plan, parsed.step, parsed.fraction)) / 100),
  );
  return { ...state, plan, value, step: parsed.step, count: parsed.count, resolving, offset, span };
}

/** 1-based step number for "Step 3 of 7". */
export const stackStepNumber = (plan: StackPlan, step: StackStepId) =>
  STACK_PLANS[plan].findIndex((entry) => entry.id === step) + 1;

export const stackStepCount = (plan: StackPlan) => STACK_PLANS[plan].length;

/**
 * Seconds still to go, projected from the time spent and the bar position;
 * `null` until the projection means something (a minute in and 3% done).
 */
export function remainingSeconds(state: StackProgressState, now: number): number | null {
  const elapsed = (now - state.startedAt) / 1000;
  if (elapsed < 60 || state.value < 3) return null;
  return Math.max(0, (elapsed * (100 - state.value)) / state.value);
}

export const formatDuration = (seconds: number) => {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return minutes >= 60
    ? `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, '0')}:${String(rest).padStart(2, '0')}`
    : `${minutes}:${String(rest).padStart(2, '0')}`;
};
