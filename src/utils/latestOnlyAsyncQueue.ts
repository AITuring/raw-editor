export interface LatestOnlyAsyncQueueOptions<Input, Output> {
  execute(input: Input): Promise<Output>;
  onResult(output: Output, input: Input): void;
  onError?(error: unknown, input: Input): void;
  onBusyChange?(busy: boolean): void;
  getKey?(input: Input): string;
}

export interface LatestOnlyAsyncQueue<Input> {
  submit(input: Input): void;
  cancel(): void;
  dispose(): void;
}

/**
 * Runs at most one expensive request at a time and retains only the newest
 * request received while that work is in flight. Results from a cancelled
 * generation are ignored, which prevents an old native preview from replacing
 * the image after the active photo or editor tool changes.
 */
export function createLatestOnlyAsyncQueue<Input, Output>(
  options: LatestOnlyAsyncQueueOptions<Input, Output>,
): LatestOnlyAsyncQueue<Input> {
  let pendingInput: Input | undefined;
  let pendingKey: string | undefined;
  let runningKey: string | undefined;
  let hasPendingInput = false;
  let isRunning = false;
  let isBusy = false;
  let generation = 0;
  let disposed = false;

  const setBusy = (busy: boolean) => {
    if (isBusy === busy) return;
    isBusy = busy;
    options.onBusyChange?.(busy);
  };

  const drain = async () => {
    if (isRunning || !hasPendingInput || disposed) return;

    isRunning = true;
    setBusy(true);
    const runGeneration = generation;

    while (hasPendingInput && runGeneration === generation && !disposed) {
      const input = pendingInput as Input;
      const inputKey = pendingKey;
      pendingInput = undefined;
      pendingKey = undefined;
      hasPendingInput = false;
      runningKey = inputKey;

      try {
        const output = await options.execute(input);
        if (runGeneration === generation && !disposed) {
          options.onResult(output, input);
        }
      } catch (error) {
        if (runGeneration === generation && !disposed) {
          options.onError?.(error, input);
        }
      } finally {
        runningKey = undefined;
      }
    }

    isRunning = false;
    if (hasPendingInput && !disposed) {
      void drain();
    } else {
      setBusy(false);
    }
  };

  return {
    submit(input) {
      if (disposed) return;
      const key = options.getKey?.(input);
      if (key !== undefined && hasPendingInput && key === pendingKey) return;
      if (key !== undefined && key === runningKey) {
        pendingInput = undefined;
        pendingKey = undefined;
        hasPendingInput = false;
        return;
      }

      pendingInput = input;
      pendingKey = key;
      hasPendingInput = true;
      setBusy(true);
      void drain();
    },
    cancel() {
      generation += 1;
      pendingInput = undefined;
      pendingKey = undefined;
      hasPendingInput = false;
      setBusy(false);
    },
    dispose() {
      disposed = true;
      generation += 1;
      pendingInput = undefined;
      pendingKey = undefined;
      hasPendingInput = false;
      setBusy(false);
    },
  };
}
