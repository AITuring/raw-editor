export const isMissingTauriListenerError = (error: unknown) =>
  error instanceof TypeError && error.message.includes("listeners[eventId].handlerId");

export const disposeTauriListener = (unlisten: () => void | Promise<void>) =>
  Promise.resolve()
    .then(() => unlisten())
    .catch((error) => {
      if (!isMissingTauriListenerError(error)) {
        console.error('Failed to dispose Tauri listener:', error);
      }
    });
