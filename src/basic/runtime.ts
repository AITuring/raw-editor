export type RawEditorMode = 'basic' | 'full';

/** Basic mode is the default product surface until Daily RAW 1.0 is complete. */
export const RAW_EDITOR_MODE: RawEditorMode =
  import.meta.env.VITE_RAW_EDITOR_MODE === 'full' ? 'full' : 'basic';

export const BASIC_MODE = RAW_EDITOR_MODE === 'basic';

