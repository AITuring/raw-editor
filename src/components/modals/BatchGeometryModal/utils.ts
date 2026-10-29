import { invoke } from '@tauri-apps/api/core';
import { Invokes, type SupportedTypes } from '../../ui/AppProperties';
import { useSettingsStore } from '../../../store/useSettingsStore';

export const normalizeExtensions = (extensions: string[]) =>
  Array.from(new Set(extensions.map((extension) => extension.trim().replace(/^\./, '').toLowerCase()).filter(Boolean)));

export const pathsFromSelection = (selection: string | string[] | null): string[] => {
  if (Array.isArray(selection)) return selection;
  return typeof selection === 'string' ? [selection] : [];
};

export const displayFileName = (path: string) => path.split(/[\\/]/).pop() || path;

export const getSupportedImageTypes = async (): Promise<SupportedTypes> => {
  const settingsStore = useSettingsStore.getState();
  if (settingsStore.supportedTypes) return settingsStore.supportedTypes;

  const supportedTypes = await invoke<SupportedTypes>(Invokes.GetSupportedFileTypes);
  settingsStore.setSupportedTypes(supportedTypes);
  return supportedTypes;
};

export const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));

export const ORIENTATION_KEYS = [
  'modals.batchGeometry.orientationStep0',
  'modals.batchGeometry.orientationStep1',
  'modals.batchGeometry.orientationStep2',
  'modals.batchGeometry.orientationStep3',
] as const;

export const normalizeOrientationSteps = (steps: number) => ((steps % 4) + 4) % 4;
