import { type ButtonVariant } from '../../ui/Button';

export interface ConfirmModalState {
  confirmText: string;
  confirmVariant: ButtonVariant;
  isOpen: boolean;
  message: string;
  onConfirm(): void;
  title: string;
}

export interface SettingsPanelProps {
  appSettings: any;
  onBack(): void;
  onLibraryRefresh(): void;
  onSettingsChange(settings: any): Promise<void>;
  rootPaths: string[];
}

export interface MyLens {
  maker: string;
  model: string;
}
