export type SourceRole = 'reference' | 'target';

export interface StyleImage {
  isPath: boolean;
  name: string;
  url: string;
  width: number;
  height: number;
  path?: string;
  image: HTMLImageElement;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

export interface StyleTransferModalProps {
  fullPage?: boolean;
  isOpen: boolean;
  onClose(): void;
}
