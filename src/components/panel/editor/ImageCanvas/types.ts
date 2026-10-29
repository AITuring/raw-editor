import { PercentCrop, Crop } from 'react-image-crop';
import { Adjustments, Coord } from '../../../../utils/adjustments';
import { SubMask, ToolType } from '../../right/Masks';
import { AppSettings, BrushSettings, SelectedImage } from '../../../ui/AppProperties';
import { RenderSize } from '../../../../hooks/useImageRenderSize';
import { type CropGuideMode, type UprightGuide } from '../../../../types/crop';

export interface CursorPreview {
  visible: boolean;
  x: number;
  y: number;
}

export interface DrawnLine {
  brushSize: number;
  feather?: number;
  flow?: number;
  points: Array<Coord>;
  tool: ToolType;
}

export interface ImageCanvasProps {
  appSettings: AppSettings | null;
  activeAiPatchContainerId: string | null;
  activeAiSubMaskId: string | null;
  activeMaskContainerId: string | null;
  activeMaskId: string | null;
  adjustments: Adjustments;
  brushSettings: BrushSettings | null;
  crop: Crop | null;
  finalPreviewUrl: string | null;
  handleCropComplete(c: Crop, cp: PercentCrop): void;
  imageRenderSize: RenderSize;
  isAiEditing: boolean;
  isCropping: boolean;
  isMaskControlHovered: boolean;
  isMasking: boolean;
  isGuidedUprightActive: boolean;
  isSliderDragging: boolean;
  isStraightenActive: boolean;
  isRotationActive?: boolean;
  maskOverlayUrl: string | null;
  onGenerateAiMask(id: string | null, start: Coord, end: Coord): void;
  onLiveMaskPreview?: (previewMaskDef: any) => void;
  onManualCleanup?(subMaskId: string, sourceX: number, sourceY: number): Promise<void> | void;
  onQuickErase(subMaskId: string | null, startPoint: Coord, endpoint: Coord): void;
  onSelectAiSubMask(id: string | null): void;
  onSelectMask(id: string | null): void;
  onSelectAiPatchContainer?: (id: string | null) => void;
  onSelectMaskContainer?: (id: string | null) => void;
  onStraighten(val: number): void;
  onUprightGuideAdd(guide: UprightGuide): void;
  selectedImage: SelectedImage;
  setCrop(crop: Crop, perfentCrop: PercentCrop): void;
  setIsMaskHovered(isHovered: boolean): void;
  setIsMaskTouchInteracting(isInteracting: boolean): void;
  showOriginal: boolean;
  transformedOriginalUrl: string | null;
  uncroppedAdjustedPreviewUrl: string | null;
  updateSubMask(id: string | null, subMask: Partial<SubMask>): void;
  interactivePatch?: { url: string; normX: number; normY: number; normW: number; normH: number } | null;
  isWbPickerActive?: boolean;
  onWbPicked?: () => void;
  setAdjustments(fn: (prev: Adjustments) => Adjustments): void;
  cropGuideMode?: CropGuideMode;
  cropGuideRotation?: number;
  uprightGuides: UprightGuide[];
  onCropGuideChange?(mode: CropGuideMode): void;
  cursorStyle: string;
  isMaxZoom?: boolean;
  liveRotation?: number | null;
  renderPreviewExternally?: boolean;
  transformState: { scale: number; positionX: number; positionY: number };
  hasRenderedFirstFrame: boolean;
}
