import { useState } from 'react';
import { Aperture, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Adjustments } from '../../utils/adjustments';
import { AppSettings } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import LensCorrectionModal from '../modals/LensCorrectionModal';
import DetailsPanel from './Details';

interface OpticsPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  appSettings: AppSettings | null;
  onDragStateChange?: (isDragging: boolean) => void;
}

export default function OpticsPanel({ adjustments, setAdjustments, appSettings, onDragStateChange }: OpticsPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state) => state.selectedImage);
  const [isLensModalOpen, setIsLensModalOpen] = useState(false);

  return (
    <div className="space-y-4">
      <button
        className="w-full flex items-center gap-3 p-3 rounded-md border border-border-color bg-bg-tertiary text-left transition-colors hover:bg-card-active"
        onClick={() => setIsLensModalOpen(true)}
        type="button"
      >
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-surface text-text-secondary">
          <Aperture aria-hidden="true" size={17} strokeWidth={1.8} />
        </span>
        <span className="min-w-0 grow">
          <span className="block text-xs font-medium text-text-primary">{t('modals.lensCorrection.title')}</span>
          <span className="mt-0.5 block truncate text-[10px] text-text-secondary">
            {adjustments.lensModel || t('editor.crop.tooltips.lens')}
          </span>
        </span>
        <ChevronRight aria-hidden="true" className="text-text-secondary" size={15} strokeWidth={1.8} />
      </button>

      <DetailsPanel
        adjustments={adjustments}
        setAdjustments={setAdjustments}
        appSettings={appSettings}
        onDragStateChange={onDragStateChange}
        variant="optics"
      />

      <LensCorrectionModal
        isOpen={isLensModalOpen}
        onClose={() => setIsLensModalOpen(false)}
        onApply={(newParams) => {
          setAdjustments((prev: Adjustments) => ({ ...prev, ...newParams }));
        }}
        currentAdjustments={adjustments}
        selectedImage={selectedImage}
      />
    </div>
  );
}
