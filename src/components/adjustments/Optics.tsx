import { useMemo, useState } from 'react';
import { Aperture, Check, ChevronRight, Loader2, TriangleAlert } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { LensAdjustment, type Adjustments } from '../../utils/adjustments';
import type { AppSettings } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import LensCorrectionModal from '../modals/LensCorrectionModal';
import DetailsPanel from './Details';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { AdjustmentSubsection, AdjustmentTabs } from '../ui/AdjustmentSubsection';
import type { LensProfileStatus } from '../../hooks/useAutoLensProfile';

interface OpticsPanelProps {
  adjustments: Adjustments;
  setAdjustments(update: Partial<Adjustments> | ((previous: Adjustments) => Adjustments)): void;
  appSettings: AppSettings | null;
  lensProfileStatus?: LensProfileStatus;
  onDragStateChange?: (isDragging: boolean) => void;
}

export default function OpticsPanel({
  adjustments,
  setAdjustments,
  appSettings,
  lensProfileStatus = 'idle',
  onDragStateChange,
}: OpticsPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state) => state.selectedImage);
  const [isLensModalOpen, setIsLensModalOpen] = useState(false);
  const activeTab = adjustments.lensCorrectionMode === 'auto' ? 'profile' : 'manual';

  const tabs = useMemo(
    () => [
      { id: 'profile' as const, label: t('adjustments.optics.profile') },
      { id: 'manual' as const, label: t('modals.lensCorrection.modeManual') },
    ],
    [t],
  );

  const updateAdjustment = (key: LensAdjustment, value: boolean | number | string) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: value }));
  };

  const handleTabChange = (tab: 'profile' | 'manual') => {
    updateAdjustment(LensAdjustment.LensCorrectionMode, tab === 'profile' ? 'auto' : 'manual');
  };

  const hasProfile = Boolean(adjustments.lensDistortionParams);
  const profileCorrectionsEnabled =
    hasProfile && Boolean(adjustments.lensDistortionEnabled && adjustments.lensVignetteEnabled);

  const handleProfileToggle = (enabled: boolean) => {
    if (enabled && !hasProfile) {
      setIsLensModalOpen(true);
      return;
    }
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      lensDistortionEnabled: enabled,
      lensVignetteEnabled: enabled,
    }));
  };

  const handleTcaToggle = (enabled: boolean) => {
    if (enabled && !hasProfile) {
      setIsLensModalOpen(true);
      return;
    }
    updateAdjustment(LensAdjustment.LensTcaEnabled, enabled);
  };

  return (
    <div className="camera-raw-section-body">
      <AdjustmentSubsection>
        <AdjustmentTabs
          ariaLabel={t('editor.adjustments.sections.optics')}
          onChange={handleTabChange}
          tabs={tabs}
          value={activeTab}
        />
      </AdjustmentSubsection>

      {activeTab === 'profile' ? (
        <>
          <AdjustmentSubsection>
            <Switch
              checked={hasProfile && Boolean(adjustments.lensTcaEnabled)}
              label={t('adjustments.optics.removeChromaticAberration')}
              onChange={handleTcaToggle}
            />
            <Switch
              checked={profileCorrectionsEnabled}
              className="mt-1"
              label={t('adjustments.optics.useProfileCorrections')}
              onChange={handleProfileToggle}
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('adjustments.optics.lensProfile')}>
            <div
              aria-live="polite"
              className="camera-raw-profile-detection"
              data-state={lensProfileStatus}
              role="status"
            >
              {lensProfileStatus === 'detecting' ? (
                <>
                  <Loader2 aria-hidden="true" className="animate-spin" size={13} />
                  <span>{t('modals.lensCorrection.detectingExif')}</span>
                </>
              ) : lensProfileStatus === 'success' && hasProfile ? (
                <>
                  <Check aria-hidden="true" size={13} />
                  <span>{t('modals.lensCorrection.lensFound')}</span>
                </>
              ) : lensProfileStatus === 'not_found' ? (
                <>
                  <TriangleAlert aria-hidden="true" size={13} />
                  <span>{t('modals.lensCorrection.lensProfileNotFound')}</span>
                </>
              ) : (
                <span>{t('modals.lensCorrection.waitingAutoDetect')}</span>
              )}
            </div>
            <dl className="camera-raw-profile-summary">
              <div>
                <dt>{t('modals.lensCorrection.selectManufacturer')}</dt>
                <dd>{adjustments.lensMaker || t('modals.lensCorrection.notFound')}</dd>
              </div>
              <div>
                <dt>{t('modals.lensCorrection.selectLensModel')}</dt>
                <dd>{adjustments.lensModel || t('modals.lensCorrection.notFound')}</dd>
              </div>
            </dl>

            <button className="camera-raw-action-row" onClick={() => setIsLensModalOpen(true)} type="button">
              <Aperture aria-hidden="true" size={16} strokeWidth={1.8} />
              <span className="camera-raw-action-title grow">{t('adjustments.optics.chooseProfile')}</span>
              <ChevronRight aria-hidden="true" size={15} strokeWidth={1.8} />
            </button>
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('modals.lensCorrection.corrections')}>
            <Slider
              disabled={!hasProfile || !adjustments.lensDistortionEnabled}
              label={t('modals.lensCorrection.distortion')}
              max={200}
              min={0}
              defaultValue={100}
              onChange={(event) =>
                updateAdjustment(LensAdjustment.LensDistortionAmount, Number.parseFloat(String(event.target.value)))
              }
              step={1}
              value={adjustments.lensDistortionAmount}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              disabled={!hasProfile || !adjustments.lensVignetteEnabled}
              label={t('modals.lensCorrection.vignetting')}
              max={200}
              min={0}
              defaultValue={100}
              onChange={(event) =>
                updateAdjustment(LensAdjustment.LensVignetteAmount, Number.parseFloat(String(event.target.value)))
              }
              step={1}
              value={adjustments.lensVignetteAmount}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              disabled={!hasProfile || !adjustments.lensTcaEnabled}
              label={t('modals.lensCorrection.chromaticAberration')}
              max={200}
              min={0}
              defaultValue={100}
              onChange={(event) =>
                updateAdjustment(LensAdjustment.LensTcaAmount, Number.parseFloat(String(event.target.value)))
              }
              step={1}
              value={adjustments.lensTcaAmount}
              onDragStateChange={onDragStateChange}
            />
          </AdjustmentSubsection>
        </>
      ) : (
        <DetailsPanel
          adjustments={adjustments}
          setAdjustments={setAdjustments}
          appSettings={appSettings}
          onDragStateChange={onDragStateChange}
          variant="optics"
        />
      )}

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
