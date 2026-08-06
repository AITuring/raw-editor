import { useMemo, useState } from 'react';
import { Aperture, Check, ChevronRight, Loader2, TriangleAlert } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  DetailsAdjustment,
  Effect,
  LensAdjustment,
  TransformAdjustment,
  type Adjustments,
} from '../../utils/adjustments';
import type { AppSettings } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import LensCorrectionModal from '../modals/LensCorrectionModal';
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
  lensProfileStatus = 'idle',
  onDragStateChange,
}: OpticsPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state) => state.selectedImage);
  const [isLensModalOpen, setIsLensModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'profile' | 'manual'>('profile');

  const tabs = useMemo(
    () => [
      { id: 'profile' as const, label: t('adjustments.optics.profile') },
      { id: 'manual' as const, label: t('modals.lensCorrection.modeManual') },
    ],
    [t],
  );

  const updateAdjustment = (key: string, value: boolean | number | string) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: value }));
  };

  const handleTabChange = (tab: 'profile' | 'manual') => {
    setActiveTab(tab);
  };

  const handleManualAdjustment = (key: DetailsAdjustment | Effect | TransformAdjustment, value: string | number) => {
    updateAdjustment(key, Number(value));
  };

  const hasProfile = Boolean(adjustments.lensDistortionParams);
  const profileCorrectionsEnabled =
    hasProfile && Boolean(adjustments.lensDistortionEnabled && adjustments.lensVignetteEnabled);
  const profileDetectionTone =
    lensProfileStatus === 'success' && hasProfile
      ? 'success'
      : lensProfileStatus === 'not_found'
        ? 'warning'
        : 'processing';

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
    <div className="camera-raw-section-body camera-raw-optics">
      <div className="camera-raw-optics-tabs">
        <AdjustmentTabs
          ariaLabel={t('editor.adjustments.sections.optics')}
          onChange={handleTabChange}
          tabs={tabs}
          value={activeTab}
        />
      </div>

      {activeTab === 'profile' ? (
        <div className="camera-raw-optics-pane" data-tab="profile">
          <AdjustmentSubsection className="camera-raw-optics-toggles">
            <Switch
              checked={hasProfile && Boolean(adjustments.lensTcaEnabled)}
              label={t('adjustments.optics.removeChromaticAberration')}
              onChange={handleTcaToggle}
            />
            <Switch
              checked={profileCorrectionsEnabled}
              label={t('adjustments.optics.useProfileCorrections')}
              onChange={handleProfileToggle}
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('adjustments.optics.lensProfile')}>
            <div
              aria-live="polite"
              className="camera-raw-profile-detection semantic-status semantic-status--badge"
              data-state={lensProfileStatus}
              data-tone={profileDetectionTone}
              role="status"
            >
              {lensProfileStatus === 'detecting' ? (
                <>
                  <Loader2 aria-hidden="true" className="animate-spin" size={12} />
                  <span>{t('modals.lensCorrection.detectingExif')}</span>
                </>
              ) : lensProfileStatus === 'success' && hasProfile ? (
                <>
                  <Check aria-hidden="true" size={12} />
                  <span>{t('modals.lensCorrection.lensFound')}</span>
                </>
              ) : lensProfileStatus === 'not_found' ? (
                <>
                  <TriangleAlert aria-hidden="true" size={12} />
                  <span>{t('modals.lensCorrection.lensProfileNotFound')}</span>
                </>
              ) : (
                <>
                  <span aria-hidden="true" className="semantic-status__dot" />
                  <span>{t('modals.lensCorrection.waitingAutoDetect')}</span>
                </>
              )}
            </div>
            <dl className="camera-raw-profile-summary">
              <div>
                <dt>{t('modals.lensCorrection.selectManufacturer')}</dt>
                <dd className="semantic-result" data-state={adjustments.lensMaker ? 'available' : 'missing'}>
                  {adjustments.lensMaker || t('modals.lensCorrection.notFound')}
                </dd>
              </div>
              <div>
                <dt>{t('modals.lensCorrection.selectLensModel')}</dt>
                <dd className="semantic-result" data-state={adjustments.lensModel ? 'available' : 'missing'}>
                  {adjustments.lensModel || t('modals.lensCorrection.notFound')}
                </dd>
              </div>
            </dl>

            <button className="camera-raw-action-row" onClick={() => setIsLensModalOpen(true)} type="button">
              <Aperture aria-hidden="true" size={14} strokeWidth={1.8} />
              <span className="camera-raw-action-title grow">{t('adjustments.optics.chooseProfile')}</span>
              <ChevronRight aria-hidden="true" size={14} strokeWidth={1.8} />
            </button>
          </AdjustmentSubsection>

          {hasProfile && (
            <AdjustmentSubsection title={t('modals.lensCorrection.corrections')}>
              <Slider
                disabled={!adjustments.lensDistortionEnabled}
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
                disabled={!adjustments.lensVignetteEnabled}
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
                disabled={!adjustments.lensTcaEnabled}
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
          )}
        </div>
      ) : (
        <div className="camera-raw-optics-pane" data-tab="manual">
          <AdjustmentSubsection title={t('modals.lensCorrection.distortion')}>
            <Slider
              label={t('modals.lensCorrection.amount')}
              max={100}
              min={-100}
              defaultValue={0}
              onChange={(event) => handleManualAdjustment(TransformAdjustment.TransformDistortion, event.target.value)}
              step={1}
              value={adjustments.transformDistortion}
              onDragStateChange={onDragStateChange}
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('adjustments.optics.defringe')}>
            <Slider
              label={t('adjustments.details.redCyan')}
              max={100}
              min={-100}
              defaultValue={0}
              onChange={(event) =>
                handleManualAdjustment(DetailsAdjustment.ChromaticAberrationRedCyan, event.target.value)
              }
              step={1}
              value={adjustments.chromaticAberrationRedCyan}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('adjustments.details.blueYellow')}
              max={100}
              min={-100}
              defaultValue={0}
              onChange={(event) =>
                handleManualAdjustment(DetailsAdjustment.ChromaticAberrationBlueYellow, event.target.value)
              }
              step={1}
              value={adjustments.chromaticAberrationBlueYellow}
              onDragStateChange={onDragStateChange}
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('adjustments.optics.lensVignetting')}>
            <Slider
              label={t('adjustments.effects.amount')}
              max={100}
              min={-100}
              defaultValue={0}
              onChange={(event) => handleManualAdjustment(Effect.VignetteAmount, event.target.value)}
              step={1}
              value={adjustments.vignetteAmount}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('adjustments.effects.midpoint')}
              max={100}
              min={0}
              defaultValue={50}
              fillOrigin="min"
              onChange={(event) => handleManualAdjustment(Effect.VignetteMidpoint, event.target.value)}
              step={1}
              value={adjustments.vignetteMidpoint}
              onDragStateChange={onDragStateChange}
            />
          </AdjustmentSubsection>
        </div>
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
