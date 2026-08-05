import { ChartArea, ChevronUp } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useShallow } from 'zustand/react/shallow';

import { BASIC_MODE } from '../../basic/runtime';
import { useEditorActions } from '../../hooks/useEditorActions';
import { useWaveformControls } from '../../hooks/useWaveformControls';
import { useEditorStore } from '../../store/useEditorStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { Adjustments, DisplayMode } from '../../utils/adjustments';
import Waveform from './editor/Waveform';

export default function DevelopHistogram() {
  const { t } = useTranslation();
  const theme = useSettingsStore((state) => state.theme);
  const { setAdjustments } = useEditorActions();
  const { setActiveWaveformChannel } = useWaveformControls();
  const { activeWaveformChannel, adjustments, histogram, isWaveformVisible, selectedImage, setEditor, waveform } =
    useEditorStore(
      useShallow((state) => ({
        activeWaveformChannel: state.activeWaveformChannel,
        adjustments: state.adjustments,
        histogram: state.histogram,
        isWaveformVisible: state.isWaveformVisible,
        selectedImage: state.selectedImage,
        setEditor: state.setEditor,
        waveform: state.waveform,
      })),
    );

  const exif = selectedImage?.exif || {};
  const captureDetails = [
    exif.PhotographicSensitivity || exif.ISOSpeed
      ? `ISO ${exif.PhotographicSensitivity || exif.ISOSpeed}`
      : null,
    exif.LensSpecification || exif.LensModel || (exif.FocalLength ? `${exif.FocalLength} mm` : null),
    exif.FNumber || exif.ApertureValue || null,
    exif.ExposureTime || null,
  ].filter(Boolean) as string[];

  const toggleLabel = t('editor.adjustments.tooltips.toggleAnalytics');

  if (!isWaveformVisible) {
    return (
      <button
        aria-label={toggleLabel}
        className="develop-histogram-collapsed"
        data-tooltip={toggleLabel}
        onClick={() => setEditor({ isWaveformVisible: true })}
        type="button"
      >
        <ChartArea aria-hidden="true" size={15} strokeWidth={1.7} />
      </button>
    );
  }

  return (
    <div className="develop-histogram">
      <div className="develop-histogram-graph">
        <Waveform
          displayMode={BASIC_MODE ? DisplayMode.Histogram : activeWaveformChannel || DisplayMode.Histogram}
          histogram={histogram}
          histogramOnly={BASIC_MODE}
          onToggleClipping={() => {
            setAdjustments((previous: Adjustments) => ({
              ...previous,
              showClipping: !previous.showClipping,
            }));
          }}
          setDisplayMode={setActiveWaveformChannel}
          showClipping={adjustments.showClipping || false}
          theme={theme}
          waveformData={waveform || null}
        />
      </div>
      <div className="develop-capture-strip">
        {captureDetails.length > 0 ? (
          captureDetails.map((detail, index) => (
            <span className="truncate" key={`${detail}-${index}`} title={detail}>
              {detail}
            </span>
          ))
        ) : (
          <span aria-hidden="true">—</span>
        )}
      </div>
      <button
        aria-label={toggleLabel}
        className="develop-histogram-collapse"
        data-tooltip={toggleLabel}
        onClick={() => setEditor({ isWaveformVisible: false })}
        type="button"
      >
        <ChevronUp aria-hidden="true" size={13} strokeWidth={1.8} />
      </button>
    </div>
  );
}
