import { useEffect, useRef, useState } from 'react';
import type { Adjustments } from '../utils/adjustments';
import type { SelectedImage } from '../components/ui/AppProperties';
import { detectLensProfileFromExif } from '../utils/lensProfiles';

export type LensProfileStatus = 'idle' | 'detecting' | 'success' | 'not_found';

interface UseAutoLensProfileProps {
  adjustments: Adjustments;
  selectedImage: SelectedImage | null;
  setAdjustments: (value: Partial<Adjustments> | ((previous: Adjustments) => Adjustments)) => void;
}

export function useAutoLensProfile({ adjustments, selectedImage, setAdjustments }: UseAutoLensProfileProps) {
  const [status, setStatus] = useState<LensProfileStatus>('idle');
  const lastCompletedRequestRef = useRef<string | null>(null);
  const exif = selectedImage?.exif;
  const exifMaker = exif?.LensMake || exif?.Make || '';
  const exifModel = exif?.LensModel || exif?.LensID || exif?.Lens || '';
  const focalLength = exif?.FocalLength || exif?.FocalLengthIn35mmFilm || '';
  const aperture = exif?.FNumber || exif?.ApertureValue || '';
  const distance = exif?.SubjectDistance || '';
  const detectionKey = selectedImage?.path
    ? [selectedImage.path, exifMaker, exifModel, focalLength, aperture, distance].join('\u0000')
    : null;

  useEffect(() => {
    if (adjustments.lensCorrectionMode !== 'auto') {
      lastCompletedRequestRef.current = null;
      setStatus('idle');
      return;
    }

    if (!selectedImage?.path || !exifModel || !detectionKey) {
      setStatus('idle');
      return;
    }

    if (adjustments.lensMaker && adjustments.lensModel && adjustments.lensDistortionParams) {
      lastCompletedRequestRef.current = detectionKey;
      setStatus('success');
      return;
    }

    // A valid profile may not contain calibration parameters for this focal
    // length/aperture. Remember that completed lookup so the adjustment update
    // does not continuously trigger the same request.
    if (lastCompletedRequestRef.current === detectionKey) return;

    let isCurrentRequest = true;
    setStatus('detecting');

    detectLensProfileFromExif(selectedImage.exif)
      .then((profile) => {
        if (!isCurrentRequest) return;
        lastCompletedRequestRef.current = detectionKey;
        if (!profile) {
          setStatus('not_found');
          return;
        }

        setAdjustments((previous) => {
          if (previous.lensCorrectionMode !== 'auto') return previous;
          if (
            previous.lensMaker === profile.maker &&
            previous.lensModel === profile.model &&
            previous.lensDistortionParams === profile.params
          ) {
            return previous;
          }
          return {
            ...previous,
            lensMaker: profile.maker,
            lensModel: profile.model,
            lensDistortionParams: profile.params,
          };
        });
        setStatus(profile.params ? 'success' : 'not_found');
      })
      .catch((error) => {
        if (!isCurrentRequest) return;
        console.error('Automatic lens profile detection failed:', error);
        setStatus('not_found');
      });

    return () => {
      isCurrentRequest = false;
    };
  }, [
    adjustments.lensCorrectionMode,
    adjustments.lensDistortionParams,
    adjustments.lensMaker,
    adjustments.lensModel,
    detectionKey,
    exifModel,
    selectedImage?.path,
    setAdjustments,
  ]);

  return status;
}
