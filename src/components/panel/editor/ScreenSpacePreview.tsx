import { forwardRef, useCallback, useEffect, useRef, useState } from 'react';

import type { InteractivePatch } from '../../../store/useEditorStore';

interface ScreenSpacePreviewProps {
  finalPreviewUrl: string | null;
  hidden: boolean;
  imagePath: string;
  interactivePatch: InteractivePatch | null;
  isMaxZoom: boolean;
  isSliderDragging: boolean;
  onProcessedFrameReady(isViewportPatch: boolean): void;
  showOriginal: boolean;
  thumbnailUrl: string;
  transformedOriginalUrl: string | null;
}

/**
 * Renders the processed bitmap outside the editor's transformed interaction
 * layer. WebKit otherwise flattens that ancestor at its fit-to-window size and
 * magnifies the cached texture, which discards source detail at 100% zoom.
 * Geometry is written imperatively by Editor so a settled preview has no CSS
 * scale transform at all.
 */
const ScreenSpacePreview = forwardRef<HTMLDivElement, ScreenSpacePreviewProps>(
  (
    {
      finalPreviewUrl,
      hidden,
      imagePath,
      interactivePatch,
      isMaxZoom,
      isSliderDragging,
      onProcessedFrameReady,
      showOriginal,
      thumbnailUrl,
      transformedOriginalUrl,
    },
    ref,
  ) => {
    const [displayState, setDisplayState] = useState({
      base: finalPreviewUrl || thumbnailUrl,
      fade: null as string | null,
    });
    const [isFadingIn, setIsFadingIn] = useState(false);
    const [originalLoaded, setOriginalLoaded] = useState(false);
    const [loadedPatch, setLoadedPatch] = useState<InteractivePatch | null>(null);
    const previousImageRef = useRef(imagePath);
    const latestPatchUrlRef = useRef(interactivePatch?.url ?? null);
    const patchReadyNotificationRef = useRef<string | null>(null);
    latestPatchUrlRef.current = interactivePatch?.url ?? null;

    useEffect(() => {
      const newSource = finalPreviewUrl || thumbnailUrl;
      const isNewImage = previousImageRef.current !== imagePath;

      if (isNewImage) {
        previousImageRef.current = imagePath;
        patchReadyNotificationRef.current = null;
        setLoadedPatch(null);
        setDisplayState({ base: newSource, fade: null });
        setIsFadingIn(false);
        return;
      }

      if (isSliderDragging || !displayState.base || displayState.base === newSource) {
        setDisplayState({ base: newSource, fade: null });
        setIsFadingIn(false);
        return;
      }

      setDisplayState((previous) => ({ base: previous.base, fade: newSource }));
      setIsFadingIn(false);

      let secondFrame = 0;
      const firstFrame = requestAnimationFrame(() => {
        secondFrame = requestAnimationFrame(() => setIsFadingIn(true));
      });
      const timer = window.setTimeout(() => {
        setDisplayState({ base: newSource, fade: null });
        setIsFadingIn(false);
      }, 150);

      return () => {
        cancelAnimationFrame(firstFrame);
        cancelAnimationFrame(secondFrame);
        window.clearTimeout(timer);
      };
    }, [finalPreviewUrl, imagePath, isSliderDragging, thumbnailUrl]);

    useEffect(() => {
      if (!transformedOriginalUrl) {
        setOriginalLoaded(false);
        return;
      }

      const image = new Image();
      image.onload = () => setOriginalLoaded(true);
      image.onerror = () => setOriginalLoaded(false);
      image.src = transformedOriginalUrl;
      if (image.complete && image.naturalWidth > 0) {
        setOriginalLoaded(true);
      } else {
        setOriginalLoaded(false);
      }

      return () => {
        image.onload = null;
        image.onerror = null;
      };
    }, [transformedOriginalUrl]);

    const currentTarget = finalPreviewUrl || thumbnailUrl;
    const baseIsReady = displayState.base === currentTarget && !displayState.fade;
    const pendingPatch = interactivePatch && interactivePatch.url !== loadedPatch?.url ? interactivePatch : null;

    useEffect(() => {
      if (baseIsReady && !interactivePatch) {
        setLoadedPatch(null);
      }
    }, [baseIsReady, interactivePatch]);

    useEffect(() => {
      if (loadedPatch?.url && patchReadyNotificationRef.current === loadedPatch.url) {
        patchReadyNotificationRef.current = null;
        onProcessedFrameReady(true);
      }
    }, [loadedPatch, onProcessedFrameReady]);

    const imageRendering = isMaxZoom ? 'pixelated' : 'auto';
    const handleProcessedFrameLoad = useCallback(
      (event: React.SyntheticEvent<HTMLImageElement>, isViewportPatch: boolean) => {
        const image = event.currentTarget;
        const notify = () => onProcessedFrameReady(isViewportPatch);
        if (typeof image.decode === 'function') {
          void image.decode().then(notify, notify);
        } else {
          notify();
        }
      },
      [onProcessedFrameReady],
    );

    const handlePendingPatchLoad = useCallback(
      (event: React.SyntheticEvent<HTMLImageElement>, patch: InteractivePatch) => {
        const image = event.currentTarget;
        const promote = () => {
          if (latestPatchUrlRef.current !== patch.url) return;
          patchReadyNotificationRef.current = patch.url;
          setLoadedPatch(patch);
        };
        if (typeof image.decode === 'function') {
          void image.decode().then(promote, promote);
        } else {
          promote();
        }
      },
      [],
    );

    return (
      <div
        aria-hidden="true"
        className="absolute z-0 pointer-events-none overflow-hidden"
        ref={ref}
        style={{ opacity: hidden ? 0 : 1 }}
      >
        {displayState.base && (
          <img
            alt=""
            draggable={false}
            onLoad={(event) => handleProcessedFrameLoad(event, false)}
            src={displayState.base}
            style={{
              height: '100%',
              imageRendering,
              inset: 0,
              objectFit: 'fill',
              position: 'absolute',
              width: '100%',
            }}
          />
        )}

        {displayState.fade && (
          <img
            alt=""
            draggable={false}
            onLoad={(event) => handleProcessedFrameLoad(event, false)}
            src={displayState.fade}
            style={{
              height: '100%',
              imageRendering,
              inset: 0,
              objectFit: 'fill',
              opacity: isFadingIn ? 1 : 0,
              position: 'absolute',
              transition: 'opacity 150ms ease-in-out',
              width: '100%',
            }}
          />
        )}

        {loadedPatch && (
          <img
            alt=""
            draggable={false}
            src={loadedPatch.url}
            style={{
              height: `${loadedPatch.normH * 100}%`,
              imageRendering,
              left: `${loadedPatch.normX * 100}%`,
              objectFit: 'fill',
              position: 'absolute',
              top: `${loadedPatch.normY * 100}%`,
              width: `${loadedPatch.normW * 100}%`,
            }}
          />
        )}

        {pendingPatch && (
          <img
            alt=""
            aria-hidden="true"
            draggable={false}
            onLoad={(event) => handlePendingPatchLoad(event, pendingPatch)}
            src={pendingPatch.url}
            style={{ height: 1, opacity: 0, position: 'absolute', width: 1 }}
          />
        )}

        {transformedOriginalUrl && (
          <img
            alt="Original"
            draggable={false}
            src={transformedOriginalUrl}
            style={{
              height: '100%',
              imageRendering,
              inset: 0,
              objectFit: 'fill',
              opacity: showOriginal && originalLoaded ? 1 : 0,
              position: 'absolute',
              transition: originalLoaded ? 'opacity 150ms ease-in-out' : 'none',
              width: '100%',
            }}
          />
        )}
      </div>
    );
  },
);

ScreenSpacePreview.displayName = 'ScreenSpacePreview';

export default ScreenSpacePreview;
