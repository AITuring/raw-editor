import React, { useCallback, useEffect, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import debounce from 'lodash.debounce';
import { useEditorStore } from '../store/useEditorStore';
import { useUIStore } from '../store/useUIStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { useLibraryStore } from '../store/useLibraryStore';
import { Adjustments, COPYABLE_ADJUSTMENT_KEYS } from '../utils/adjustments';
import { Invokes, Panel } from '../components/ui/AppProperties';
import { debouncedSave } from './useEditorActions';
import { globalImageCache } from '../utils/ImageLRUCache';
import { parsePreviewResponse } from '../utils/previewProtocol';
import { createImageObjectUrl } from '../utils/imageObjectUrl';
import { calculatePreviewTargetResolution, resolvePreviewRenderPlan } from '../utils/previewResolution';
import { BASIC_MODE } from '../basic/runtime';
import { createLatestOnlyAsyncQueue } from '../utils/latestOnlyAsyncQueue';

interface UncroppedPreviewRequest {
  adjustments: Adjustments;
  key: string;
  path: string;
}

export function useImageProcessing(
  transformWrapperRef: any,
  prevAdjustmentsRef: React.RefObject<any>,
  renderRefs: {
    previewJobIdRef: React.RefObject<number>;
    latestRenderedJobIdRef: React.RefObject<number>;
    currentResRef: React.RefObject<number>;
  },
) {
  const { previewJobIdRef, latestRenderedJobIdRef, currentResRef } = renderRefs;

  const selectedImage = useEditorStore((state) => state.selectedImage);
  const adjustments = useEditorStore((state) => state.adjustments);
  const previewOverride = useEditorStore((state) => state.previewOverride);
  const isWaveformVisible = useEditorStore((state) => state.isWaveformVisible);
  const activeWaveformChannel = useEditorStore((state) => state.activeWaveformChannel);
  const displaySize = useEditorStore((state) => state.displaySize);
  const viewportRevision = useEditorStore((state) => state.viewportRevision);
  const baseRenderSize = useEditorStore((state) => state.baseRenderSize);
  const originalSize = useEditorStore((state) => state.originalSize);
  const showOriginal = useEditorStore((state) => state.showOriginal);
  const isSliderDragging = useEditorStore((state) => state.isSliderDragging);
  const transformedOriginalUrl = useEditorStore((state) => state.transformedOriginalUrl);
  const setEditor = useEditorStore((state) => state.setEditor);

  const activeView = useUIStore((state) => state.activeView);
  const activeRightPanel = useUIStore((state) => state.activeRightPanel);
  const appSettings = useSettingsStore((state) => state.appSettings);
  const multiSelectedPaths = useLibraryStore((state) => state.multiSelectedPaths);

  const inFlightCountRef = useRef(0);
  const pendingApplyRef = useRef<{ adjustments: Adjustments; targetRes?: number } | null>(null);
  const currentOriginalResRef = useRef<number>(0);
  const originalPreviewJobIdRef = useRef(0);
  const lastViewportRequestKeyRef = useRef<string | null>(null);
  const dragIdleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeWaveformChannelRef = useRef(activeWaveformChannel);
  activeWaveformChannelRef.current = activeWaveformChannel;

  const selectedImagePathRef = useRef<string | null>(null);
  useEffect(() => {
    selectedImagePathRef.current = selectedImage?.path ?? null;
  }, [selectedImage?.path]);

  const uncroppedPreviewAdjustmentsRef = useRef(adjustments);
  uncroppedPreviewAdjustmentsRef.current = adjustments;

  const uncroppedPreviewKey = useMemo(() => {
    const previewAdjustments: Partial<Adjustments> = { ...adjustments };
    delete previewAdjustments.crop;
    delete previewAdjustments.aspectRatio;
    delete previewAdjustments.constrainCrop;
    delete previewAdjustments.rotation;
    return JSON.stringify(previewAdjustments);
  }, [adjustments]);

  const geometricAdjustmentsKey = useMemo(() => {
    if (!adjustments) return '';
    const {
      crop,
      rotation,
      flipHorizontal,
      flipVertical,
      orientationSteps,
      transformDistortion,
      transformProjection,
      transformVertical,
      transformHorizontal,
      transformRotate,
      transformAspect,
      transformScale,
      transformXOffset,
      transformYOffset,
      lensDistortionAmount,
      lensVignetteAmount,
      lensTcaAmount,
      lensDistortionEnabled,
      lensTcaEnabled,
      lensVignetteEnabled,
      lensDistortionParams,
    } = adjustments;
    return JSON.stringify({
      crop,
      rotation,
      flipHorizontal,
      flipVertical,
      orientationSteps,
      transformDistortion,
      transformProjection,
      transformVertical,
      transformHorizontal,
      transformRotate,
      transformAspect,
      transformScale,
      transformXOffset,
      transformYOffset,
      lensDistortionAmount,
      lensVignetteAmount,
      lensTcaAmount,
      lensDistortionEnabled,
      lensTcaEnabled,
      lensVignetteEnabled,
      lensDistortionParams,
    });
  }, [
    adjustments?.crop,
    adjustments?.rotation,
    adjustments?.flipHorizontal,
    adjustments?.flipVertical,
    adjustments?.orientationSteps,
    adjustments?.transformDistortion,
    adjustments?.transformProjection,
    adjustments?.transformVertical,
    adjustments?.transformHorizontal,
    adjustments?.transformRotate,
    adjustments?.transformAspect,
    adjustments?.transformScale,
    adjustments?.transformXOffset,
    adjustments?.transformYOffset,
    adjustments?.lensDistortionAmount,
    adjustments?.lensVignetteAmount,
    adjustments?.lensTcaAmount,
    adjustments?.lensDistortionEnabled,
    adjustments?.lensTcaEnabled,
    adjustments?.lensVignetteEnabled,
    adjustments?.lensDistortionParams,
  ]);

  const calculateROI = useCallback(() => {
    if (!transformWrapperRef.current) return null;
    const state = transformWrapperRef.current.instance.transformState;
    if (!state) return null;

    if (!baseRenderSize) return null;

    const { scale, positionX, positionY } = state;
    const { width: baseW, height: baseH, offsetX, offsetY, containerWidth, containerHeight } = baseRenderSize;

    if (!baseW || !baseH || !containerWidth || !containerHeight) return null;
    if (scale <= 1.01) return null;

    const overscanScreenPixels = 96;
    const paddingX = overscanScreenPixels / (baseW * scale);
    const paddingY = overscanScreenPixels / (baseH * scale);

    const visibleLeft = -positionX / scale;
    const visibleTop = -positionY / scale;
    const visibleRight = visibleLeft + containerWidth / scale;
    const visibleBottom = visibleTop + containerHeight / scale;

    const imgLeft = offsetX;
    const imgTop = offsetY;
    const imgRight = offsetX + baseW;
    const imgBottom = offsetY + baseH;

    const intersectLeft = Math.max(visibleLeft, imgLeft);
    const intersectTop = Math.max(visibleTop, imgTop);
    const intersectRight = Math.min(visibleRight, imgRight);
    const intersectBottom = Math.min(visibleBottom, imgBottom);

    if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
      return null;
    }

    const roiX = (intersectLeft - imgLeft) / baseW;
    const roiY = (intersectTop - imgTop) / baseH;
    const roiW = (intersectRight - intersectLeft) / baseW;
    const roiH = (intersectBottom - intersectTop) / baseH;

    const clampedX = Math.max(0, roiX - paddingX);
    const clampedY = Math.max(0, roiY - paddingY);
    const clampedRight = Math.min(1, roiX + roiW + paddingX);
    const clampedBottom = Math.min(1, roiY + roiH + paddingY);
    const clampedW = clampedRight - clampedX;
    const clampedH = clampedBottom - clampedY;

    if (clampedW > 0.999 && clampedH > 0.999) return null;

    return [clampedX, clampedY, clampedW, clampedH] as [number, number, number, number];
  }, [baseRenderSize, transformWrapperRef]);

  const executeApplyAdjustments = useCallback(
    async (currentAdjustments: Adjustments, dragging: boolean = false, targetRes?: number) => {
      const currentPath = selectedImage?.path;
      if (!currentPath) return;

      const payload = structuredClone(currentAdjustments);
      const { patchesSentToBackend } = useEditorStore.getState();
      const newlySentPatches = new Set<string>();

      const processSubMasks = (subMasks: any[]) => {
        if (!Array.isArray(subMasks)) return;
        subMasks.forEach((sm: any) => {
          if (sm.id && sm.parameters) {
            const keys = ['mask_data_base64', 'maskDataBase64'];
            let foundMaskData = false;

            for (const key of keys) {
              if (sm.parameters[key] !== undefined && sm.parameters[key] !== null) {
                foundMaskData = true;
                if (patchesSentToBackend.has(sm.id)) {
                  sm.parameters[key] = null;
                }
              }
            }
            if (foundMaskData && !patchesSentToBackend.has(sm.id)) {
              newlySentPatches.add(sm.id);
            }
          }
        });
      };

      if (payload.aiPatches && Array.isArray(payload.aiPatches)) {
        payload.aiPatches.forEach((p: any) => {
          if (p.id && p.patchData && !p.isLoading) {
            if (patchesSentToBackend.has(p.id)) {
              p.patchData = null;
            } else {
              newlySentPatches.add(p.id);
            }
          }
          if (p.subMasks) processSubMasks(p.subMasks);
        });
      }

      if (payload.masks && Array.isArray(payload.masks)) {
        payload.masks.forEach((container: any) => {
          if (container.subMasks) processSubMasks(container.subMasks);
        });
      }

      const jobId = ++previewJobIdRef.current;
      const roi = calculateROI();
      const renderPlan = resolvePreviewRenderPlan({
        requestedResolution: targetRes ?? appSettings?.editorPreviewResolution ?? 1920,
        originalSize,
        isInteractive: dragging,
        hasViewportRoi: roi !== null,
        editorPreviewResolution: appSettings?.editorPreviewResolution,
        livePreviewQuality: appSettings?.livePreviewQuality,
      });

      try {
        const buffer: ArrayBuffer = await invoke(Invokes.ApplyAdjustments, {
          jsAdjustments: payload,
          isInteractive: dragging,
          targetResolution: renderPlan.targetResolution,
          renderTier: renderPlan.tier,
          roi: roi || null,
          computeWaveform: !!isWaveformVisible && !BASIC_MODE,
          activeWaveformChannel: BASIC_MODE ? null : activeWaveformChannelRef.current || null,
          // The native display surface is still available in full/debug mode.
          // Basic mode keeps the processed JPEG in the WebView so a failed
          // native-surface handoff can never leave the editor blank.
          preferNativeDisplay: !BASIC_MODE,
        });

        if (newlySentPatches.size > 0) {
          newlySentPatches.forEach((id) => patchesSentToBackend.add(id));
        }

        if (currentPath !== selectedImagePathRef.current) return;

        if (buffer && buffer.byteLength > 0 && jobId >= latestRenderedJobIdRef.current) {
          latestRenderedJobIdRef.current = jobId;

          const response = parsePreviewResponse(buffer);
          if (response.kind === 'wgpu') {
            setEditor((state) => {
              if (state.interactivePatch && state.interactivePatch.url) URL.revokeObjectURL(state.interactivePatch.url);
              return { interactivePatch: null };
            });
            return;
          }

          if (response.kind === 'patch') {
            const blob = new Blob([response.imageBuffer], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);

            setEditor((state) => {
              const previousPatchUrl = state.interactivePatch?.url;
              if (previousPatchUrl) setTimeout(() => URL.revokeObjectURL(previousPatchUrl), 100);
              return {
                interactivePatch: {
                  url,
                  normX: response.normX,
                  normY: response.normY,
                  normW: response.normW,
                  normH: response.normH,
                },
              };
            });
          } else {
            const blob = new Blob([response.imageBuffer], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);

            if (currentPath !== selectedImagePathRef.current || jobId < latestRenderedJobIdRef.current) {
              URL.revokeObjectURL(url);
              return;
            }

            setEditor((state) => {
              const prevUrl = state.finalPreviewUrl;
              if (prevUrl && prevUrl.startsWith('blob:') && !globalImageCache.isProtected(prevUrl)) {
                setTimeout(() => {
                  if (!globalImageCache.isProtected(prevUrl)) {
                    URL.revokeObjectURL(prevUrl);
                  }
                }, 250);
              }
              return { finalPreviewUrl: url };
            });

            setEditor((state) => {
              const previousPatchUrl = state.interactivePatch?.url;
              if (previousPatchUrl) setTimeout(() => URL.revokeObjectURL(previousPatchUrl), 500);
              return { interactivePatch: null };
            });
          }
        }
      } catch (err) {
        if (err !== 'Superseded or worker failed') {
          console.error('Failed to apply adjustments:', err);
        }
        if (!dragging) {
          setEditor((state) => {
            if (state.interactivePatch && state.interactivePatch.url) URL.revokeObjectURL(state.interactivePatch.url);
            return { interactivePatch: null };
          });
        }
      }
    },
    [
      selectedImage?.path,
      calculateROI,
      isWaveformVisible,
      setEditor,
      previewJobIdRef,
      latestRenderedJobIdRef,
      appSettings?.editorPreviewResolution,
      appSettings?.livePreviewQuality,
      originalSize,
    ],
  );

  const flushPipeline = useCallback(() => {
    if (inFlightCountRef.current >= 3) return;
    if (!pendingApplyRef.current) return;

    const { adjustments, targetRes } = pendingApplyRef.current;
    pendingApplyRef.current = null;

    inFlightCountRef.current += 1;

    executeApplyAdjustments(adjustments, true, targetRes).finally(() => {
      inFlightCountRef.current -= 1;
      if (pendingApplyRef.current) {
        requestAnimationFrame(() => flushPipeline());
      }
    });
  }, [executeApplyAdjustments]);

  const applyAdjustments = useCallback(
    (currentAdjustments: Adjustments, dragging: boolean = false, targetRes?: number) => {
      if (!selectedImage?.isReady) return;

      if (dragging) {
        pendingApplyRef.current = { adjustments: currentAdjustments, targetRes };
        flushPipeline();
      } else {
        pendingApplyRef.current = null;
        executeApplyAdjustments(currentAdjustments, false, targetRes);
      }
    },
    [selectedImage?.isReady, flushPipeline, executeApplyAdjustments],
  );

  const uncroppedPreviewQueue = useMemo(
    () =>
      createLatestOnlyAsyncQueue<UncroppedPreviewRequest, ArrayBuffer>({
        execute: (request) =>
          invoke(Invokes.GenerateUncroppedPreview, {
            jsAdjustments: request.adjustments,
          }),
        getKey: (request) => request.key,
        onBusyChange: (busy) => setEditor({ isCropPreviewUpdating: busy }),
        onError: (error, request) => {
          if (request.path === selectedImagePathRef.current) {
            console.error('Failed to generate uncropped preview:', error);
          }
        },
        onResult: (buffer, request) => {
          if (request.path !== selectedImagePathRef.current) return;
          const url = createImageObjectUrl(buffer, 'image/jpeg');
          if (!url) {
            console.error('Failed to generate uncropped preview: native command returned no image data');
            return;
          }
          setEditor({ uncroppedAdjustedPreviewUrl: url });
        },
      }),
    [setEditor],
  );

  useEffect(() => () => uncroppedPreviewQueue.dispose(), [uncroppedPreviewQueue]);

  const generateUncroppedPreview = useCallback(() => {
    const currentPath = selectedImage?.path;
    if (!selectedImage?.isReady || !currentPath) return;
    uncroppedPreviewQueue.submit({
      adjustments: uncroppedPreviewAdjustmentsRef.current,
      key: `${currentPath}:${uncroppedPreviewKey}`,
      path: currentPath,
    });
  }, [selectedImage?.isReady, selectedImage?.path, uncroppedPreviewKey, uncroppedPreviewQueue]);

  const calculateTargetRes = useCallback(() => {
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    return calculatePreviewTargetResolution({
      displaySize,
      baseRenderSize,
      originalSize,
      editorPreviewResolution: appSettings?.editorPreviewResolution,
      enableZoomHifi: appSettings?.enableZoomHifi,
      useFullDpiRendering: appSettings?.useFullDpiRendering,
      highResZoomMultiplier: appSettings?.highResZoomMultiplier,
      devicePixelRatio: dpr,
    });
  }, [
    appSettings?.enableZoomHifi,
    appSettings?.editorPreviewResolution,
    appSettings?.highResZoomMultiplier,
    appSettings?.useFullDpiRendering,
    displaySize.width,
    displaySize.height,
    baseRenderSize.width,
    baseRenderSize.height,
    originalSize,
  ]);

  const requestHiFiZoom = useMemo(
    () =>
      debounce((currentAdjustments: Adjustments, targetRes: number) => {
        currentResRef.current = targetRes;
        applyAdjustments(currentAdjustments, false, targetRes);
      }, 50),
    [applyAdjustments, currentResRef],
  );

  const generateOriginalPreview = useCallback(
    async (currentAdjustments: Adjustments, targetRes: number) => {
      const currentPath = selectedImagePathRef.current;
      if (!currentPath) return false;

      const jobId = ++originalPreviewJobIdRef.current;
      const buffer: ArrayBuffer = await invoke(Invokes.GenerateOriginalTransformedPreview, {
        jsAdjustments: currentAdjustments,
        targetResolution: targetRes,
      });
      if (jobId !== originalPreviewJobIdRef.current || currentPath !== selectedImagePathRef.current) {
        return false;
      }

      const url = createImageObjectUrl(buffer, 'image/jpeg');
      if (!url) throw new Error('Original preview returned no image data');
      currentOriginalResRef.current = targetRes;
      setEditor({ transformedOriginalUrl: url });
      return true;
    },
    [setEditor],
  );

  const requestHiFiOriginalZoom = useMemo(
    () =>
      debounce(async (currentAdjustments: Adjustments, targetRes: number) => {
        if (targetRes <= currentOriginalResRef.current) return;

        try {
          await generateOriginalPreview(currentAdjustments, targetRes);
        } catch (e) {
          console.error('Failed to generate hi-fi original preview:', e);
        }
      }, 200),
    [generateOriginalPreview],
  );

  useEffect(() => {
    if (activeView === 'editor' && activeRightPanel === Panel.Crop && selectedImage?.isReady) {
      generateUncroppedPreview();
    } else {
      uncroppedPreviewQueue.cancel();
      setEditor({ uncroppedAdjustedPreviewUrl: null, isCropPreviewUpdating: false });
    }
  }, [
    activeView,
    activeRightPanel,
    selectedImage?.isReady,
    generateUncroppedPreview,
    setEditor,
    uncroppedPreviewQueue,
  ]);

  useEffect(() => {
    if (activeView === 'editor' && selectedImage?.isReady && displaySize.width > 0 && !isSliderDragging) {
      let baseRes = calculateTargetRes();
      if (originalSize.width > 0 && originalSize.height > 0) {
        const maxRes = Math.max(originalSize.width, originalSize.height);
        if (baseRes > maxRes) baseRes = maxRes;
      }
      const finalRes = Math.round(baseRes);

      const roi = calculateROI();
      const roiKey = roi ? roi.map((value) => value.toFixed(5)).join(':') : 'full';
      const requestKey = `${selectedImage.path}:${finalRes}:${roiKey}`;
      if (requestKey !== lastViewportRequestKeyRef.current) {
        lastViewportRequestKeyRef.current = requestKey;
        requestHiFiZoom(adjustments, finalRes);
      }
    }
    return () => {
      requestHiFiZoom.cancel();
    };
  }, [
    activeView,
    displaySize.width,
    displaySize.height,
    viewportRevision,
    calculateTargetRes,
    calculateROI,
    selectedImage?.isReady,
    selectedImage?.path,
    isSliderDragging,
    requestHiFiZoom,
    originalSize,
  ]);

  useEffect(() => {
    if (!selectedImage?.isReady) return;

    if (dragIdleTimer.current) clearTimeout(dragIdleTimer.current);

    const targetRes = calculateTargetRes();
    const renderAdjustments = previewOverride ?? adjustments;

    // The crop workspace renders the uncropped geometry preview through its own
    // latest-only queue. Starting the hidden main preview for every pointer move
    // would make both native jobs compete and is especially noticeable on large
    // RAW files. A final main preview is still rendered when the pointer is released.
    if (activeView === 'editor' && activeRightPanel === Panel.Crop && isSliderDragging) {
      return;
    }

    if (activeView !== 'editor') {
      if (isSliderDragging) return;
    }

    if (isSliderDragging) {
      if (appSettings?.enableLivePreviews !== false) {
        applyAdjustments(renderAdjustments, true, targetRes);
      }
    } else {
      dragIdleTimer.current = setTimeout(() => {
        currentResRef.current = targetRes;

        applyAdjustments(renderAdjustments, false, targetRes);

        if (previewOverride) return;

        debouncedSave(selectedImage.path, adjustments);

        const otherPaths = multiSelectedPaths.filter((p) => p !== selectedImage.path);
        if (appSettings?.copyPasteSettings?.autoSync && otherPaths.length > 0) {
          const prev = prevAdjustmentsRef.current;
          if (prev && prev.path === selectedImage.path) {
            const delta: Partial<Adjustments> = {};
            const includedKeys = appSettings?.copyPasteSettings?.includedAdjustments || COPYABLE_ADJUSTMENT_KEYS;
            for (const key of Object.keys(adjustments) as Array<keyof Adjustments>) {
              if (includedKeys.includes(key as string)) {
                if (JSON.stringify(adjustments[key]) !== JSON.stringify(prev.adjustments[key])) {
                  (delta as any)[key] = adjustments[key];
                }
              }
            }
            if (Object.keys(delta).length > 0) {
              otherPaths.forEach((p) => globalImageCache.delete(p));
              invoke(Invokes.ApplyAdjustmentsToPaths, { paths: otherPaths, adjustments: delta }).catch((err) => {
                console.error('Failed to apply adjustments to multi-selection:', err);
              });
            }
          }
        }
        prevAdjustmentsRef.current = { path: selectedImage.path, adjustments };
      }, 50);
    }

    return () => {
      if (dragIdleTimer.current) clearTimeout(dragIdleTimer.current);
    };
  }, [
    activeView,
    activeRightPanel,
    adjustments,
    previewOverride,
    selectedImage?.path,
    selectedImage?.isReady,
    isSliderDragging,
    multiSelectedPaths,
    appSettings?.enableLivePreviews,
    appSettings?.copyPasteSettings?.includedAdjustments,
    appSettings?.copyPasteSettings?.autoSync,
    isWaveformVisible,
  ]);

  useEffect(() => {
    originalPreviewJobIdRef.current += 1;
    setEditor({ transformedOriginalUrl: null });
    currentOriginalResRef.current = 0;
  }, [geometricAdjustmentsKey, selectedImage?.path, setEditor]);

  useEffect(() => {
    if (
      activeView === 'editor' &&
      showOriginal &&
      selectedImage?.isReady &&
      displaySize.width > 0 &&
      !isSliderDragging
    ) {
      const targetRes = calculateTargetRes();
      if (targetRes > currentOriginalResRef.current) {
        requestHiFiOriginalZoom(adjustments, targetRes);
      }
    }
    return () => {
      requestHiFiOriginalZoom.cancel();
    };
  }, [
    activeView,
    showOriginal,
    displaySize.width,
    displaySize.height,
    calculateTargetRes,
    selectedImage?.isReady,
    isSliderDragging,
    requestHiFiOriginalZoom,
    originalSize,
  ]);

  useEffect(() => {
    let isEffectActive = true;
    const generate = async () => {
      if (activeView === 'editor' && showOriginal && selectedImage?.path && !transformedOriginalUrl) {
        try {
          const targetRes = calculateTargetRes();
          await generateOriginalPreview(adjustments, targetRes);
        } catch (e) {
          if (isEffectActive) {
            console.error('Failed to generate original preview:', e);
            setEditor({ showOriginal: false });
          }
        }
      }
    };
    generate();
    return () => {
      isEffectActive = false;
      originalPreviewJobIdRef.current += 1;
    };
  }, [
    activeView,
    showOriginal,
    selectedImage?.path,
    adjustments,
    transformedOriginalUrl,
    calculateTargetRes,
    generateOriginalPreview,
    setEditor,
  ]);

  return {
    applyAdjustments,
    executeApplyAdjustments,
  };
}
