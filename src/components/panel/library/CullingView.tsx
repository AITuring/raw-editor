import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { List, useListCallbackRef } from 'react-window';
import clsx from 'clsx';
import { ImageFile } from '../../ui/AppProperties';
import Text from '../../ui/Text';
import { TextColors, TextVariants } from '../../../types/typography';
import { useProcessStore } from '../../../store/useProcessStore';
import type { SyncViewport } from './CullingView/types';
import { CullingPreview } from './CullingView/CullingPreview';
import { VirtualizedCullingRow } from './CullingView/CullingRow';

export default function CullingView(props: any) {
  const { t } = useTranslation();
  const {
    imageList,
    multiSelectedPaths,
    activePath,
    onImageClick,
    imageRatings,
    thumbnailAspectRatio,
    onContextMenu,
    onImageDoubleClick,
    onRequestThumbnails,
    onClearSelection,
    onEmptyAreaContextMenu,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const [listHeight, setListHeight] = useState(0);
  const [sidebarWidth, setSidebarWidth] = useState(224);
  const isResizing = useRef(false);

  const requestQueueRef = useRef<Set<string>>(new Set());
  const requestTimeoutRef = useRef<any>(null);

  const [hoveredCullingPath, setHoveredCullingPath] = useState<string | null>(null);

  const [syncViewport, setSyncViewport] = useState<SyncViewport>({
    isActive: false,
    zoom: 1,
    pan: { x: 0, y: 0 },
    isDragging: false,
  });

  const [showRateBar, setShowRateBar] = useState(false);
  const [showInfoBar, setShowInfoBar] = useState(false);

  const [listHandle, setListHandle] = useListCallbackRef();
  const prevActivePath = useRef<string | null>(null);
  const prevListElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!listHandle?.element || imageList.length === 0 || !activePath) return;

    const element = listHandle.element as HTMLElement;
    const isPathChanged = activePath !== prevActivePath.current;
    const isElementChanged = element !== prevListElement.current;

    if (isPathChanged || isElementChanged) {
      const isInitial = prevActivePath.current === null || isElementChanged;
      prevActivePath.current = activePath;
      prevListElement.current = element;

      const index = imageList.findIndex((img: ImageFile) => img.path === activePath);
      if (index !== -1) {
        const rowHeight = sidebarWidth - 16;
        const targetTop = index * rowHeight;
        const clientHeight = element.clientHeight;
        const scrollTop = element.scrollTop;
        const itemBottom = targetTop + rowHeight;
        const SCROLL_OFFSET = 40;

        if (isInitial) {
          element.scrollTo({
            top: Math.max(0, targetTop - clientHeight / 2 + rowHeight / 2),
            behavior: 'instant',
          });
        } else if (itemBottom > scrollTop + clientHeight) {
          element.scrollTo({
            top: itemBottom - clientHeight + SCROLL_OFFSET,
            behavior: 'smooth',
          });
        } else if (targetTop < scrollTop) {
          element.scrollTo({
            top: Math.max(0, targetTop - SCROLL_OFFSET),
            behavior: 'smooth',
          });
        }
      }
    }
  }, [activePath, listHandle, imageList, sidebarWidth]);

  const queueThumbnailRequest = useCallback(
    (path: string) => {
      if (!onRequestThumbnails) return;
      if (useProcessStore.getState().thumbnails[path]) return;
      requestQueueRef.current.add(path);
      if (!requestTimeoutRef.current) {
        requestTimeoutRef.current = setTimeout(() => {
          const paths = Array.from(requestQueueRef.current);
          if (paths.length > 0) {
            onRequestThumbnails(paths);
            requestQueueRef.current.clear();
          }
          requestTimeoutRef.current = null;
        }, 50);
      }
    },
    [onRequestThumbnails],
  );

  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setListHeight(containerRef.current.clientHeight);
      }
    };
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  const resize = useCallback((mouseMoveEvent: MouseEvent) => {
    if (!isResizing.current) return;
    const newWidth = window.innerWidth - mouseMoveEvent.clientX;
    if (newWidth >= 160 && newWidth <= 450) {
      setSidebarWidth(newWidth);
    }
  }, []);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener('mousemove', resize);
    document.removeEventListener('mouseup', stopResizing);
  }, [resize]);

  const startResizing = useCallback(
    (mouseDownEvent: React.MouseEvent) => {
      mouseDownEvent.preventDefault();
      isResizing.current = true;
      document.addEventListener('mousemove', resize);
      document.addEventListener('mouseup', stopResizing);
    },
    [resize, stopResizing],
  );

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', resize);
      document.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  const rowProps = useMemo(
    () => ({
      imageList,
      multiSelectedPaths,
      activePath,
      thumbnailAspectRatio,
      imageRatings,
      onContextMenu,
      onImageDoubleClick,
      onImageClick,
      queueThumbnailRequest,
      sidebarWidth,
      hoveredCullingPath,
    }),
    [
      imageList,
      multiSelectedPaths,
      activePath,
      thumbnailAspectRatio,
      imageRatings,
      onContextMenu,
      onImageDoubleClick,
      onImageClick,
      queueThumbnailRequest,
      sidebarWidth,
      hoveredCullingPath,
    ],
  );

  const displayPaths = multiSelectedPaths.slice(-6);
  const displayImages = displayPaths
    .map((p: string) => imageList.find((img: ImageFile) => img.path === p))
    .filter(Boolean);
  const displayCount = displayImages.length;

  const handleSidebarEmptyClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest('[data-bench-id="thumbnail"]') && !target.closest('button')) {
      onClearSelection?.();
    }
  };

  const handleSidebarEmptyContextMenu = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest('[data-bench-id="thumbnail"]') && !target.closest('button')) {
      onEmptyAreaContextMenu?.(e);
    }
  };

  return (
    <div className="flex w-full h-full min-h-0 bg-transparent">
      <div className="flex-1 flex overflow-hidden relative bg-transparent">
        {displayCount === 0 ? (
          <div className="m-auto text-center">
            <Text variant={TextVariants.heading} color={TextColors.secondary}>
              {t('library.culling.selectImagesToCompare')}
            </Text>
            <Text className="mt-2 text-text-secondary">{t('library.culling.clickThumbnailsHint')}</Text>
          </div>
        ) : (
          <div
            className={clsx(
              'grid gap-2 w-full h-full p-2',
              displayCount === 1 && 'grid-cols-1 grid-rows-1',
              displayCount === 2 && 'grid-cols-2 grid-rows-1',
              displayCount === 3 && 'grid-cols-2 grid-rows-2',
              displayCount === 4 && 'grid-cols-2 grid-rows-2',
              displayCount === 5 && 'grid-cols-3 grid-rows-2',
              displayCount === 6 && 'grid-cols-3 grid-rows-2',
            )}
          >
            {displayImages.map((img: ImageFile, index: number) => (
              <CullingPreview
                key={img.path}
                image={img}
                rating={imageRatings?.[img.path] || 0}
                onContextMenu={onContextMenu}
                onImageDoubleClick={onImageDoubleClick}
                isActive={activePath === img.path}
                isSelected={true}
                isFullWidth={(displayCount === 3 && index === 2) || (displayCount === 5 && index === 4)}
                syncViewport={syncViewport}
                setSyncViewport={setSyncViewport}
                hoveredPath={hoveredCullingPath}
                setHoveredCullingPath={setHoveredCullingPath}
                showRateBar={showRateBar}
                setShowRateBar={setShowRateBar}
                showInfoBar={showInfoBar}
                setShowInfoBar={setShowInfoBar}
              />
            ))}
          </div>
        )}
      </div>

      <div
        ref={containerRef}
        style={{ width: sidebarWidth }}
        className="relative flex shrink-0 flex-col border-l border-border-color bg-bg-secondary/50"
        onClick={handleSidebarEmptyClick}
        onContextMenu={handleSidebarEmptyContextMenu}
      >
        <div
          onMouseDown={startResizing}
          className="absolute top-0 bottom-0 left-0 w-1 cursor-col-resize hover:bg-surface/50 active:bg-surface transition-colors z-40"
        />
        <div key={`${sidebarWidth}-${thumbnailAspectRatio}`} style={{ height: listHeight, width: '100%' }}>
          <List
            listRef={setListHandle}
            rowCount={imageList.length}
            rowHeight={sidebarWidth - 16}
            rowComponent={VirtualizedCullingRow}
            rowProps={rowProps}
            className="custom-scrollbar"
          />
        </div>
      </div>
    </div>
  );
}
