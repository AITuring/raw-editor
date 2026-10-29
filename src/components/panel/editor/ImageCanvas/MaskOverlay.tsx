import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Ellipse, Line, Transformer, Group, Circle, Rect } from 'react-konva';
import { Adjustments, Coord } from '../../../../utils/adjustments';
import { Mask, SubMask, SubMaskMode } from '../../right/Masks';
import type { DrawnLine } from './types';
import { OptimizedBrushLine, SourcePreviewLine } from './BrushLines';

interface MaskOverlayProps {
  adjustments: Adjustments;
  imageHeight: number;
  imageWidth: number;
  onMaskInteractionEnd(): void;
  onMaskInteractionStart(event?: any): void;
  isToolActive: boolean;
  isSelected: boolean;
  showBrushStrokes?: boolean;
  onMaskMouseEnter(): void;
  onMaskMouseLeave(): void;
  onPreviewUpdate?(id: string, subMask: Partial<SubMask>): void;
  onSelect(): void;
  onUpdate(id: string, subMask: Partial<SubMask>): void;
  scale: number;
  subMask: SubMask;
  offsetX: number;
  offsetY: number;
  stageScale: number;
}

export const MaskOverlay = memo(
  ({
    adjustments,
    imageHeight,
    imageWidth,
    onMaskInteractionEnd,
    onMaskInteractionStart,
    isToolActive,
    isSelected,
    showBrushStrokes = true,
    onMaskMouseEnter,
    onMaskMouseLeave,
    onPreviewUpdate,
    onSelect,
    onUpdate,
    scale,
    subMask,
    offsetX,
    offsetY,
    stageScale,
  }: MaskOverlayProps) => {
    const shapeRef = useRef<any>(null);
    const trRef = useRef<any>(null);
    const rotateStartRef = useRef<any>(null);

    const crop = adjustments.crop;
    const isPercent = crop?.unit === '%';
    const cropX = crop ? (isPercent ? (crop.x / 100) * imageWidth : crop.x) : 0;
    const cropY = crop ? (isPercent ? (crop.y / 100) * imageHeight : crop.y) : 0;
    const cropW = crop ? (isPercent ? (crop.width / 100) * imageWidth : crop.width) : imageWidth;
    const cropH = crop ? (isPercent ? (crop.height / 100) * imageHeight : crop.height) : imageHeight;

    const [p, setP] = useState(subMask.parameters);
    const pRef = useRef(p);
    const isDragging = useRef(false);

    const dragStartPointer = useRef<Coord | null>(null);
    const dragStartParams = useRef<any>(null);

    const getPointer = useCallback(
      (stage: any) => {
        const pos = stage.getPointerPosition();
        if (!pos) return null;
        return { x: pos.x / stageScale - offsetX, y: pos.y / stageScale - offsetY };
      },
      [offsetX, offsetY, stageScale],
    );

    useEffect(() => {
      if (!isDragging.current) {
        setP(subMask.parameters);
        pRef.current = subMask.parameters;
      }
    }, [subMask.parameters]);

    const updateP = useCallback((newP: any) => {
      setP(newP);
      pRef.current = newP;
    }, []);

    const handleMaskTouchStart = useCallback(
      (e: any) => {
        if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;

        onMaskInteractionStart(e);
        if (e.evt.cancelable) e.evt.preventDefault();
        e.evt.stopPropagation?.();
      },
      [onMaskInteractionStart],
    );

    const handleMaskTouchEnd = useCallback(() => {
      onMaskInteractionEnd();
    }, [onMaskInteractionEnd]);

    const handleSelect = isToolActive ? undefined : onSelect;

    useEffect(() => {
      if (isSelected && trRef.current && shapeRef.current) {
        trRef.current?.nodes([shapeRef.current]);
        trRef.current?.getLayer().batchDraw();
      }
    }, [isSelected, isToolActive]);

    const lockDragBoundFunc = useCallback(function (this: any) {
      return this.getAbsolutePosition();
    }, []);

    const handleRadialDragStart = useCallback(
      (e: any) => {
        if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;
        isDragging.current = true;
        onMaskInteractionStart(e);
        dragStartPointer.current = getPointer(e.target.getStage());
        dragStartParams.current = { ...pRef.current };
      },
      [onMaskInteractionStart, getPointer],
    );

    const handleRadialDragMove = useCallback(
      (e: any) => {
        const pointerPos = getPointer(e.target.getStage());
        if (!pointerPos || !dragStartPointer.current || !dragStartParams.current) return;

        const dx = (pointerPos.x - dragStartPointer.current.x) / scale;
        const dy = (pointerPos.y - dragStartPointer.current.y) / scale;

        const newP = {
          ...dragStartParams.current,
          centerX: dragStartParams.current.centerX + dx,
          centerY: dragStartParams.current.centerY + dy,
        };

        updateP(newP);
        if (onPreviewUpdate) onPreviewUpdate(subMask.id, { parameters: newP });

        onUpdate(subMask.id, { parameters: newP });
      },
      [scale, updateP, onPreviewUpdate, subMask.id, getPointer, onUpdate],
    );

    const handleRadialDragEnd = useCallback(() => {
      isDragging.current = false;
      onMaskInteractionEnd();
      onUpdate(subMask.id, { parameters: pRef.current });
    }, [subMask.id, onMaskInteractionEnd, onUpdate]);

    const handleRadialTransformStart = useCallback(
      (e: any) => {
        isDragging.current = true;
        onMaskInteractionStart(e);
      },
      [onMaskInteractionStart],
    );

    const handleRadialTransform = useCallback(() => {
      const node = shapeRef.current;
      if (!node) return;

      const scaleX = Math.abs(node.scaleX());
      const scaleY = Math.abs(node.scaleY());

      if (pRef.current.radiusX * scaleX < 5 || pRef.current.radiusY * scaleY < 5) {
        node.scaleX(node.lastValidScaleX || 1);
        node.scaleY(node.lastValidScaleY || 1);
      } else {
        node.lastValidScaleX = scaleX;
        node.lastValidScaleY = scaleY;
      }

      const newRadiusX = pRef.current.radiusX * node.scaleX();
      const newRadiusY = pRef.current.radiusY * node.scaleY();

      const newP = {
        ...pRef.current,
        centerX: node.x() / scale + cropX,
        centerY: node.y() / scale + cropY,
        radiusX: newRadiusX,
        radiusY: newRadiusY,
        rotation: node.rotation(),
      };

      if (onPreviewUpdate) {
        onPreviewUpdate(subMask.id, { parameters: newP });
      }

      onUpdate(subMask.id, { parameters: newP });
    }, [onPreviewUpdate, scale, cropX, cropY, subMask.id, onUpdate]);

    const handleRadialTransformEnd = useCallback(() => {
      const node = shapeRef.current;
      if (!node) return;

      const scaleX = node.scaleX();
      const scaleY = node.scaleY();

      const newRadiusX = pRef.current.radiusX * scaleX;
      const newRadiusY = pRef.current.radiusY * scaleY;

      node.scaleX(1);
      node.scaleY(1);

      const newP = {
        ...pRef.current,
        centerX: node.x() / scale + cropX,
        centerY: node.y() / scale + cropY,
        radiusX: newRadiusX,
        radiusY: newRadiusY,
        rotation: node.rotation(),
      };

      updateP(newP);
      isDragging.current = false;
      onMaskInteractionEnd();
      onUpdate(subMask.id, { parameters: newP });
    }, [scale, cropX, cropY, updateP, onMaskInteractionEnd, onUpdate, subMask.id]);

    const setRotateCursor = useCallback(
      (stage: any, pointerPos: any) => {
        const cx = (pRef.current.centerX - cropX) * scale;
        const cy = (pRef.current.centerY - cropY) * scale;
        const angle = Math.atan2(pointerPos.y - cy, pointerPos.x - cx) * (180 / Math.PI);

        const svgStr = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0px 1px 2px rgba(0,0,0,0.8));">
          <g transform="rotate(${Math.round(angle)} 16 16)">
            <path d="M 23 9 A 10 10 0 0 1 23 23" />
            <path d="M 28 9 L 23 9 L 23 14" />
            <path d="M 28 23 L 23 23 L 23 18" />
          </g>
        </svg>`;
        const encodedSvg = encodeURIComponent(svgStr);
        stage.container().style.cursor = `url('data:image/svg+xml;utf8,${encodedSvg}') 16 16, crosshair`;
      },
      [cropX, cropY, scale],
    );

    const handleRotateStart = useCallback(
      (e: any) => {
        if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;

        isDragging.current = true;
        onMaskInteractionStart(e);
        e.cancelBubble = true;
        if (e.evt && e.evt.cancelable) e.evt.preventDefault();

        const stage = e.target.getStage();
        const pointer = getPointer(stage);
        if (!pointer) return;

        const cx = (pRef.current.centerX - cropX) * scale;
        const cy = (pRef.current.centerY - cropY) * scale;

        const startAngle = Math.atan2(pointer.y - cy, pointer.x - cx);
        rotateStartRef.current = {
          angle: startAngle,
          rotation: pRef.current.rotation || 0,
        };
      },
      [onMaskInteractionStart, cropX, cropY, scale, getPointer],
    );

    const handleRotateMove = useCallback(
      (e: any) => {
        if (!rotateStartRef.current) return;
        const stage = e.target.getStage();
        const pointer = getPointer(stage);
        if (!pointer) return;

        setRotateCursor(stage, pointer);

        const cx = (pRef.current.centerX - cropX) * scale;
        const cy = (pRef.current.centerY - cropY) * scale;

        const currentAngle = Math.atan2(pointer.y - cy, pointer.x - cx);
        const angleDiff = currentAngle - rotateStartRef.current.angle;
        const angleDiffDeg = (angleDiff * 180) / Math.PI;

        const newRotation = rotateStartRef.current.rotation + angleDiffDeg;

        const newP = {
          ...pRef.current,
          rotation: newRotation,
        };

        updateP(newP);
        if (onPreviewUpdate) onPreviewUpdate(subMask.id, { parameters: newP });
        onUpdate(subMask.id, { parameters: newP });
      },
      [cropX, cropY, scale, updateP, onPreviewUpdate, subMask.id, setRotateCursor, getPointer, onUpdate],
    );

    const handleRotateEnd = useCallback(
      (e: any) => {
        isDragging.current = false;
        rotateStartRef.current = null;
        onMaskInteractionEnd();
        onUpdate(subMask.id, { parameters: pRef.current });

        if (e?.target?.getStage) {
          e.target.getStage().container().style.cursor = '';
        }
      },
      [subMask.id, onMaskInteractionEnd, onUpdate],
    );

    const handleRotateHoverMove = useCallback(
      (e: any) => {
        if (isToolActive || isDragging.current) return;
        const stage = e.target.getStage();
        const pointer = getPointer(stage);
        if (pointer) setRotateCursor(stage, pointer);
      },
      [isToolActive, setRotateCursor, getPointer],
    );

    const handleRotateMouseEnter = useCallback(
      (e: any) => {
        onMaskMouseEnter();
        if (!isToolActive && !isDragging.current) {
          const stage = e.target.getStage();
          const pointer = getPointer(stage);
          if (pointer) setRotateCursor(stage, pointer);
        }
      },
      [onMaskMouseEnter, isToolActive, setRotateCursor, getPointer],
    );

    const handleRotateMouseLeave = useCallback(
      (e: any) => {
        onMaskMouseLeave();
        if (!isDragging.current) {
          const stage = e.target.getStage();
          stage.container().style.cursor = '';
        }
      },
      [onMaskMouseLeave],
    );

    const handleLinearGroupDragStart = useCallback(
      (e: any) => {
        if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;
        isDragging.current = true;
        onMaskInteractionStart(e);
        dragStartPointer.current = getPointer(e.target.getStage());
        dragStartParams.current = { ...pRef.current };
        e.cancelBubble = true;
      },
      [onMaskInteractionStart, getPointer],
    );

    const handleLinearGroupDragMove = useCallback(
      (e: any) => {
        const pointerPos = getPointer(e.target.getStage());
        if (!pointerPos || !dragStartPointer.current || !dragStartParams.current) return;

        const dx = (pointerPos.x - dragStartPointer.current.x) / scale;
        const dy = (pointerPos.y - dragStartPointer.current.y) / scale;

        const newP = {
          ...dragStartParams.current,
          startX: dragStartParams.current.startX + dx,
          startY: dragStartParams.current.startY + dy,
          endX: dragStartParams.current.endX + dx,
          endY: dragStartParams.current.endY + dy,
        };

        updateP(newP);
        if (onPreviewUpdate) onPreviewUpdate(subMask.id, { parameters: newP });
        onUpdate(subMask.id, { parameters: newP });
      },
      [scale, updateP, onPreviewUpdate, subMask.id, getPointer, onUpdate],
    );

    const handleLinearGroupDragEnd = useCallback(
      (e: any) => {
        isDragging.current = false;
        e.cancelBubble = true;
        onMaskInteractionEnd();
        onUpdate(subMask.id, { parameters: pRef.current });
      },
      [subMask.id, onMaskInteractionEnd, onUpdate],
    );

    const handleLinearPointDragStart = useCallback(
      (e: any) => {
        if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;
        isDragging.current = true;
        onMaskInteractionStart(e);
        e.cancelBubble = true;
      },
      [onMaskInteractionStart],
    );

    const handleLinearPointDragMove = useCallback(
      (e: any, pointType: string) => {
        const stage = e.target.getStage();
        const pointerPos = getPointer(stage);
        if (!pointerPos) return;

        const newX = pointerPos.x / scale + cropX;
        const newY = pointerPos.y / scale + cropY;

        const newP = { ...pRef.current };
        if (pointType === 'start') {
          newP.startX = newX;
          newP.startY = newY;
        } else {
          newP.endX = newX;
          newP.endY = newY;
        }
        updateP(newP);
        if (onPreviewUpdate) onPreviewUpdate(subMask.id, { parameters: newP });
        onUpdate(subMask.id, { parameters: newP });
      },
      [scale, cropX, cropY, updateP, onPreviewUpdate, subMask.id, getPointer, onUpdate],
    );

    const handleLinearRangeDragMove = useCallback(
      (e: any) => {
        const stage = e.target.getStage();
        const pointerPos = getPointer(stage);
        if (!pointerPos) return;

        const { startX, startY, endX, endY } = pRef.current;
        const sX = (startX - cropX) * scale;
        const sY = (startY - cropY) * scale;
        const eX = (endX - cropX) * scale;
        const eY = (endY - cropY) * scale;

        const dx = eX - sX;
        const dy = eY - sY;
        const len = Math.sqrt(dx * dx + dy * dy);

        let newRange = pRef.current.range;
        if (len > 0) {
          const dist = Math.abs(dx * (sY - pointerPos.y) - (sX - pointerPos.x) * dy) / len;
          newRange = Math.max(0.1, dist / scale);
        }

        const newP = { ...pRef.current, range: newRange };
        updateP(newP);
        if (onPreviewUpdate) onPreviewUpdate(subMask.id, { parameters: newP });

        onUpdate(subMask.id, { parameters: newP });
      },
      [scale, cropX, cropY, updateP, onPreviewUpdate, subMask.id, getPointer, onUpdate],
    );

    const handleLinearPointDragEnd = useCallback(
      (e: any) => {
        isDragging.current = false;
        e.cancelBubble = true;
        onMaskInteractionEnd();
        onUpdate(subMask.id, { parameters: pRef.current });
      },
      [subMask.id, onMaskInteractionEnd, onUpdate],
    );

    if (!subMask.visible) {
      return null;
    }

    const commonProps = {
      dash: [4, 4],
      onClick: handleSelect,
      onTap: handleSelect,
      opacity: isSelected ? 1 : 0.7,
      stroke: isSelected
        ? '#0ea5e9'
        : subMask.mode === SubMaskMode.Subtractive
          ? '#f43f5e'
          : subMask.mode === SubMaskMode.Intersect
            ? '#a855f7'
            : 'white',
      strokeScaleEnabled: false,
      strokeWidth: isSelected ? 3 : 2,
    };

    if (subMask.type === Mask.AiSubject || subMask.type === Mask.QuickEraser) {
      const { startX, startY, endX, endY } = p;
      if (startX !== undefined && startY !== undefined && endX !== undefined && endY !== undefined) {
        const isPoint = Math.abs(startX - endX) < 1e-6 && Math.abs(startY - endY) < 1e-6;
        if (isPoint) {
          return (
            <Circle
              x={(startX - cropX) * scale}
              y={(startY - cropY) * scale}
              radius={5}
              stroke={isSelected ? '#0ea5e9' : 'white'}
              strokeWidth={2}
              listening={!isToolActive}
              onClick={handleSelect}
              onTap={handleSelect}
              onTouchEnd={handleMaskTouchEnd}
              onTouchStart={handleMaskTouchStart}
              onMouseEnter={onMaskMouseEnter}
              onMouseLeave={onMaskMouseLeave}
              shadowColor="black"
              shadowBlur={2}
              shadowOpacity={0.8}
            />
          );
        } else {
          return (
            <Rect
              height={Math.max(0.1, Math.abs(endY - startY) * scale)}
              onMouseEnter={onMaskMouseEnter}
              onMouseLeave={onMaskMouseLeave}
              onTouchEnd={handleMaskTouchEnd}
              onTouchStart={handleMaskTouchStart}
              width={Math.max(0.1, Math.abs(endX - startX) * scale)}
              x={(Math.min(startX, endX) - cropX) * scale}
              y={(Math.min(startY, endY) - cropY) * scale}
              {...commonProps}
            />
          );
        }
      }
      return null;
    }

    if (
      subMask.type === Mask.Brush ||
      subMask.type === Mask.Flow ||
      subMask.type === Mask.Clone ||
      subMask.type === Mask.Heal
    ) {
      const { lines = [], sourceX, sourceY } = p;

      let dx = 0;
      let dy = 0;
      let hasSource = false;

      if (
        (subMask.type === Mask.Clone || subMask.type === Mask.Heal) &&
        sourceX !== undefined &&
        sourceY !== undefined &&
        lines.length > 0
      ) {
        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity;
        for (const line of lines) {
          for (const pt of line.points) {
            if (pt.x < minX) minX = pt.x;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.y > maxY) maxY = pt.y;
          }
        }
        if (minX !== Infinity) {
          const cx = (minX + maxX) / 2;
          const cy = (minY + maxY) / 2;
          dx = sourceX - cx;
          dy = sourceY - cy;
          hasSource = true;
        }
      }

      return (
        <Group
          onClick={handleSelect}
          onTap={handleSelect}
          onTouchEnd={handleMaskTouchEnd}
          onTouchStart={handleMaskTouchStart}
        >
          <Group visible={showBrushStrokes !== false}>
            {lines.map((line: DrawnLine, i: number) => (
              <OptimizedBrushLine key={i} line={line} scale={scale} cropX={cropX} cropY={cropY} />
            ))}

            {hasSource &&
              isSelected &&
              lines.map((line: DrawnLine, i: number) => (
                <SourcePreviewLine
                  key={`source-${i}`}
                  line={line}
                  scale={scale}
                  cropX={cropX}
                  cropY={cropY}
                  dx={dx}
                  dy={dy}
                />
              ))}
          </Group>

          {sourceX !== undefined && sourceY !== undefined && isSelected && (
            <Group x={(sourceX - cropX) * scale} y={(sourceY - cropY) * scale}>
              <Circle
                radius={6 / stageScale}
                stroke="white"
                strokeWidth={2 / stageScale}
                shadowColor="black"
                shadowBlur={2 / stageScale}
              />
              <Circle
                radius={6 / stageScale}
                stroke="black"
                strokeWidth={1 / stageScale}
                dash={[2 / stageScale, 2 / stageScale]}
              />
              <Line
                points={[-10 / stageScale, 0, 10 / stageScale, 0]}
                stroke="white"
                strokeWidth={1.5 / stageScale}
                shadowColor="black"
                shadowBlur={2 / stageScale}
              />
              <Line
                points={[0, -10 / stageScale, 0, 10 / stageScale]}
                stroke="white"
                strokeWidth={1.5 / stageScale}
                shadowColor="black"
                shadowBlur={2 / stageScale}
              />
            </Group>
          )}
        </Group>
      );
    }

    if (subMask.type === Mask.Radial) {
      const { centerX, centerY, radiusX, radiusY, rotation } = p;
      if (p.isInitialDraw && (radiusX < 1 || radiusY < 2)) return null;

      return (
        <Group>
          {isSelected && !isToolActive && (
            <Ellipse
              x={(centerX - cropX) * scale}
              y={(centerY - cropY) * scale}
              radiusX={Math.max(0.1, radiusX * scale) + 35}
              radiusY={Math.max(0.1, radiusY * scale) + 35}
              rotation={rotation}
              fill="transparent"
              draggable
              dragBoundFunc={lockDragBoundFunc}
              onDragStart={handleRotateStart}
              onDragMove={handleRotateMove}
              onDragEnd={handleRotateEnd}
              onMouseEnter={handleRotateMouseEnter}
              onMouseMove={handleRotateHoverMove}
              onMouseLeave={handleRotateMouseLeave}
              onTouchStart={handleRotateStart}
              onTouchMove={handleRotateMove}
              onTouchEnd={handleRotateEnd}
            />
          )}

          <Ellipse
            {...commonProps}
            ref={shapeRef}
            fill="transparent"
            draggable={!isToolActive}
            dragBoundFunc={lockDragBoundFunc}
            onDragStart={handleRadialDragStart}
            onDragMove={handleRadialDragMove}
            onDragEnd={handleRadialDragEnd}
            onMouseEnter={(e: any) => {
              onMaskMouseEnter();
              if (!isToolActive && !isDragging.current) {
                e.target.getStage().container().style.cursor = 'move';
              }
            }}
            onMouseLeave={(e: any) => {
              onMaskMouseLeave();
              if (!isDragging.current && e?.target?.getStage) {
                e.target.getStage().container().style.cursor = '';
              }
            }}
            onTouchEnd={handleMaskTouchEnd}
            onTouchStart={handleMaskTouchStart}
            radiusX={Math.max(0.1, radiusX * scale)}
            radiusY={Math.max(0.1, radiusY * scale)}
            rotation={rotation}
            x={(centerX - cropX) * scale}
            y={(centerY - cropY) * scale}
          />
          {isSelected && !isToolActive && (
            <Transformer
              ref={trRef}
              centeredScaling={true}
              rotateEnabled={false}
              enabledAnchors={[
                'top-left',
                'top-right',
                'bottom-left',
                'bottom-right',
                'top-center',
                'bottom-center',
                'middle-left',
                'middle-right',
              ]}
              onMouseDown={(e) => {
                if (e.evt && typeof e.evt.button === 'number' && e.evt.button !== 0) return;
                e.cancelBubble = true;
                e.evt.preventDefault();
              }}
              onTouchStart={(e) => {
                handleMaskTouchStart(e);
                e.cancelBubble = true;
                e.evt.preventDefault();
              }}
              onTouchEnd={handleMaskTouchEnd}
              boundBoxFunc={(oldBox, newBox) => {
                if (Math.abs(newBox.width) < 5 || Math.abs(newBox.height) < 5) {
                  return oldBox;
                }
                return newBox;
              }}
              onTransformStart={handleRadialTransformStart}
              onTransform={handleRadialTransform}
              onTransformEnd={handleRadialTransformEnd}
              onMouseEnter={onMaskMouseEnter}
              onMouseLeave={onMaskMouseLeave}
            />
          )}
        </Group>
      );
    }

    if (subMask.type === Mask.Linear) {
      const defaultRange = Math.min(cropW, cropH) * 0.1;
      const { startX, startY, endX, endY, range = defaultRange } = p;

      const flickDistX = startX - endX;
      const flickDistY = startY - endY;
      if (p.isInitialDraw && Math.sqrt(flickDistX * flickDistX + flickDistY * flickDistY) < 1) return null;

      const sX = (startX - cropX) * scale;
      const sY = (startY - cropY) * scale;
      const eX = (endX - cropX) * scale;
      const eY = (endY - cropY) * scale;
      const r = range * scale;

      const idx = endX - startX;
      const idy = endY - startY;
      const angle = Math.atan2(idy, idx);
      const angleDeg = (angle * 180) / Math.PI;

      const centerX = sX + (eX - sX) / 2;
      const centerY = sY + (eY - sY) / 2;

      const nx = -Math.sin(angle);
      const ny = Math.cos(angle);
      const dx_norm = Math.cos(angle);
      const dy_norm = Math.sin(angle);

      const EXT = 5000;
      const topRangePts = [
        sX + nx * r - dx_norm * EXT,
        sY + ny * r - dy_norm * EXT,
        eX + nx * r + dx_norm * EXT,
        eY + ny * r + dy_norm * EXT,
      ];
      const botRangePts = [
        sX - nx * r - dx_norm * EXT,
        sY - ny * r - dy_norm * EXT,
        eX - nx * r + dx_norm * EXT,
        eY - ny * r + dy_norm * EXT,
      ];

      const lineProps = {
        ...commonProps,
        strokeWidth: isSelected ? 2.5 : 2,
        dash: [6, 6],
        hitStrokeWidth: 40,
      };

      const showFeatherLines = isSelected && (!isToolActive || p.isInitialDraw);

      return (
        <Group>
          <Group
            x={centerX}
            y={centerY}
            rotation={angleDeg}
            draggable={isSelected && !isToolActive}
            dragBoundFunc={lockDragBoundFunc}
            onDragStart={handleLinearGroupDragStart}
            onDragMove={handleLinearGroupDragMove}
            onDragEnd={handleLinearGroupDragEnd}
            onClick={handleSelect}
            onTap={handleSelect}
            onTouchEnd={handleMaskTouchEnd}
            onTouchStart={handleMaskTouchStart}
            onMouseEnter={(e: any) => {
              onMaskMouseEnter();
              if (!isToolActive) e.target.getStage().container().style.cursor = 'move';
            }}
            onMouseLeave={(e: any) => {
              onMaskMouseLeave();
              e.target.getStage().container().style.cursor = '';
            }}
          >
            <Line points={[-5000, 0, 5000, 0]} {...lineProps} dash={[2, 3]} />
          </Group>

          {showFeatherLines && (
            <>
              <Line
                points={topRangePts}
                {...lineProps}
                draggable={!isToolActive}
                dragBoundFunc={lockDragBoundFunc}
                onDragStart={handleLinearPointDragStart}
                onDragMove={handleLinearRangeDragMove}
                onDragEnd={handleLinearPointDragEnd}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  if (!isToolActive) e.target.getStage().container().style.cursor = 'row-resize';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
              <Line
                points={botRangePts}
                {...lineProps}
                draggable={!isToolActive}
                dragBoundFunc={lockDragBoundFunc}
                onDragStart={handleLinearPointDragStart}
                onDragMove={handleLinearRangeDragMove}
                onDragEnd={handleLinearPointDragEnd}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  if (!isToolActive) e.target.getStage().container().style.cursor = 'row-resize';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
            </>
          )}

          {isSelected && !isToolActive && (
            <>
              <Circle
                x={sX}
                y={sY}
                radius={8 / stageScale}
                fill="#0ea5e9"
                stroke="white"
                strokeWidth={2 / stageScale}
                draggable
                dragBoundFunc={lockDragBoundFunc}
                onDragStart={handleLinearPointDragStart}
                onDragMove={(e) => handleLinearPointDragMove(e, 'start')}
                onDragEnd={handleLinearPointDragEnd}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  e.target.getStage().container().style.cursor = 'grab';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
              <Circle
                x={eX}
                y={eY}
                radius={8 / stageScale}
                fill="#0ea5e9"
                stroke="white"
                strokeWidth={2 / stageScale}
                draggable
                dragBoundFunc={lockDragBoundFunc}
                onDragStart={handleLinearPointDragStart}
                onDragMove={(e) => handleLinearPointDragMove(e, 'end')}
                onDragEnd={handleLinearPointDragEnd}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  e.target.getStage().container().style.cursor = 'grab';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
            </>
          )}

          {!isSelected && (
            <>
              <Line
                points={topRangePts}
                {...lineProps}
                opacity={0.7}
                stroke="white"
                listening={true}
                onClick={handleSelect}
                onTap={handleSelect}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  if (!isToolActive) e.target.getStage().container().style.cursor = 'row-resize';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
              <Line
                points={botRangePts}
                {...lineProps}
                opacity={0.7}
                stroke="white"
                listening={true}
                onClick={handleSelect}
                onTap={handleSelect}
                onTouchEnd={handleMaskTouchEnd}
                onTouchStart={handleMaskTouchStart}
                onMouseEnter={(e: any) => {
                  onMaskMouseEnter();
                  if (!isToolActive) e.target.getStage().container().style.cursor = 'row-resize';
                }}
                onMouseLeave={(e: any) => {
                  onMaskMouseLeave();
                  e.target.getStage().container().style.cursor = '';
                }}
              />
            </>
          )}
        </Group>
      );
    }

    if (subMask.type === Mask.Color || subMask.type === Mask.Luminance) {
      const { targetX, targetY } = p;
      if (targetX !== undefined && targetX >= 0 && targetY !== undefined && targetY >= 0) {
        return (
          <Circle
            x={(targetX - cropX) * scale}
            y={(targetY - cropY) * scale}
            radius={5}
            stroke={isSelected ? '#0ea5e9' : 'white'}
            strokeWidth={2}
            listening={false}
            onTouchEnd={handleMaskTouchEnd}
            onTouchStart={handleMaskTouchStart}
            shadowColor="black"
            shadowBlur={2}
            shadowOpacity={0.8}
          />
        );
      }
      return null;
    }
    return null;
  },
);
