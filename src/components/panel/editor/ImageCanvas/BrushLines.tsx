import { memo, useMemo } from 'react';
import { Line, Group } from 'react-konva';
import type { DrawnLine } from './types';

export const OptimizedBrushLine = memo(
  ({ line, scale, cropX, cropY }: { line: DrawnLine; scale: number; cropX: number; cropY: number }) => {
    const flattenedPoints = useMemo(() => {
      const pts = new Float32Array(line.points.length * 2);
      for (let i = 0; i < line.points.length; i++) {
        pts[i * 2] = (line.points[i].x - cropX) * scale;
        pts[i * 2 + 1] = (line.points[i].y - cropY) * scale;
      }
      return Array.from(pts);
    }, [line.points, scale, cropX, cropY]);

    return (
      <Line
        hitStrokeWidth={line.brushSize * scale}
        lineCap="round"
        lineJoin="round"
        points={flattenedPoints}
        stroke="transparent"
        strokeScaleEnabled={false}
        perfectDrawEnabled={false}
        shadowForStrokeEnabled={false}
      />
    );
  },
);

export const SourcePreviewLine = memo(
  ({
    line,
    scale,
    cropX,
    cropY,
    dx,
    dy,
  }: {
    line: DrawnLine;
    scale: number;
    cropX: number;
    cropY: number;
    dx: number;
    dy: number;
  }) => {
    const flattenedPoints = useMemo(() => {
      const pts = new Float32Array(line.points.length * 2);
      for (let i = 0; i < line.points.length; i++) {
        pts[i * 2] = (line.points[i].x + dx - cropX) * scale;
        pts[i * 2 + 1] = (line.points[i].y + dy - cropY) * scale;
      }
      return Array.from(pts);
    }, [line.points, scale, cropX, cropY, dx, dy]);

    return (
      <Group>
        <Line
          lineCap="round"
          lineJoin="round"
          points={flattenedPoints}
          stroke="rgba(255, 255, 255, 0.15)"
          strokeWidth={line.brushSize * scale}
          strokeScaleEnabled={false}
          perfectDrawEnabled={false}
          shadowForStrokeEnabled={false}
        />
        <Line
          lineCap="round"
          lineJoin="round"
          points={flattenedPoints}
          stroke="white"
          strokeWidth={1.5}
          dash={[4, 4]}
          opacity={0.8}
          strokeScaleEnabled={false}
          perfectDrawEnabled={false}
          shadowForStrokeEnabled={false}
        />
      </Group>
    );
  },
);
