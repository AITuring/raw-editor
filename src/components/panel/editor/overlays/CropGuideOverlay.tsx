import type { SVGProps } from 'react';
import type { CropGuideMode } from '../../../../types/crop';

interface CropGuideOverlayProps {
  width: number;
  height: number;
  mode: CropGuideMode;
  rotation: number;
  denseVisible?: boolean;
}

interface Point {
  x: number;
  y: number;
}

const PHI = (1 + Math.sqrt(5)) / 2;

function projectPointOntoLine(point: Point, start: Point, end: Point): Point {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const projection = lengthSquared === 0 ? 0 : ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;

  return {
    x: start.x + projection * dx,
    y: start.y + projection * dy,
  };
}

export default function CropGuideOverlay({
  width,
  height,
  mode,
  rotation,
  denseVisible = false,
}: CropGuideOverlayProps) {
  if (width <= 0 || height <= 0 || (mode === 'none' && !denseVisible)) return null;

  const strokeProps: SVGProps<SVGLineElement> = {
    fill: 'none',
    stroke: 'rgb(248 247 243 / 82%)',
    strokeWidth: 1,
    vectorEffect: 'non-scaling-stroke',
  };

  const renderDenseGrid = () => (
    <g opacity={0.62}>
      {Array.from({ length: 17 }, (_, index) => {
        const position = ((index + 1) / 18) * 100;
        return (
          <line key={`dense-v-${index}`} x1={`${position}%`} y1={0} x2={`${position}%`} y2={height} {...strokeProps} />
        );
      })}
      {Array.from({ length: 17 }, (_, index) => {
        const position = ((index + 1) / 18) * 100;
        return (
          <line key={`dense-h-${index}`} x1={0} y1={`${position}%`} x2={width} y2={`${position}%`} {...strokeProps} />
        );
      })}
    </g>
  );

  const renderThirds = () => (
    <g>
      {[1 / 3, 2 / 3].map((position) => (
        <line
          key={`thirds-v-${position}`}
          x1={width * position}
          y1={0}
          x2={width * position}
          y2={height}
          {...strokeProps}
        />
      ))}
      {[1 / 3, 2 / 3].map((position) => (
        <line
          key={`thirds-h-${position}`}
          x1={0}
          y1={height * position}
          x2={width}
          y2={height * position}
          {...strokeProps}
        />
      ))}
    </g>
  );

  const renderGrid = () => (
    <g opacity={0.78}>
      {[0.25, 0.5, 0.75].map((position) => (
        <line
          key={`grid-v-${position}`}
          x1={width * position}
          y1={0}
          x2={width * position}
          y2={height}
          {...strokeProps}
        />
      ))}
      {[0.25, 0.5, 0.75].map((position) => (
        <line
          key={`grid-h-${position}`}
          x1={0}
          y1={height * position}
          x2={width}
          y2={height * position}
          {...strokeProps}
        />
      ))}
    </g>
  );

  const renderDiagonal = () => {
    const diagonalSpan = Math.min(width, height);
    return (
      <g>
        <line x1={0} y1={0} x2={diagonalSpan} y2={diagonalSpan} {...strokeProps} />
        <line x1={width} y1={0} x2={width - diagonalSpan} y2={diagonalSpan} {...strokeProps} />
        <line x1={0} y1={height} x2={diagonalSpan} y2={height - diagonalSpan} {...strokeProps} />
        <line x1={width} y1={height} x2={width - diagonalSpan} y2={height - diagonalSpan} {...strokeProps} />
      </g>
    );
  };

  const renderPhiGrid = () => {
    const positions = [1 / (PHI * PHI), 1 / PHI];
    return (
      <g>
        {positions.map((position) => (
          <line
            key={`phi-v-${position}`}
            x1={width * position}
            y1={0}
            x2={width * position}
            y2={height}
            {...strokeProps}
          />
        ))}
        {positions.map((position) => (
          <line
            key={`phi-h-${position}`}
            x1={0}
            y1={height * position}
            x2={width}
            y2={height * position}
            {...strokeProps}
          />
        ))}
      </g>
    );
  };

  const renderGoldenTriangle = () => {
    const orientation = ((rotation % 2) + 2) % 2;
    const mainStart = orientation % 2 === 0 ? { x: 0, y: height } : { x: 0, y: 0 };
    const mainEnd = orientation % 2 === 0 ? { x: width, y: 0 } : { x: width, y: height };
    const firstCorner = orientation === 0 ? { x: 0, y: 0 } : { x: width, y: 0 };
    const secondCorner = { x: width - firstCorner.x, y: height - firstCorner.y };
    const firstIntersection = projectPointOntoLine(firstCorner, mainStart, mainEnd);
    const secondIntersection = projectPointOntoLine(secondCorner, mainStart, mainEnd);

    return (
      <g>
        <line x1={mainStart.x} y1={mainStart.y} x2={mainEnd.x} y2={mainEnd.y} {...strokeProps} />
        <line
          x1={firstCorner.x}
          y1={firstCorner.y}
          x2={firstIntersection.x}
          y2={firstIntersection.y}
          {...strokeProps}
        />
        <line
          x1={secondCorner.x}
          y1={secondCorner.y}
          x2={secondIntersection.x}
          y2={secondIntersection.y}
          {...strokeProps}
        />
      </g>
    );
  };

  const renderGoldenSpiral = () => {
    const orientation = ((rotation % 8) + 8) % 8;
    const quarterTurn = orientation % 4;
    const isMirrored = orientation >= 4;
    const baseWidth = 1000;
    const baseHeight = baseWidth / PHI;
    const path =
      'M 0 618.03 A 618.03 618.03 0 0 1 618.03 0 A 381.97 381.97 0 0 1 1000 381.97 A 236.06 236.06 0 0 1 763.94 618.03 A 145.91 145.91 0 0 1 618.03 472.12 A 90.15 90.15 0 0 1 708.18 381.97 A 55.76 55.76 0 0 1 763.94 437.73 A 34.39 34.39 0 0 1 729.55 472.12 A 21.37 21.37 0 0 1 708.18 450.75 A 13.12 13.12 0 0 1 721.30 437.77 A 8.11 8.11 0 0 1 729.41 445.88';
    const scaleX = quarterTurn % 2 === 0 ? width / baseWidth : height / baseWidth;
    const scaleY = quarterTurn % 2 === 0 ? height / baseHeight : width / baseHeight;
    const transform = `translate(${width / 2} ${height / 2}) rotate(${quarterTurn * 90}) scale(${isMirrored ? -scaleX : scaleX} ${scaleY}) translate(${-baseWidth / 2} ${-baseHeight / 2})`;

    return <path d={path} {...strokeProps} strokeLinecap="round" strokeLinejoin="round" transform={transform} />;
  };

  const renderSelectedGuide = () => {
    switch (mode) {
      case 'thirds':
        return renderThirds();
      case 'grid':
        return renderGrid();
      case 'diagonal':
        return renderDiagonal();
      case 'goldenTriangle':
        return renderGoldenTriangle();
      case 'phiGrid':
        return renderPhiGrid();
      case 'goldenSpiral':
        return renderGoldenSpiral();
      default:
        return null;
    }
  };

  return (
    <svg
      aria-hidden="true"
      className="crop-guide-overlay"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      width={width}
    >
      {denseVisible ? renderDenseGrid() : renderSelectedGuide()}
    </svg>
  );
}
