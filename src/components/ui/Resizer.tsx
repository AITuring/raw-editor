import clsx from 'clsx';
import type { PointerEventHandler } from 'react';
import { Orientation } from './AppProperties';

interface ResizerProps {
  direction: Orientation;
  onMouseDown: PointerEventHandler<HTMLDivElement>;
}

const Resizer = ({ direction, onMouseDown }: ResizerProps) => (
  <div
    className={clsx('app-resizer', {
      'is-vertical': direction === Orientation.Vertical,
      'is-horizontal': direction === Orientation.Horizontal,
    })}
    role="separator"
    aria-orientation={direction === Orientation.Vertical ? 'vertical' : 'horizontal'}
    onPointerDown={onMouseDown}
    style={{ touchAction: 'none' }}
  />
);

export default Resizer;
