import React from 'react';
import clsx from 'clsx';
import Text from './Text';
import { TextVariants } from '../../types/typography';

interface SwitchProps {
  checked: boolean;
  className?: string;
  disabled?: boolean;
  id?: string;
  label: string;
  onChange(val: boolean): any;
  tooltip?: string;
  trackClassName?: string;
}

/**
 * A beautiful, reusable, and accessible toggle switch component.
 *
 * @param {string} label - The text label for the switch.
 * @param {boolean} checked - The current state of the switch.
 * @param {function(boolean): void} onChange - Callback function that receives the new boolean state.
 * @param {boolean} [disabled=false] - Whether the switch is interactive.
 * @param {string} [className=''] - Additional classes for the container.
 * @param {string} [trackClassName] - Custom classes for the switch's background track.
 */
const Switch = ({
  checked,
  className = '',
  disabled = false,
  id,
  label,
  onChange,
  tooltip,
  trackClassName,
}: SwitchProps) => {
  const uniqueId = id || `switch-${label.replace(/\s+/g, '-').toLowerCase()}`;

  return (
    <label
      className={clsx(
        'flex items-center justify-between',
        disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
        className,
      )}
      htmlFor={uniqueId}
      data-tooltip={tooltip}
    >
      <Text variant={TextVariants.label} className="select-none">
        {label}
      </Text>
      <div className="relative h-4 w-8 shrink-0">
        <input
          checked={checked}
          className="peer sr-only"
          disabled={disabled}
          id={uniqueId}
          onChange={(e: any) => !disabled && onChange(e.target.checked)}
          type="checkbox"
        />
        <div
          className={clsx(
            'h-full w-full rounded-full border border-border-color transition-colors duration-150 peer-focus-visible:ring-1 peer-focus-visible:ring-accent peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-bg-secondary',
            checked ? 'bg-accent/35' : 'bg-card-active/60',
            trackClassName,
          )}
        />
        <div
          className={clsx('absolute left-0.5 top-0.5 h-3 w-3 rounded-full transition-transform duration-150 ease-out', {
            'bg-accent': checked,
            'bg-text-secondary/80': !checked,
          })}
          style={{ transform: `translateX(${checked ? 16 : 0}px)` }}
        />
      </div>
    </label>
  );
};

export default Switch;
