import type { CSSProperties, ReactNode } from 'react';
import clsx from 'clsx';

interface TaskProgressStyle extends CSSProperties {
  '--task-progress-value': number;
}

interface TaskProgressProps {
  ariaLabel: string;
  className?: string;
  compact?: boolean;
  current?: number | null;
  detail?: ReactNode;
  indeterminate?: boolean;
  label?: ReactNode;
  showPercentage?: boolean;
  total?: number | null;
  value?: number | null;
}

const clampPercentage = (value: number) => Math.min(100, Math.max(0, value));

export default function TaskProgress({
  ariaLabel,
  className,
  compact = false,
  current,
  detail,
  indeterminate,
  label,
  showPercentage = true,
  total,
  value,
}: TaskProgressProps) {
  const explicitValue = typeof value === 'number' && Number.isFinite(value) ? value : null;
  const countValue =
    typeof current === 'number' && Number.isFinite(current) && typeof total === 'number' && total > 0
      ? (current / total) * 100
      : null;
  const percentage = clampPercentage(explicitValue ?? countValue ?? 0);
  const isDeterminate = indeterminate !== true && (explicitValue !== null || countValue !== null);
  const roundedPercentage = Math.round(percentage);
  const meta = detail ?? (isDeterminate && showPercentage ? `${roundedPercentage}%` : null);
  const style = { '--task-progress-value': percentage / 100 } as TaskProgressStyle;

  return (
    <div
      className={clsx('task-progress', compact && 'task-progress--compact', className)}
      data-state={isDeterminate ? 'determinate' : 'indeterminate'}
    >
      {(label || meta) && (
        <div className="task-progress__header">
          <span className="task-progress__label">{label}</span>
          {meta && <span className="task-progress__meta">{meta}</span>}
        </div>
      )}
      <div
        aria-label={ariaLabel}
        aria-valuemax={isDeterminate ? 100 : undefined}
        aria-valuemin={isDeterminate ? 0 : undefined}
        aria-valuenow={isDeterminate ? roundedPercentage : undefined}
        aria-valuetext={typeof label === 'string' ? label : undefined}
        className="task-progress__track"
        role="progressbar"
      >
        {isDeterminate ? (
          <span className="task-progress__fill" style={style} />
        ) : (
          <span className="task-progress__indeterminate" />
        )}
      </div>
    </div>
  );
}
