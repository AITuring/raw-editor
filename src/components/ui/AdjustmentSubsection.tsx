import clsx from 'clsx';
import { useRef } from 'react';

interface AdjustmentSubsectionProps {
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  description?: React.ReactNode;
  title?: React.ReactNode;
}

export function AdjustmentSubsection({ actions, children, className, description, title }: AdjustmentSubsectionProps) {
  return (
    <div className={clsx('camera-raw-subsection', className)}>
      {(title || actions) && (
        <div className="camera-raw-subsection-header">
          {title && <div className="camera-raw-subsection-title">{title}</div>}
          {actions && <div className="camera-raw-subsection-actions">{actions}</div>}
        </div>
      )}
      {description && <div className="camera-raw-subsection-description">{description}</div>}
      {children}
    </div>
  );
}

export interface AdjustmentTab<T extends string> {
  id: T;
  label: string;
}

interface AdjustmentTabsProps<T extends string> {
  ariaLabel: string;
  onChange: (value: T) => void;
  tabs: readonly AdjustmentTab<T>[];
  value: T;
}

export function AdjustmentTabs<T extends string>({ ariaLabel, onChange, tabs, value }: AdjustmentTabsProps<T>) {
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const focusTab = (index: number) => {
    const wrappedIndex = (index + tabs.length) % tabs.length;
    const tab = tabs[wrappedIndex];
    onChange(tab.id);
    buttonRefs.current[wrappedIndex]?.focus();
  };

  return (
    <div aria-label={ariaLabel} className="camera-raw-tabs" role="radiogroup">
      {tabs.map((tab, index) => {
        const isSelected = tab.id === value;
        return (
          <button
            aria-checked={isSelected}
            className="camera-raw-tab"
            key={tab.id}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowRight') {
                event.preventDefault();
                focusTab(index + 1);
              } else if (event.key === 'ArrowLeft') {
                event.preventDefault();
                focusTab(index - 1);
              } else if (event.key === 'Home') {
                event.preventDefault();
                focusTab(0);
              } else if (event.key === 'End') {
                event.preventDefault();
                focusTab(tabs.length - 1);
              }
            }}
            ref={(element) => {
              buttonRefs.current[index] = element;
            }}
            role="radio"
            tabIndex={isSelected ? 0 : -1}
            type="button"
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
