import { ChevronDown, Eye, EyeOff } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

interface CollapsibleSectionProps {
  canToggleVisibility?: boolean;
  children: React.ReactNode;
  isContentVisible: boolean;
  isOpen: boolean;
  onContextMenu?: React.MouseEventHandler<HTMLDivElement>;
  onToggle(): void;
  onToggleVisibility?(): void;
  title: string;
}

export default function CollapsibleSection({
  canToggleVisibility = true,
  children,
  isContentVisible,
  isOpen,
  onContextMenu,
  onToggle,
  onToggleVisibility = () => {},
  title,
}: CollapsibleSectionProps) {
  const { t } = useTranslation();
  const visibilityLabel = isContentVisible
    ? t('ui.collapsibleSection.disableSection')
    : t('ui.collapsibleSection.enableSection');

  return (
    <section
      className={clsx('develop-collapsible group/section', !isContentVisible && 'is-disabled')}
      onContextMenu={onContextMenu}
    >
      <div className="develop-collapsible-header">
        <button aria-expanded={isOpen} className="develop-collapsible-toggle" onClick={onToggle} type="button">
          <span className="truncate">{title}</span>
          <ChevronDown
            aria-hidden="true"
            className={clsx('develop-collapsible-chevron', isOpen && 'is-open')}
            size={15}
            strokeWidth={1.8}
          />
        </button>

        {canToggleVisibility && (
          <button
            aria-label={visibilityLabel}
            aria-pressed={isContentVisible}
            className="develop-collapsible-visibility"
            data-tooltip={visibilityLabel}
            onClick={onToggleVisibility}
            type="button"
          >
            {isContentVisible ? (
              <Eye aria-hidden="true" size={14} strokeWidth={1.7} />
            ) : (
              <EyeOff aria-hidden="true" size={14} strokeWidth={1.7} />
            )}
          </button>
        )}
      </div>

      {isOpen && (
        <div className={clsx('develop-collapsible-content', !isContentVisible && 'pointer-events-none opacity-35')}>
          {children}
        </div>
      )}
    </section>
  );
}
