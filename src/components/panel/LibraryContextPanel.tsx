import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Panel } from '../ui/AppProperties';

interface LibraryContextPanelProps {
  panel: Panel;
  width: number;
  onClose(): void;
  onWidthChange(event: ReactPointerEvent<HTMLDivElement>): void;
  renderPanel(panel: Panel): ReactNode;
}

export default function LibraryContextPanel({
  panel,
  width,
  onClose,
  onWidthChange,
  renderPanel,
}: LibraryContextPanelProps) {
  const { t } = useTranslation();

  return (
    <div className="library-context-panel flex h-full shrink-0 overflow-hidden" style={{ width }}>
      <div aria-hidden="true" className="app-resizer is-vertical" onPointerDown={onWidthChange} />
      <aside className="ui-chrome-panel relative flex h-full min-w-0 flex-1 flex-col">
        <button
          aria-label={t('library.actions.closePanel')}
          className="ui-icon-button library-context-close"
          data-tooltip={t('library.actions.closePanel')}
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" size={16} />
        </button>
        <div className="min-h-0 flex-1 overflow-hidden">{renderPanel(panel)}</div>
      </aside>
    </div>
  );
}
