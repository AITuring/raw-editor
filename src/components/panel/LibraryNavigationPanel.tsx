import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import { FolderTree } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useUIStore } from '../../store/useUIStore';
import { Panel } from '../ui/AppProperties';

interface LibraryNavigationPanelProps {
  width: number;
  onWidthChange(event: ReactPointerEvent<HTMLDivElement>): void;
  renderPanel(panel: Panel): ReactNode;
}

const COLLAPSED_WIDTH = 200;

export default function LibraryNavigationPanel({ width, onWidthChange, renderPanel }: LibraryNavigationPanelProps) {
  const { t } = useTranslation();
  const setUI = useUIStore((state) => state.setUI);
  const isCollapsed = width < COLLAPSED_WIDTH;

  return (
    <aside
      aria-label={t('library.folders.sourcesTitle')}
      className="ui-chrome-panel flex h-full shrink-0"
      style={{ width }}
    >
      {isCollapsed ? (
        <button
          aria-label={t('library.folders.sourcesTitle')}
          className="flex min-w-0 flex-1 items-start justify-center pt-3 text-text-secondary transition-colors hover:bg-surface hover:text-text-primary"
          data-tooltip={t('library.folders.sourcesTitle')}
          onClick={() => setUI({ leftPanelWidth: 280 })}
          type="button"
        >
          <FolderTree aria-hidden="true" size={19} />
        </button>
      ) : (
        <div className="min-w-0 flex-1 overflow-hidden">{renderPanel(Panel.FolderTree)}</div>
      )}
      <div aria-hidden="true" className="app-resizer is-vertical" onPointerDown={onWidthChange} />
    </aside>
  );
}
