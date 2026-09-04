import { AnimatePresence, motion } from 'framer-motion';
import { Crop, Download, Eraser, Info, Layers, SlidersHorizontal, SwatchBook, type LucideIcon } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

import { BASIC_MODE } from '../../basic/runtime';
import { useUIStore } from '../../store/useUIStore';
import { Panel } from '../ui/AppProperties';
import DevelopHistogram from './DevelopHistogram';

interface DevelopPanelProps {
  activePanel: Panel | null;
  isResizing: boolean;
  onPanelSelect(panel: Panel): void;
  onWidthChange(event: React.PointerEvent<HTMLDivElement>): void;
  renderPanel(panel: Panel): React.ReactNode;
  width: number;
}

interface ToolDefinition {
  icon: LucideIcon;
  panel: Panel;
  titleKey: string;
}

const PRIMARY_TOOLS: ToolDefinition[] = [
  { panel: Panel.Adjustments, icon: SlidersHorizontal, titleKey: 'editor.switcher.tooltips.adjust' },
  { panel: Panel.Crop, icon: Crop, titleKey: 'editor.switcher.tooltips.crop' },
  { panel: Panel.Masks, icon: Layers, titleKey: 'editor.switcher.tooltips.masks' },
  ...(!BASIC_MODE
    ? [{ panel: Panel.Ai, icon: Eraser, titleKey: 'editor.switcher.tooltips.inpaint' } satisfies ToolDefinition]
    : []),
  { panel: Panel.Presets, icon: SwatchBook, titleKey: 'editor.switcher.tooltips.presets' },
];

const SECONDARY_TOOLS: ToolDefinition[] = [
  { panel: Panel.Metadata, icon: Info, titleKey: 'editor.switcher.tooltips.info' },
  { panel: Panel.Export, icon: Download, titleKey: 'editor.switcher.tooltips.export' },
];

function DevelopToolButton({
  activePanel,
  definition,
  onSelect,
}: {
  activePanel: Panel | null;
  definition: ToolDefinition;
  onSelect(panel: Panel): void;
}) {
  const { t } = useTranslation();
  const Icon = definition.icon;
  const isActive = activePanel === definition.panel;
  const label = t(definition.titleKey as never) as string;

  return (
    <button
      aria-label={label}
      aria-pressed={isActive}
      className={clsx('develop-tool-button', isActive && 'is-active')}
      data-tooltip={label}
      onClick={() => onSelect(definition.panel)}
      type="button"
    >
      <Icon aria-hidden="true" size={17} strokeWidth={1.7} />
    </button>
  );
}

export default function DevelopPanel({
  activePanel,
  isResizing,
  onPanelSelect,
  onWidthChange,
  renderPanel,
  width,
}: DevelopPanelProps) {
  const { t } = useTranslation();
  const isFullScreen = useUIStore((state) => state.isFullScreen);
  const isInstantTransition = useUIStore((state) => state.isInstantTransition);
  const isCollapsed = width < 200;
  const renderedPanel = activePanel ?? Panel.Adjustments;
  const showsHistogram = ![Panel.Export, Panel.Metadata].includes(renderedPanel);

  return (
    <aside
      aria-label={t('editor.adjustments.title')}
      className={clsx(
        'develop-panel-shell',
        isCollapsed && 'is-collapsed',
        isFullScreen && 'is-hidden',
        !isInstantTransition && !isResizing && 'is-animated',
      )}
      style={{ width: isFullScreen ? 0 : isCollapsed ? 'var(--ui-tool-rail-width)' : width }}
    >
      <div aria-hidden="true" className="develop-panel-resizer" onPointerDown={onWidthChange} />

      <div className="develop-panel-content">
        <AnimatePresence initial={false} mode="wait">
          <motion.div
            animate={{ opacity: 1 }}
            className="absolute inset-0 flex flex-col overflow-hidden"
            exit={{ opacity: 0 }}
            initial={isInstantTransition || isResizing ? false : { opacity: 0 }}
            key={renderedPanel}
            transition={{ duration: 0.12, ease: [0.22, 1, 0.36, 1] }}
          >
            {showsHistogram && <DevelopHistogram />}
            <div className="relative min-h-0 flex-1 overflow-hidden">{renderPanel(renderedPanel)}</div>
          </motion.div>
        </AnimatePresence>
      </div>

      <nav className="develop-tool-rail">
        <div className="develop-tool-group">
          {PRIMARY_TOOLS.map((definition) => (
            <DevelopToolButton
              activePanel={renderedPanel}
              definition={definition}
              key={definition.panel}
              onSelect={onPanelSelect}
            />
          ))}
        </div>

        <div className="develop-tool-group develop-tool-group-secondary">
          {SECONDARY_TOOLS.map((definition) => (
            <DevelopToolButton
              activePanel={renderedPanel}
              definition={definition}
              key={definition.panel}
              onSelect={onPanelSelect}
            />
          ))}
        </div>
      </nav>
    </aside>
  );
}
