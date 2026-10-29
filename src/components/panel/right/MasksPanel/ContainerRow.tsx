import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import {
  ClipboardPaste,
  Copy,
  Eye,
  EyeOff,
  FileEdit,
  FolderOpen,
  Folder as FolderIcon,
  Plus,
  PlusSquare,
  RotateCcw,
  Trash2,
  SwatchBook,
} from 'lucide-react';

import { SubMask } from '../Masks';
import { INITIAL_MASK_ADJUSTMENTS } from '../../../../utils/adjustments';
import { useContextMenu } from '../../../../context/ContextMenuContext';
import { OPTION_SEPARATOR } from '../../../ui/AppProperties';
import Text from '../../../ui/Text';
import { TextColors, TextWeights } from '../../../../types/typography';
import { SubMaskRow } from './SubMaskRow';

export function ContainerRow({
  container,
  isSelected,
  hasActiveChild,
  isExpanded,
  onToggle,
  onSelect,
  renamingId,
  setRenamingId,
  tempName,
  setTempName,
  updateContainer,
  handleDelete,
  handleDuplicate,
  handleDuplicateAndInvert,
  handlePasteMask,
  copyMaskToClipboard,
  copiedMask,
  presets,
  setAdjustments,
  activeDragItem,
  activeMaskId,
  onSelectContainer,
  onSelectMask,
  updateSubMask,
  handleDeleteSubMask,
  handleDuplicateSubMask,
  handleDuplicateAndInvertSubMask,
  handlePasteSubMask,
  copySubMaskToClipboard,
  copiedSubMask,
  analyzingSubMaskId,
  setIsMaskControlHovered,
  onAddComponent,
}: any) {
  const { t } = useTranslation();
  const { setNodeRef: setDroppableRef, isOver } = useDroppable({
    id: container.id,
    data: { type: 'Container', item: container },
  });
  const {
    attributes,
    listeners,
    setNodeRef: setDraggableRef,
    isDragging,
  } = useDraggable({ id: container.id, data: { type: 'Container', item: container } });
  const { showContextMenu } = useContextMenu();

  const setCombinedRef = (node: HTMLElement | null) => {
    setDroppableRef(node);
    setDraggableRef(node);
  };

  const handleRenameSubmit = () => {
    if (tempName.trim()) {
      const newName = tempName.trim();
      setAdjustments((prev: any) => {
        const updatedMasks = prev.masks.map((m: any) => (m.id === container.id ? { ...m, name: newName } : m));
        return { ...prev, masks: updatedMasks };
      });
    }
    setRenamingId(null);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const generatePresetSubmenu = (list: any[]): any[] =>
      list
        .map((item) => {
          if (item.folder)
            return { label: item.folder.name, icon: FolderIcon, submenu: generatePresetSubmenu(item.folder.children) };
          if (item.preset || item.adjustments)
            return {
              label: item.name || item.preset.name,
              onClick: () => {
                const newAdj = { ...container.adjustments, ...(item.adjustments || item.preset.adjustments) };
                newAdj.sectionVisibility = { ...container.adjustments.sectionVisibility, ...newAdj.sectionVisibility };
                updateContainer(container.id, { adjustments: newAdj });
              },
            };
          return null;
        })
        .filter(Boolean);

    showContextMenu(e.clientX, e.clientY, [
      {
        label: t('editor.masks.actions.rename'),
        icon: FileEdit,
        onClick: () => {
          setRenamingId(container.id);
          setTempName(container.name);
        },
      },
      { label: t('editor.masks.actions.duplicateMask'), icon: PlusSquare, onClick: () => handleDuplicate(container) },
      {
        label: t('editor.masks.actions.duplicateAndInvertMask'),
        icon: RotateCcw,
        onClick: () => handleDuplicateAndInvert(container),
      },
      { label: t('editor.masks.actions.copyMask'), icon: Copy, onClick: () => copyMaskToClipboard(container) },
      {
        label: t('editor.masks.actions.pasteMask'),
        icon: ClipboardPaste,
        disabled: !copiedMask,
        onClick: () => handlePasteMask(container.id),
      },
      {
        label: t('editor.masks.actions.pasteMaskAdjustments'),
        icon: ClipboardPaste,
        disabled: !copiedMask,
        onClick: () => {
          if (copiedMask) {
            updateContainer(container.id, { adjustments: JSON.parse(JSON.stringify(copiedMask.adjustments)) });
          }
        },
      },
      {
        label: t('editor.masks.actions.applyPreset'),
        icon: SwatchBook,
        submenu: generatePresetSubmenu(presets).length
          ? generatePresetSubmenu(presets)
          : [{ label: t('editor.masks.actions.noPresets'), disabled: true }],
      },
      { type: OPTION_SEPARATOR },
      {
        label: t('editor.masks.actions.resetMaskAdjustments'),
        icon: RotateCcw,
        onClick: () =>
          updateContainer(container.id, { adjustments: JSON.parse(JSON.stringify(INITIAL_MASK_ADJUSTMENTS)) }),
      },
      {
        label: t('editor.masks.actions.deleteMask'),
        icon: Trash2,
        isDestructive: true,
        onClick: () => handleDelete(container.id),
      },
    ]);
  };

  const isDraggingContainer = activeDragItem?.type === 'Container';
  let borderClass = '';

  if (isOver) {
    if (isDraggingContainer) {
      borderClass = 'border-t-2 border-accent';
    } else if (
      (activeDragItem?.type === 'SubMask' && activeDragItem?.parentId !== container.id) ||
      activeDragItem?.type === 'Creation'
    ) {
      borderClass = 'bg-card-active border border-accent/50';
    }
  }

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: isDragging ? 0.4 : 1, height: 'auto' }}
      exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
      ref={setCombinedRef}
      className="overflow-hidden"
    >
      <div
        {...listeners}
        {...attributes}
        className={`flex items-center gap-2 p-2 rounded-md transition-colors group
             ${isSelected ? 'bg-surface' : 'hover:bg-card-active'}
             ${borderClass}`}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onContextMenu={onContextMenu}
      >
        <Text
          as="div"
          color={hasActiveChild || isExpanded ? TextColors.primary : TextColors.secondary}
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          className="p-0.5 rounded transition-colors cursor-pointer"
        >
          {isExpanded ? <FolderOpen size={18} /> : <FolderIcon size={18} />}
        </Text>
        <div
          className="flex-1 min-w-0 cursor-pointer"
          onDoubleClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
        >
          {renamingId === container.id ? (
            <input
              autoFocus
              className="bg-bg-primary text-sm w-full rounded-sm px-1 outline-hidden border border-accent"
              value={tempName}
              onChange={(e) => setTempName(e.target.value)}
              onBlur={handleRenameSubmit}
              onKeyDown={(e) => e.key === 'Enter' && handleRenameSubmit()}
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <Text color={TextColors.primary} weight={TextWeights.medium} className="truncate select-none">
              {container.name}
            </Text>
          )}
        </div>
        <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            className="p-1 hover:text-text-primary text-text-secondary"
            onMouseEnter={() => setIsMaskControlHovered(true)}
            onMouseLeave={() => setIsMaskControlHovered(false)}
            data-tooltip={container.visible ? t('editor.masks.actions.hideMask') : t('editor.masks.actions.showMask')}
            onClick={(e) => {
              e.stopPropagation();
              updateContainer(container.id, { visible: !container.visible });
            }}
          >
            {container.visible ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <button
            className="p-1 hover:text-status-error text-text-secondary"
            onClick={(e) => {
              e.stopPropagation();
              handleDelete(container.id);
            }}
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden pl-2 border-l-[1.5px] border-border-color/50 ml-3.75"
            layout
          >
            <AnimatePresence mode="popLayout" initial={false}>
              {container.subMasks.map((subMask: SubMask, index: number) => (
                <SubMaskRow
                  key={subMask.id}
                  subMask={subMask}
                  index={index + 1}
                  totalCount={container.subMasks.length}
                  containerId={container.id}
                  isActive={activeMaskId === subMask.id}
                  parentVisible={container.visible}
                  activeDragItem={activeDragItem}
                  onSelect={() => {
                    onSelectContainer(container.id);
                    onSelectMask(subMask.id);
                  }}
                  updateSubMask={updateSubMask}
                  handleDelete={() => handleDeleteSubMask(container.id, subMask.id)}
                  handleDuplicate={() => handleDuplicateSubMask(container.id, subMask, index + 1)}
                  handleDuplicateAndInvert={() => handleDuplicateAndInvertSubMask(container.id, subMask)}
                  handlePaste={() => handlePasteSubMask(container.id, index + 1)}
                  handleCopy={() => copySubMaskToClipboard(subMask)}
                  hasCopiedSubMask={!!copiedSubMask}
                  analyzingSubMaskId={analyzingSubMaskId}
                  renamingId={renamingId}
                  setRenamingId={setRenamingId}
                  tempName={tempName}
                  setTempName={setTempName}
                  setIsMaskControlHovered={setIsMaskControlHovered}
                />
              ))}
            </AnimatePresence>

            <AnimatePresence initial={false}>
              {(isSelected || hasActiveChild || container.subMasks.length === 0) && (
                <motion.div
                  key="add-component-btn"
                  layout="position"
                  initial={{ opacity: 0, height: 0, overflow: 'hidden' }}
                  animate={{ opacity: 1, height: 'auto', overflow: 'hidden' }}
                  exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                  transition={{ duration: 0.2 }}
                >
                  <Text
                    as="div"
                    weight={TextWeights.medium}
                    className="flex items-center gap-2 p-2 rounded-md transition-colors transition-opacity opacity-70 hover:opacity-100 hover:bg-card-active cursor-pointer hover:text-text-primary"
                    onClick={(e) => {
                      e.stopPropagation();
                      onAddComponent(e);
                    }}
                  >
                    <div className="relative w-4 h-4 ml-1 shrink-0 flex items-center justify-center">
                      <Plus size={16} />
                    </div>
                    <span className="select-none">{t('editor.masks.actions.addNewComponent')}</span>
                  </Text>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
