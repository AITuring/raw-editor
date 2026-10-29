import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import {
  Circle,
  ClipboardPaste,
  Copy,
  Eye,
  EyeOff,
  FileEdit,
  Plus,
  PlusSquare,
  RotateCcw,
  Trash2,
  Wand2,
  FolderOpen,
} from 'lucide-react';

import { useContextMenu } from '../../../../context/ContextMenuContext';
import { Mask, SubMask, MASK_ICON_MAP } from '../Masks';
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
  handlePastePatch,
  copyPatchToClipboard,
  copiedPatch,
  activeDragItem,
  activeSubMaskId,
  activePatchContainerId,
  onSelectContainer,
  onSelectSubMask,
  updateSubMask,
  handleDeleteSubMask,
  handleDuplicateSubMask,
  handleDuplicateAndInvertSubMask,
  handlePasteSubMask,
  copySubMaskToClipboard,
  copiedSubMask,
  analyzingSubMaskId,
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

  const isStandalone = container.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(container.subMasks[0].type);
  const firstSubMask = container.subMasks[0];
  const isRowSelected = isStandalone ? container.id === activePatchContainerId : isSelected;

  const setCombinedRef = (node: HTMLElement | null) => {
    setDroppableRef(node);
    setDraggableRef(node);
  };

  const handleRenameSubmit = () => {
    if (tempName.trim()) {
      updateContainer(container.id, { name: tempName.trim() });
    }
    setRenamingId(null);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const menuOptions: any[] = [
      {
        label: t('editor.ai.actions.rename'),
        icon: FileEdit,
        onClick: () => {
          setRenamingId(container.id);
          setTempName(container.name);
        },
      },
      { label: t('editor.ai.actions.duplicateEdit'), icon: PlusSquare, onClick: () => handleDuplicate(container) },
      {
        label: t('editor.ai.actions.duplicateAndInvertEdit'),
        icon: RotateCcw,
        onClick: () => handleDuplicateAndInvert(container),
      },
      { label: t('editor.ai.actions.copyEdit'), icon: Copy, onClick: () => copyPatchToClipboard(container) },
      {
        label: t('editor.ai.actions.pasteEdit'),
        icon: ClipboardPaste,
        disabled: !copiedPatch,
        onClick: () => handlePastePatch(container.id),
      },
      { type: OPTION_SEPARATOR },
    ];

    if (!isStandalone) {
      menuOptions.push({
        label: t('editor.ai.actions.resetSelection'),
        icon: RotateCcw,
        onClick: () => updateContainer(container.id, { subMasks: [] }),
      });
    }

    menuOptions.push({
      label: t('editor.ai.actions.deleteEdit'),
      icon: Trash2,
      isDestructive: true,
      onClick: () => handleDelete(container.id),
    });

    showContextMenu(e.clientX, e.clientY, menuOptions);
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
      if (!isStandalone) {
        borderClass = 'bg-card-active border border-accent/50';
      }
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
                ${isRowSelected ? 'bg-surface' : 'hover:bg-card-active'}
                ${borderClass}`}
        onClick={(e) => {
          e.stopPropagation();
          if (isStandalone) {
            onSelectContainer(container.id);
            onSelectSubMask(firstSubMask.id);
          } else {
            onSelect();
          }
        }}
        onContextMenu={onContextMenu}
      >
        <Text
          as="div"
          color={hasActiveChild || isExpanded || isStandalone ? TextColors.primary : TextColors.secondary}
          onClick={(e) => {
            e.stopPropagation();
            if (isStandalone) {
              onSelectContainer(container.id);
              onSelectSubMask(firstSubMask.id);
            } else {
              onToggle();
            }
          }}
          className="p-0.5 rounded transition-colors cursor-pointer"
        >
          {isStandalone ? (
            (() => {
              const StandaloneIcon = MASK_ICON_MAP[firstSubMask.type as Mask] || Circle;
              return <StandaloneIcon size={18} />;
            })()
          ) : isExpanded ? (
            <FolderOpen size={18} />
          ) : (
            <Wand2 size={18} />
          )}
        </Text>
        <div
          className="flex-1 min-w-0 cursor-pointer"
          onDoubleClick={(e) => {
            e.stopPropagation();
            if (!isStandalone) {
              onToggle();
            }
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
            data-tooltip={container.visible ? t('editor.ai.actions.hideEdit') : t('editor.ai.actions.showEdit')}
            onClick={(e) => {
              e.stopPropagation();
              updateContainer(container.id, { visible: !container.visible });
            }}
          >
            {container.visible ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <button
            className="p-1 hover:text-status-error text-text-secondary"
            data-tooltip={t('editor.ai.actions.deleteEdit')}
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
        {!isStandalone && isExpanded && (
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
                  isActive={activeSubMaskId === subMask.id}
                  parentVisible={container.visible}
                  activeDragItem={activeDragItem}
                  onSelect={() => {
                    onSelectContainer(container.id);
                    onSelectSubMask(subMask.id);
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
                  isParentLoading={container.isLoading}
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
                    <span className="select-none">{t('editor.ai.actions.addNewComponent')}</span>
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
