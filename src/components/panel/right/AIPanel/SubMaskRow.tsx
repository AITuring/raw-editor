import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import {
  Circle,
  ClipboardPaste,
  Copy,
  FileEdit,
  Loader2,
  Minus,
  Plus,
  PlusSquare,
  RotateCcw,
  Trash2,
  SquaresIntersect,
} from 'lucide-react';

import { useContextMenu } from '../../../../context/ContextMenuContext';
import { Mask, SubMaskMode, MASK_ICON_MAP, getSubMaskName } from '../Masks';
import { OPTION_SEPARATOR } from '../../../ui/AppProperties';
import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';

export function SubMaskRow({
  subMask,
  index,
  totalCount,
  containerId,
  isActive,
  parentVisible,
  onSelect,
  updateSubMask,
  handleDelete,
  handleDuplicate,
  handleDuplicateAndInvert,
  handlePaste,
  handleCopy,
  hasCopiedSubMask,
  activeDragItem,
  analyzingSubMaskId,
  renamingId,
  setRenamingId,
  tempName,
  setTempName,
  isParentLoading,
}: any) {
  const { t } = useTranslation();
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: subMask.id,
    data: { type: 'SubMask', item: subMask, parentId: containerId },
  });
  const { setNodeRef: setDroppableRef, isOver } = useDroppable({
    id: subMask.id,
    data: { type: 'SubMask', item: subMask, parentId: containerId },
  });
  const setCombinedRef = (node: HTMLElement | null) => {
    setNodeRef(node);
    setDroppableRef(node);
  };
  const MaskIcon = MASK_ICON_MAP[subMask.type as Mask] || Circle;
  const { showContextMenu } = useContextMenu();
  const [isHovered, setIsHovered] = useState(false);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isDraggingContainer = activeDragItem?.type === 'Container';
  const isAnalyzing = subMask.id === analyzingSubMaskId || (isParentLoading && subMask.type === Mask.QuickEraser);

  const handleMouseEnter = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setIsHovered(true);
  };
  const handleMouseLeave = () => {
    hoverTimeoutRef.current = setTimeout(() => setIsHovered(false), 1000);
  };
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    };
  }, []);

  const handleRenameSubmit = () => {
    if (tempName.trim()) {
      const newName = tempName.trim();
      updateSubMask(subMask.id, { name: newName });
    }
    setRenamingId(null);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    showContextMenu(e.clientX, e.clientY, [
      {
        label: t('editor.ai.actions.rename'),
        icon: FileEdit,
        onClick: () => {
          setRenamingId(subMask.id);
          setTempName(getSubMaskName(subMask));
        },
      },
      { label: t('editor.ai.actions.duplicateComponent'), icon: PlusSquare, onClick: handleDuplicate },
      { label: t('editor.ai.actions.duplicateAndInvertComponent'), icon: RotateCcw, onClick: handleDuplicateAndInvert },
      { label: t('editor.ai.actions.copyComponent'), icon: Copy, onClick: handleCopy },
      {
        label: t('editor.ai.actions.pasteComponent'),
        icon: ClipboardPaste,
        disabled: !hasCopiedSubMask,
        onClick: handlePaste,
      },
      { type: OPTION_SEPARATOR },
      { label: t('editor.ai.actions.deleteComponent'), icon: Trash2, isDestructive: true, onClick: handleDelete },
    ]);
  };
  const showNumber = isHovered && totalCount > 1;

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, x: -15 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: -15, scale: 0.95, transition: { duration: 0.2 } }}
      ref={setCombinedRef}
      {...attributes}
      {...listeners}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`flex items-center gap-2 p-2 rounded-md transition-colors group cursor-pointer
            ${isActive ? 'bg-surface' : 'hover:bg-card-active'}
            ${isOver && !isDraggingContainer ? 'border-t-2 border-accent' : ''}
            ${isDragging ? 'opacity-40 z-50' : ''}
            ${parentVisible === false ? 'opacity-50' : ''}
            ${isDraggingContainer ? 'opacity-30 pointer-events-none' : ''}
            transition-opacity duration-300`}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onContextMenu={onContextMenu}
    >
      <Text
        as="div"
        variant={TextVariants.small}
        weight={TextWeights.bold}
        className="relative w-4 h-4 ml-1 shrink-0 flex items-center justify-center"
      >
        <AnimatePresence mode="wait" initial={false}>
          {isAnalyzing ? (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute"
            >
              <Loader2 size={16} className="animate-spin" />
            </motion.div>
          ) : showNumber ? (
            <motion.span
              key="number"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute"
            >
              {index}
            </motion.span>
          ) : (
            <motion.div
              key="icon"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute"
            >
              <MaskIcon size={16} />
            </motion.div>
          )}
        </AnimatePresence>
      </Text>
      {renamingId === subMask.id ? (
        <input
          autoFocus
          className="bg-bg-primary text-sm w-full rounded px-1 outline-none border border-accent"
          value={tempName}
          onChange={(e) => setTempName(e.target.value)}
          onBlur={handleRenameSubmit}
          onKeyDown={(e) => e.key === 'Enter' && handleRenameSubmit()}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <Text color={TextColors.primary} className="flex-1 truncate select-none">
          {getSubMaskName(subMask)}
        </Text>
      )}
      <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
        {index > 1 && (
          <button
            className="p-1 hover:text-text-primary text-text-secondary"
            data-tooltip={
              subMask.mode === SubMaskMode.Additive
                ? t('editor.ai.actions.switchToSubtract')
                : subMask.mode === SubMaskMode.Subtractive
                  ? t('editor.ai.actions.switchToIntersect')
                  : t('editor.ai.actions.switchToAdd')
            }
            onClick={(e) => {
              e.stopPropagation();
              updateSubMask(subMask.id, {
                mode:
                  subMask.mode === SubMaskMode.Additive
                    ? SubMaskMode.Subtractive
                    : subMask.mode === SubMaskMode.Subtractive
                      ? SubMaskMode.Intersect
                      : SubMaskMode.Additive,
              });
            }}
          >
            {subMask.mode === SubMaskMode.Additive ? (
              <Plus size={16} />
            ) : subMask.mode === SubMaskMode.Subtractive ? (
              <Minus size={16} />
            ) : (
              <SquaresIntersect size={16} />
            )}
          </button>
        )}
        <button
          className="p-1 hover:text-status-error text-text-secondary"
          data-tooltip={t('editor.ai.actions.deleteComponent')}
          onClick={(e) => {
            e.stopPropagation();
            handleDelete();
          }}
        >
          <Trash2 size={16} />
        </button>
      </div>
    </motion.div>
  );
}
