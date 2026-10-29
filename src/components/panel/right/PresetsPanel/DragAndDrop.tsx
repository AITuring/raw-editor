import { useCallback } from 'react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { PresetListType } from '../../../../hooks/usePresets';
import { Folder as FolderIcon, FolderOpen } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';
import { PresetItemDisplay } from './PresetItemDisplay';

interface DroppableFolderItemProps {
  children: any;
  folder: any;
  isExpanded: boolean;
  onContextMenu(event: any, folder: any): void;
  onToggle(id: string): void;
}

interface DraggablePresetItemProps {
  isGeneratingPreviews: boolean;
  onApply(preset: any): void;
  onContextMenu(event: any, preset: any): void;
  preset: any;
  previewUrl: string;
  isActive?: boolean;
  intensity?: number;
  onIntensityChange?: (val: number) => void;
  onDragStateChange?: (isDragging: boolean) => void;
}

interface FolderProps {
  folder: any;
}

export function FolderItemDisplay({ folder }: FolderProps) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-surface cursor-grabbing w-full">
      <div className="p-1">
        <FolderIcon size={18} />
      </div>
      <Text color={TextColors.primary} weight={TextWeights.medium} className="grow truncate select-none">
        {folder.name}
      </Text>
      <Text as="span" weight={TextWeights.medium} className="ml-auto pr-1">
        {folder.children?.length || 0}
      </Text>
    </div>
  );
}

export function DraggablePresetItem({
  preset,
  onApply,
  onContextMenu,
  previewUrl,
  isGeneratingPreviews,
  isActive,
  intensity,
  onIntensityChange,
  onDragStateChange,
}: DraggablePresetItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef: setDraggableNodeRef,
    isDragging,
  } = useDraggable({
    id: preset.id,
    data: { type: PresetListType.Preset, preset },
  });

  const { setNodeRef: setDroppableNodeRef, isOver } = useDroppable({
    data: { type: PresetListType.Preset, preset },
    id: preset.id,
  });

  const setCombinedRef = useCallback(
    (node: any) => {
      setDraggableNodeRef(node);
      setDroppableNodeRef(node);
    },
    [setDraggableNodeRef, setDroppableNodeRef],
  );

  const style = {
    borderRadius: '10px',
    opacity: isDragging ? 0.4 : 1,
    outline: isOver ? '2px solid var(--color-primary)' : '2px solid transparent',
    outlineOffset: '-2px',
    touchAction: 'none',
  };

  return (
    <div
      onClick={() => onApply(preset)}
      onContextMenu={(e: any) => onContextMenu(e, { preset })}
      ref={setCombinedRef}
      style={style}
    >
      <motion.div
        {...listeners}
        {...attributes}
        className="cursor-grab"
        whileTap={{ scale: isActive ? 1 : 0.98 }}
        transition={{ type: 'spring', stiffness: 400, damping: 17 }}
      >
        <PresetItemDisplay
          preset={preset}
          previewUrl={previewUrl}
          isGeneratingPreviews={isGeneratingPreviews}
          isActive={isActive}
          intensity={intensity}
          onIntensityChange={onIntensityChange}
          onDragStateChange={onDragStateChange}
        />
      </motion.div>
    </div>
  );
}

function DroppableFolderItem({ folder, onContextMenu, children, onToggle, isExpanded }: DroppableFolderItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef: setDraggableNodeRef,
    isDragging,
  } = useDraggable({
    data: { type: PresetListType.Folder, folder },
    id: folder.id,
  });

  const { setNodeRef: setDroppableNodeRef, isOver } = useDroppable({
    data: { type: PresetListType.Folder, folder },
    id: folder.id,
  });

  const style = {
    opacity: isDragging ? 0.4 : 1,
    touchAction: 'none',
  };

  const hasChildren = folder.children && folder.children.length > 0;

  return (
    <div
      className={`rounded-lg transition-colors ${isOver ? 'bg-surface-hover' : ''}`}
      ref={setDroppableNodeRef}
      style={style}
    >
      <div
        className="flex items-center gap-2 p-2 rounded-lg bg-surface cursor-pointer"
        onContextMenu={(e: any) => onContextMenu(e, { folder })}
      >
        <div className="p-1 cursor-grab" ref={setDraggableNodeRef} {...listeners} {...attributes}>
          {isExpanded ? (
            <FolderOpen
              className="text-primary"
              onClick={(e: any) => {
                e.stopPropagation();
                onToggle(folder.id);
              }}
              size={18}
            />
          ) : (
            <FolderIcon
              className="text-text-secondary"
              onClick={(e: any) => {
                e.stopPropagation();
                onToggle(folder.id);
              }}
              size={18}
            />
          )}
        </div>
        <Text
          color={TextColors.primary}
          weight={TextWeights.medium}
          className="grow truncate select-none"
          onClick={() => onToggle(folder.id)}
        >
          {folder.name}
        </Text>
        <Text as="span" variant={TextVariants.small} color={TextColors.secondary} className="ml-auto pr-1">
          {folder.children?.length || 0}
        </Text>
      </div>
      <AnimatePresence>
        {isExpanded && hasChildren && (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="ml-4 pl-2 border-l-[1.5px] border-border-color/50 space-y-2 overflow-hidden pt-2"
            exit={{ height: 0, opacity: 0 }}
            initial={{ height: 0, opacity: 0 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function RootDroppableArea({
  children,
  onContextMenu,
}: {
  children: React.ReactNode;
  onContextMenu: (e: any) => void;
}) {
  const { setNodeRef, isOver } = useDroppable({ id: 'root' });

  return (
    <div
      className={`ui-panel-body space-y-2 transition-colors ${isOver ? 'bg-surface-hover' : ''}`}
      onContextMenu={onContextMenu}
      ref={setNodeRef}
    >
      {children}
    </div>
  );
}

export { DroppableFolderItem };
