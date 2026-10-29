import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useDraggable, useDroppable } from '@dnd-kit/core';

import { Mask, SubMask, getMaskTypeName } from '../Masks';
import { MaskContainer } from '../../../../utils/adjustments';
import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';

export interface DragData {
  type: 'Container' | 'SubMask' | 'Creation';
  item?: MaskContainer | SubMask;
  maskType?: Mask;
  parentId?: string;
}

export function MasksListRoot({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  const { setNodeRef, isOver } = useDroppable({ id: 'mask-list-root' });

  return (
    <motion.div
      key="masks-list-container"
      ref={setNodeRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex-col transition-colors ${isOver ? 'bg-surface' : ''}`}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
}

export function NewMaskDropZone({ isOver }: { isOver: boolean }) {
  const { t } = useTranslation();
  return (
    <motion.div
      layout
      initial={{ opacity: 0, height: 0, marginTop: 0 }}
      animate={{ opacity: 1, height: 'auto', marginTop: '4px' }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={`p-3 rounded-lg text-center ${isOver ? 'border border-accent/80 bg-bg-tertiary/50' : ''}`}
    >
      <Text weight={TextWeights.medium}>{t('editor.masks.dropzoneText')}</Text>
    </motion.div>
  );
}

export function DraggableGridItem({ maskType, onClick, onRightClick, isDraggable, activeMaskContainerId }: any) {
  const { t } = useTranslation();
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `create-${maskType.id || maskType.type}`,
    data: { type: 'Creation', maskType: maskType.type },
    disabled: !isDraggable,
  });

  const tooltip = maskType.disabled
    ? t('editor.masks.comingSoon')
    : maskType.id === 'others'
      ? t('editor.masks.tooltips.showMore')
      : activeMaskContainerId
        ? t('editor.masks.tooltips.addToCurrent', { name: getMaskTypeName(maskType) })
        : t('editor.masks.tooltips.createNew', { name: getMaskTypeName(maskType) });

  return (
    <motion.div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      onClick={onClick}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
      }}
      onMouseDown={(event) => {
        if (event.button !== 2) return;
        onRightClick(event);
      }}
      className={`bg-surface text-text-primary rounded-lg p-2 flex flex-col items-center justify-center gap-2 aspect-square transition-colors
                ${maskType.disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-card-active active:bg-accent/20'} ${isDragging ? 'opacity-50' : ''}`}
      data-tooltip={tooltip}
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
    >
      <maskType.icon size={24} />{' '}
      <Text as="span" variant={TextVariants.small} color={TextColors.primary}>
        {getMaskTypeName(maskType)}
      </Text>
    </motion.div>
  );
}
