import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useDraggable, useDroppable } from '@dnd-kit/core';

import { Mask, SubMask, formatMaskTypeName } from '../Masks';
import { AiPatch } from '../../../../utils/adjustments';
import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';

export interface DragData {
  type: 'Container' | 'SubMask' | 'Creation';
  item?: AiPatch | SubMask;
  maskType?: Mask;
  parentId?: string;
}

export function AiListRoot({
  children,
  onClick,
  activeDragItem,
  hasPatches,
}: {
  children: React.ReactNode;
  onClick: () => void;
  activeDragItem: any;
  hasPatches: boolean;
}) {
  const { setNodeRef, isOver } = useDroppable({ id: 'ai-list-root' });

  return (
    <motion.div
      key="ai-list"
      ref={setNodeRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex flex-col transition-colors ${isOver ? 'bg-surface' : ''}`}
      onClick={onClick}
    >
      {children}
      <AnimatePresence>
        {activeDragItem?.type === 'Creation' && hasPatches && <NewMaskDropZone isOver={isOver} />}
      </AnimatePresence>
    </motion.div>
  );
}

function NewMaskDropZone({ isOver }: { isOver: boolean }) {
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
      <Text weight={TextWeights.medium}>{t('editor.ai.dropzoneText')}</Text>
    </motion.div>
  );
}

export function DraggableGridItem({ maskType, isGenerating, onClick }: any) {
  const { t } = useTranslation();
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `create-ai-${maskType.type}`,
    data: { type: 'Creation', maskType: maskType.type },
    disabled: isGenerating,
  });
  return (
    <motion.div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      onClick={onClick}
      className={`bg-surface text-text-primary rounded-lg p-2 flex flex-col items-center justify-center gap-2 aspect-square transition-colors
            ${
              maskType.disabled || isGenerating
                ? 'opacity-50 cursor-not-allowed'
                : 'hover:bg-card-active active:bg-accent/20'
            }
            ${isDragging ? 'opacity-50' : ''}`}
      data-tooltip={
        maskType.disabled
          ? t('editor.ai.comingSoon')
          : t('editor.ai.createNewTooltip', { name: formatMaskTypeName(maskType.type) })
      }
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
    >
      <maskType.icon size={24} />{' '}
      <Text as="span" variant={TextVariants.small} color={TextColors.primary}>
        {formatMaskTypeName(maskType.type)}
      </Text>
    </motion.div>
  );
}
