import { Folder, FolderOpen, ChevronUp, ChevronDown, Album as AlbumIcon } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';
import { AlbumItem, AlbumGroup, Album } from '../../../ui/AppProperties';
import { ALBUM_ICONS } from './albumIcons';

const getAlbumImageCount = (item: any): number => {
  if (item.type === 'album' && item.images) {
    return item.images.length;
  }
  if (item.type === 'group' && item.children) {
    return item.children.reduce((sum: number, child: any) => sum + getAlbumImageCount(child), 0);
  }
  return 0;
};

export function AlbumTreeNode({
  item,
  expandedGroups,
  onToggle,
  onSelectAlbum,
  onContextMenu,
  selectedAlbumId,
  showImageCounts,
}: {
  item: AlbumItem;
  expandedGroups: Set<string>;
  onToggle: (id: string) => void;
  onSelectAlbum: (id: string, name: string, images: string[]) => void;
  onContextMenu: (e: any, item: AlbumItem) => void;
  selectedAlbumId: string | null;
  showImageCounts: boolean;
}) {
  const isGroup = item.type === 'group';
  const isExpanded = expandedGroups.has(item.id);
  const isSelected = item.id === selectedAlbumId;
  const imageCount = getAlbumImageCount(item);

  let ItemIcon: React.ElementType = isGroup ? (isExpanded ? FolderOpen : Folder) : AlbumIcon;
  if (item.icon && ALBUM_ICONS[item.icon]) {
    ItemIcon = ALBUM_ICONS[item.icon];
  }
  const iconKey = item.icon || (isGroup ? (isExpanded ? 'group-open' : 'group-closed') : 'album');

  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        className={clsx('flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected,
          'hover:bg-card-active': !isSelected,
        })}
        onClick={() => (isGroup ? onToggle(item.id) : onSelectAlbum(item.id, item.name, (item as Album).images))}
        onContextMenu={(e) => onContextMenu(e, item)}
      >
        <div className="relative w-5 h-5 flex items-center justify-center p-0.5 rounded-sm text-text-secondary shrink-0">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={iconKey}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute"
            >
              <ItemIcon size={16} />
            </motion.div>
          </AnimatePresence>
        </div>

        <span onDoubleClick={() => isGroup && onToggle(item.id)} className="truncate flex-1 select-none">
          <span className="truncate">{item.name}</span>
          {imageCount > 0 && (
            <Text
              as="span"
              variant={TextVariants.small}
              color={TextColors.secondary}
              className={clsx(
                'ml-1 inline-block transition-[opacity,transform] duration-150 ease-out',
                showImageCounts ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-2',
              )}
            >
              ({imageCount})
            </Text>
          )}
        </span>

        {isGroup && (
          <div
            className="text-text-secondary p-0.5 rounded-sm hover:bg-surface/50"
            onClick={(e) => {
              e.stopPropagation();
              onToggle(item.id);
            }}
          >
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        )}
      </div>

      <AnimatePresence>
        {isGroup && isExpanded && (item as AlbumGroup).children.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="pl-1 border-l-[1.5px] border-border-color/50 ml-3.75 overflow-hidden"
          >
            <div className="py-1">
              <AnimatePresence>
                {(item as AlbumGroup).children.map((child) => (
                  <motion.div
                    key={child.id}
                    initial={{ opacity: 0, height: 0, x: -10 }}
                    animate={{ opacity: 1, height: 'auto', x: 0 }}
                    exit={{ opacity: 0, height: 0, x: -10, overflow: 'hidden' }}
                    transition={{ duration: 0.2 }}
                  >
                    <AlbumTreeNode
                      item={child}
                      expandedGroups={expandedGroups}
                      onToggle={onToggle}
                      onSelectAlbum={onSelectAlbum}
                      onContextMenu={onContextMenu}
                      selectedAlbumId={selectedAlbumId}
                      showImageCounts={showImageCounts}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Text>
  );
}
