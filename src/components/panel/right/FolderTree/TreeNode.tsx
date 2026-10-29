import { Folder, FolderOpen, ChevronUp, ChevronDown } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import Text from '../../../ui/Text';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../../../types/typography';
import { type FolderTree } from './treeUtils';
import { ALBUM_ICONS } from './albumIcons';

interface TreeNodeProps {
  expandedFolders: Set<string>;
  isExpanded: boolean;
  node: FolderTree;
  onContextMenu(event: any, path: string, isPinned?: boolean): void;
  onFolderSelect(folder: string): void;
  onToggle(path: string): void;
  selectedPath: string | null;
  pinnedFolders: string[];
  showImageCounts: boolean;
  isInstantTransition: boolean;
  folderIcons: Record<string, string>;
}

export interface VisibleProps {
  index: number;
  total: number;
}

export function TreeNode({
  expandedFolders,
  isExpanded,
  node,
  onContextMenu,
  onFolderSelect,
  onToggle,
  selectedPath,
  pinnedFolders,
  showImageCounts,
  isInstantTransition,
  folderIcons,
}: TreeNodeProps) {
  const hasChildren = node.hasSubdirs || (node.children && node.children.length > 0);
  const isSelected = node.path === selectedPath;
  const isPinned = pinnedFolders.includes(node.path);

  const handleFolderIconClick = (e: any) => {
    e.stopPropagation();
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const handleNameClick = () => {
    onFolderSelect(node.path);
  };

  const handleNameDoubleClick = () => {
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const containerVariants: any = {
    closed: { height: 0, opacity: 0, transition: { duration: 0.2, ease: 'easeInOut' } },
    open: { height: 'auto', opacity: 1, transition: { duration: 0.25, ease: 'easeInOut' } },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -15 },
    visible: ({ index, total }: VisibleProps) => ({
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.25,
        delay: total < 8 ? index * 0.05 : 0,
      },
    }),
    exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
  };

  const currentFolderIconKey = folderIcons[node.path];
  let ResolvedIcon: React.ElementType = isExpanded ? FolderOpen : Folder;
  if (currentFolderIconKey && ALBUM_ICONS[currentFolderIconKey]) {
    ResolvedIcon = ALBUM_ICONS[currentFolderIconKey];
  }
  const iconKey = currentFolderIconKey || (isExpanded ? 'folder-open' : 'folder-closed');

  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        className={clsx('flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected,
          'hover:bg-card-active': !isSelected,
        })}
        onClick={handleNameClick}
        onContextMenu={(e: any) => onContextMenu(e, node.path, isPinned)}
      >
        <div
          className={clsx(
            'relative w-5 h-5 flex items-center justify-center p-0.5 rounded-sm transition-colors shrink-0',
            {
              [TEXT_COLOR_KEYS[TextColors.secondary]]: !isExpanded,
              'hover:bg-surface-hover': !isSelected && hasChildren,
            },
          )}
          onClick={handleFolderIconClick}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={iconKey}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute"
            >
              <ResolvedIcon size={16} />
            </motion.div>
          </AnimatePresence>
        </div>

        <span onDoubleClick={handleNameDoubleClick} className="truncate select-none flex-1">
          <span className="truncate">{node.name}</span>
          {typeof node.imageCount === 'number' && node.imageCount > 0 && (
            <Text
              as="span"
              variant={TextVariants.small}
              color={TextColors.secondary}
              className={clsx(
                'ml-1 inline-block transition-[opacity,transform] duration-150 ease-out',
                showImageCounts ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-2',
              )}
            >
              ({node.imageCount})
            </Text>
          )}
        </span>

        {hasChildren && (
          <Text
            as="div"
            color={TextColors.secondary}
            className="p-0.5 rounded-sm hover:bg-surface/50"
            onClick={handleFolderIconClick}
          >
            {isExpanded ? <ChevronUp size={16} className="shrink-0" /> : <ChevronDown size={16} className="shrink-0" />}
          </Text>
        )}
      </div>

      <AnimatePresence initial={false}>
        {hasChildren && isExpanded && node.children && node.children.length > 0 && (
          <motion.div
            animate="open"
            className="pl-1 border-l-[1.5px] border-border-color/50 ml-3.75 overflow-hidden"
            exit="closed"
            initial={isInstantTransition ? 'open' : 'closed'}
            key="children-container"
            variants={containerVariants}
          >
            <div className="py-1">
              <AnimatePresence>
                {node?.children?.map((childNode: any, index: number) => (
                  <motion.div
                    animate="visible"
                    custom={{ index, total: node.children.length }}
                    exit="exit"
                    initial={isInstantTransition ? 'visible' : 'hidden'}
                    key={childNode.path}
                    layout={isInstantTransition ? false : 'position'}
                    variants={itemVariants}
                  >
                    <TreeNode
                      expandedFolders={expandedFolders}
                      isExpanded={expandedFolders.has(childNode.path)}
                      node={childNode}
                      onContextMenu={onContextMenu}
                      onFolderSelect={onFolderSelect}
                      onToggle={onToggle}
                      selectedPath={selectedPath}
                      pinnedFolders={pinnedFolders}
                      showImageCounts={showImageCounts}
                      isInstantTransition={isInstantTransition}
                      folderIcons={folderIcons}
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
