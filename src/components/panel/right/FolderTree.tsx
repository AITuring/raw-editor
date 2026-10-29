import { Search, X, Plus } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { useState, useMemo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import Text from '../../ui/Text';
import { TextVariants, TextWeights } from '../../../types/typography';
import { useShallow } from 'zustand/react/shallow';
import { useLibraryStore } from '../../../store/useLibraryStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { AlbumItem, Invokes, FolderTreeSort, SortDirection } from '../../ui/AppProperties';
import {
  type FolderTree,
  filterTree,
  getAutoExpandedPaths,
  filterAlbumTree,
  getAutoExpandedAlbumGroups,
  sortFolderTree,
} from './FolderTree/treeUtils';
import { FolderSortMenu } from './FolderTree/FolderSortMenu';
import { SectionHeader } from './FolderTree/SectionHeader';
import { AlbumTreeNode } from './FolderTree/AlbumTreeNode';
import { type VisibleProps, TreeNode } from './FolderTree/TreeNode';

export type { FolderTree } from './FolderTree/treeUtils';

interface FolderTreeProps {
  isResizing: boolean;
  onContextMenu(event: any, path: string | null, isPinned?: boolean): void;
  onAlbumContextMenu(event: any, item: AlbumItem | null): void;
  onFolderSelect(folder: string): void;
  onSelectAlbum(albumId: string, albumName: string, images: string[]): void;
  onToggleFolder(folder: string): void;
  onOpenFolder(): void;
  style: any;
  isInstantTransition: boolean;
}

export default function FolderTree({
  isResizing,
  onContextMenu,
  onAlbumContextMenu,
  onFolderSelect,
  onSelectAlbum,
  onToggleFolder,
  onOpenFolder,
  style,
  isInstantTransition,
}: FolderTreeProps) {
  const { t } = useTranslation();
  const { appSettings, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );
  const {
    folderTrees,
    pinnedFolderTrees,
    currentFolderPath: selectedPath,
    expandedFolders,
    isTreeLoading: isLoading,
    albumTree,
    activeAlbumId,
    expandedAlbumGroups,
  } = useLibraryStore(
    useShallow((state) => ({
      folderTrees: state.folderTrees,
      pinnedFolderTrees: state.pinnedFolderTrees,
      currentFolderPath: state.currentFolderPath,
      expandedFolders: state.expandedFolders,
      isTreeLoading: state.isTreeLoading,
      albumTree: state.albumTree,
      activeAlbumId: state.activeAlbumId,
      expandedAlbumGroups: state.expandedAlbumGroups,
    })),
  );

  const [searchQuery, setSearchQuery] = useState('');
  const [isHovering, setIsHovering] = useState(false);
  const [isSortMenuOpen, setIsSortMenuOpen] = useState(false);
  const pinnedFolders = appSettings?.pinnedFolders || [];
  const openSections = appSettings?.openTreeSections ?? ['current'];
  const showImageCounts = appSettings?.enableFolderImageCounts ?? false;
  const folderIcons = appSettings?.folderIcons || {};
  const folderTreeSort: FolderTreeSort = appSettings?.folderTreeSort || { key: 'name', order: SortDirection.Ascending };
  const showHeaderButtons = isHovering || isSortMenuOpen;

  useEffect(() => {
    invoke(Invokes.GetAlbums).then((res: any) => useLibraryStore.getState().setLibrary({ albumTree: res }));
  }, []);

  const toggleSection = (section: string) => {
    if (appSettings) {
      const isOpen = openSections.includes(section);
      const newSections = isOpen ? openSections.filter((s) => s !== section) : [...openSections, section];

      handleSettingsChange({ ...appSettings, openTreeSections: newSections });
    }
  };

  const handleEmptyAreaContextMenu = (e: any) => {
    if (e.target === e.currentTarget) {
      onContextMenu(e, null, false);
    }
  };

  const toggleAlbumGroup = (id: string) => {
    useLibraryStore.getState().setLibrary((state) => {
      const next = new Set(state.expandedAlbumGroups);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { expandedAlbumGroups: next };
    });
  };

  const trimmedQuery = searchQuery.trim();
  const isSearching = trimmedQuery.length > 1;

  const filteredTrees = useMemo(() => {
    let base = folderTrees;
    if (isSearching) {
      base = base.map((tree: any) => filterTree(tree, trimmedQuery)).filter((t: any) => t !== null);
    }
    return sortFolderTree(base, folderTreeSort);
  }, [folderTrees, trimmedQuery, isSearching, folderTreeSort]);

  const filteredPinnedTrees = useMemo(() => {
    let base = pinnedFolderTrees;
    if (isSearching) {
      base = base.map((pinnedTree) => filterTree(pinnedTree, trimmedQuery)).filter((t): t is FolderTree => t !== null);
    }
    return sortFolderTree(base, folderTreeSort);
  }, [pinnedFolderTrees, trimmedQuery, isSearching, folderTreeSort]);

  const searchAutoExpandedFolders = useMemo(() => {
    if (!isSearching) return new Set<string>();
    const newExpanded = new Set<string>();
    filteredTrees.forEach((t: any) => getAutoExpandedPaths(t, newExpanded));
    filteredPinnedTrees.forEach((pinned) => getAutoExpandedPaths(pinned, newExpanded));
    return newExpanded;
  }, [isSearching, filteredTrees, filteredPinnedTrees]);

  const effectiveExpandedFolders = useMemo(() => {
    return new Set([...expandedFolders, ...searchAutoExpandedFolders]);
  }, [expandedFolders, searchAutoExpandedFolders]);

  const filteredAlbumTree = useMemo(() => {
    let base = albumTree;
    if (isSearching) {
      base = base
        .map((item: AlbumItem) => filterAlbumTree(item, trimmedQuery))
        .filter((item): item is AlbumItem => item !== null);
    }
    return base;
  }, [albumTree, trimmedQuery, isSearching]);

  const searchAutoExpandedAlbumGroups = useMemo(() => {
    if (!isSearching) return new Set<string>();
    const newExpanded = new Set<string>();
    filteredAlbumTree.forEach((t: any) => getAutoExpandedAlbumGroups(t, newExpanded));
    return newExpanded;
  }, [isSearching, filteredAlbumTree]);

  const effectiveExpandedAlbumGroups = useMemo(() => {
    return new Set([...expandedAlbumGroups, ...searchAutoExpandedAlbumGroups]);
  }, [expandedAlbumGroups, searchAutoExpandedAlbumGroups]);

  useEffect(() => {
    if (isSearching && appSettings) {
      const hasPinnedResults = filteredPinnedTrees && filteredPinnedTrees.length > 0;
      const hasBaseResults = filteredTrees && filteredTrees.length > 0;
      const hasAlbumResults = filteredAlbumTree && filteredAlbumTree.length > 0;

      const newSections = [...openSections];
      let changed = false;

      if (hasPinnedResults && !newSections.includes('pinned')) {
        newSections.push('pinned');
        changed = true;
      }
      if (hasBaseResults && !newSections.includes('current')) {
        newSections.push('current');
        changed = true;
      }
      if (hasAlbumResults && !newSections.includes('albums')) {
        newSections.push('albums');
        changed = true;
      }

      if (changed) {
        handleSettingsChange({ ...appSettings, openTreeSections: newSections });
      }
    }
  }, [
    isSearching,
    filteredTrees,
    filteredPinnedTrees,
    filteredAlbumTree,
    openSections,
    handleSettingsChange,
    appSettings,
  ]);

  const isPinnedOpen = openSections.includes('pinned');
  const isCurrentOpen = openSections.includes('current');
  const isAlbumsOpen = openSections.includes('albums');

  const hasVisiblePinnedTrees = filteredPinnedTrees && filteredPinnedTrees.length > 0;
  const hasVisibleAlbums = filteredAlbumTree && filteredAlbumTree.length > 0;
  const showAlbumsSection = hasVisibleAlbums || (!isSearching && albumTree.length === 0);

  return (
    <div
      className={clsx(
        'ui-panel-root relative shrink-0 bg-bg-secondary',
        !isResizing && 'transition-[width] duration-300 ease-in-out',
      )}
      style={style}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <div className="ui-panel-header">
        <Text variant={TextVariants.heading}>{t('library.folders.sourcesTitle', 'Sources')}</Text>
      </div>

      <div className="ui-panel-body flex flex-col">
        <div className="pb-3">
          <div className="flex items-center">
            <div className="relative flex-1 min-w-0">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
              <input
                type="text"
                placeholder={t('library.folders.searchPlaceholder')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 w-full truncate rounded-md border border-border-color bg-surface pl-9 pr-8 text-xs focus:border-accent focus:outline-hidden"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-sm hover:bg-card-active"
                  data-tooltip={t('library.folders.tooltips.clearSearch')}
                >
                  <X size={16} className="text-text-secondary" />
                </button>
              )}
            </div>

            <AnimatePresence>
              {showHeaderButtons && (
                <motion.div
                  initial={{ width: 0, opacity: 0, marginLeft: 0 }}
                  animate={{ width: 'auto', opacity: 1, marginLeft: 4 }}
                  exit={{ width: 0, opacity: 0, marginLeft: 0 }}
                  transition={{ duration: 0.2, ease: 'easeInOut' }}
                  className={clsx(
                    'flex items-center shrink-0',
                    isSortMenuOpen ? 'overflow-visible' : 'overflow-hidden',
                  )}
                >
                  <FolderSortMenu
                    sort={folderTreeSort}
                    onChange={(newSort) => {
                      if (appSettings) handleSettingsChange({ ...appSettings, folderTreeSort: newSort });
                    }}
                    isOpen={isSortMenuOpen}
                    setIsOpen={setIsSortMenuOpen}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <LayoutGroup id="folder-tree">
          <div className="flex-1 overflow-y-auto" onContextMenu={handleEmptyAreaContextMenu}>
            {hasVisiblePinnedTrees && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.pinned')}
                    isOpen={isPinnedOpen}
                    onToggle={() => toggleSection('pinned')}
                  />
                </div>
                <AnimatePresence initial={false}>
                  {isPinnedOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="pt-1 pb-2">
                        <AnimatePresence>
                          {filteredPinnedTrees.map((pinnedTree, index) => (
                            <motion.div
                              key={pinnedTree.path}
                              animate="visible"
                              custom={{ index, total: filteredPinnedTrees.length }}
                              exit="exit"
                              initial={isInstantTransition ? 'visible' : 'hidden'}
                              layout={isInstantTransition ? false : 'position'}
                              variants={{
                                hidden: { opacity: 0, x: -15 },
                                visible: ({ index, total }: VisibleProps) => ({
                                  opacity: 1,
                                  x: 0,
                                  transition: { duration: 0.25, delay: total < 8 ? index * 0.05 : 0 },
                                }),
                                exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
                              }}
                            >
                              <TreeNode
                                expandedFolders={effectiveExpandedFolders}
                                isExpanded={effectiveExpandedFolders.has(pinnedTree.path)}
                                node={pinnedTree}
                                onContextMenu={onContextMenu}
                                onFolderSelect={onFolderSelect}
                                onToggle={onToggleFolder}
                                selectedPath={selectedPath}
                                pinnedFolders={pinnedFolders}
                                showImageCounts={showImageCounts && isHovering}
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
              </>
            )}

            {showAlbumsSection && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.albums')}
                    isOpen={isAlbumsOpen}
                    onToggle={() => toggleSection('albums')}
                  />
                </div>
                <AnimatePresence>
                  {isAlbumsOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                      onContextMenu={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onAlbumContextMenu(e, null);
                      }}
                    >
                      <div className="pt-1 pb-2">
                        <AnimatePresence>
                          {filteredAlbumTree.map((item: any) => (
                            <motion.div
                              key={item.id}
                              initial={{ opacity: 0, height: 0, x: -15 }}
                              animate={{ opacity: 1, height: 'auto', x: 0 }}
                              exit={{ opacity: 0, height: 0, x: -15, overflow: 'hidden' }}
                              transition={{ duration: 0.2 }}
                              layout="position"
                            >
                              <AlbumTreeNode
                                item={item}
                                expandedGroups={effectiveExpandedAlbumGroups}
                                onToggle={toggleAlbumGroup}
                                onSelectAlbum={onSelectAlbum}
                                onContextMenu={onAlbumContextMenu}
                                selectedAlbumId={activeAlbumId}
                                showImageCounts={showImageCounts && isHovering}
                              />
                            </motion.div>
                          ))}
                        </AnimatePresence>
                        {albumTree.length === 0 && !isSearching && (
                          <motion.div layout="position">
                            <Text variant={TextVariants.small} className="p-2 text-center">
                              {t('library.folders.albumsEmpty')}
                            </Text>
                          </motion.div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {filteredTrees && filteredTrees.length > 0 && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.folders')}
                    isOpen={isCurrentOpen}
                    onToggle={() => toggleSection('current')}
                  />
                </div>
                <AnimatePresence initial={false}>
                  {isCurrentOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="pt-1">
                        <AnimatePresence>
                          {filteredTrees.map((tree: any, index: number) => (
                            <motion.div
                              key={tree.path}
                              animate="visible"
                              custom={{ index, total: filteredTrees.length }}
                              exit="exit"
                              initial={isInstantTransition ? 'visible' : 'hidden'}
                              layout={isInstantTransition ? false : 'position'}
                              variants={{
                                hidden: { opacity: 0, x: -15 },
                                visible: ({ index, total }: VisibleProps) => ({
                                  opacity: 1,
                                  x: 0,
                                  transition: { duration: 0.25, delay: total < 8 ? index * 0.05 : 0 },
                                }),
                                exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
                              }}
                            >
                              <TreeNode
                                expandedFolders={effectiveExpandedFolders}
                                isExpanded={effectiveExpandedFolders.has(tree.path)}
                                node={tree}
                                onContextMenu={onContextMenu}
                                onFolderSelect={onFolderSelect}
                                onToggle={onToggleFolder}
                                selectedPath={selectedPath}
                                pinnedFolders={pinnedFolders}
                                showImageCounts={showImageCounts && isHovering}
                                isInstantTransition={isInstantTransition}
                                folderIcons={folderIcons}
                              />
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        <AnimatePresence initial={false}>
                          {isHovering && !isSearching && (
                            <motion.div
                              layout="position"
                              initial={{ opacity: 0, height: 0, overflow: 'hidden' }}
                              animate={{ opacity: 1, height: 'auto', overflow: 'hidden' }}
                              exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                              transition={{ duration: 0.2 }}
                            >
                              <Text
                                as="div"
                                weight={TextWeights.medium}
                                className="flex items-center gap-2 p-2 mt-1 rounded-md transition-colors transition-opacity opacity-70 hover:opacity-100 hover:bg-card-active cursor-pointer hover:text-text-primary"
                                onClick={(e: React.MouseEvent) => {
                                  e.stopPropagation();
                                  onOpenFolder();
                                }}
                              >
                                <div className="relative w-4 h-4 ml-1 shrink-0 flex items-center justify-center">
                                  <Plus size={16} />
                                </div>
                                <span className="select-none">{t('library.folders.addFolder')}</span>
                              </Text>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {!filteredTrees?.length && !hasVisiblePinnedTrees && !hasVisibleAlbums && isSearching && (
              <Text className="p-2 text-center">{t('library.folders.noFoldersFound')}</Text>
            )}

            {folderTrees.length === 0 && pinnedFolderTrees.length === 0 && !isSearching && (
              <div className="pt-1">
                {isLoading ? (
                  <Text className="animate-pulse p-2">{t('library.folders.loading')}</Text>
                ) : (
                  <Text className="p-2">{t('library.folders.openFolderInstruction')}</Text>
                )}
              </div>
            )}
          </div>
        </LayoutGroup>
      </div>
    </div>
  );
}
