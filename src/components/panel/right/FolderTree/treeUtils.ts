import { AlbumItem, FolderTreeSort, SortDirection } from '../../../ui/AppProperties';

export interface FolderTree {
  children: FolderTree[];
  isDir: boolean;
  name: string;
  path: string;
  imageCount?: number;
  hasSubdirs?: boolean;
  modified?: number;
  created?: number;
}

export const filterTree = (node: FolderTree | null, query: string): FolderTree | null => {
  if (!node) {
    return null;
  }

  const lowerCaseQuery = query.toLowerCase();
  const isMatch = node.name.toLowerCase().includes(lowerCaseQuery);

  if (!node.children || node.children.length === 0) {
    return isMatch ? node : null;
  }

  const filteredChildren = node.children
    .map((child: FolderTree) => filterTree(child, query))
    .filter((child: FolderTree | null): child is FolderTree => child !== null);

  if (isMatch || filteredChildren.length > 0) {
    return { ...node, children: filteredChildren };
  }

  return null;
};

export const getAutoExpandedPaths = (node: FolderTree, paths: Set<string>) => {
  if (node.children && node.children.length > 0) {
    paths.add(node.path);
    node.children.forEach((child: FolderTree) => getAutoExpandedPaths(child, paths));
  }
};

export const filterAlbumTree = (node: AlbumItem | null, query: string): AlbumItem | null => {
  if (!node) return null;

  const lowerCaseQuery = query.toLowerCase();
  const isMatch = node.name.toLowerCase().includes(lowerCaseQuery);

  if (node.type === 'album') {
    return isMatch ? node : null;
  }

  if (node.type === 'group') {
    const filteredChildren = node.children
      .map((child: AlbumItem) => filterAlbumTree(child, query))
      .filter((child): child is AlbumItem => child !== null);

    if (isMatch || filteredChildren.length > 0) {
      return { ...node, children: filteredChildren };
    }
  }

  return null;
};

export const getAutoExpandedAlbumGroups = (node: AlbumItem, groups: Set<string>) => {
  if (node.type === 'group' && node.children.length > 0) {
    groups.add(node.id);
    node.children.forEach((child) => getAutoExpandedAlbumGroups(child, groups));
  }
};

export const sortFolderTree = (nodes: FolderTree[], sort: FolderTreeSort): FolderTree[] => {
  if (!nodes) return [];
  const sorted = [...nodes].sort((a, b) => {
    let comparison = 0;
    if (sort.key === 'name') comparison = a.name.localeCompare(b.name);
    else if (sort.key === 'modified') comparison = (a.modified || 0) - (b.modified || 0);
    else if (sort.key === 'created') comparison = (a.created || 0) - (b.created || 0);
    else if (sort.key === 'imageCount') comparison = (a.imageCount || 0) - (b.imageCount || 0);
    return sort.order === SortDirection.Ascending ? comparison : -comparison;
  });
  return sorted.map((node) => ({
    ...node,
    children: node.children && node.children.length > 0 ? sortFolderTree(node.children, sort) : node.children,
  }));
};
