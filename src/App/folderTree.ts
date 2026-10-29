export const insertChildrenIntoTree = (node: any, targetPath: string, newChildren: any[]): any => {
  if (!node) return null;

  if (node.path === targetPath) {
    const mergedChildren = newChildren.map((newChild: any) => {
      const existingChild = node.children?.find((c: any) => c.path === newChild.path);
      if (existingChild && existingChild.children && existingChild.children.length > 0) {
        return { ...newChild, children: existingChild.children };
      }
      return newChild;
    });
    return { ...node, children: mergedChildren };
  }

  if (node.children && node.children.length > 0) {
    return {
      ...node,
      children: node.children.map((child: any) => insertChildrenIntoTree(child, targetPath, newChildren)),
    };
  }

  return node;
};
