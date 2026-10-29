import { useState, useEffect, useRef, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { v4 as uuidv4 } from 'uuid';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragStartEvent,
  pointerWithin,
} from '@dnd-kit/core';
import { Circle, ClipboardPaste, Folder as FolderIcon, Minus, Plus, RotateCcw, SquaresIntersect } from 'lucide-react';

import {
  Mask,
  MaskType,
  SubMask,
  MASK_PANEL_CREATION_TYPES,
  OTHERS_MASK_TYPES,
  MASK_ICON_MAP,
  SubMaskMode,
  ToolType,
  formatMaskTypeName,
  getSubMaskName,
  getMaskTypeName,
} from './Masks';
import {
  Adjustments,
  INITIAL_MASK_ADJUSTMENTS,
  INITIAL_MASK_CONTAINER,
  MaskContainer,
} from '../../../utils/adjustments';
import { useContextMenu } from '../../../context/ContextMenuContext';
import { OPTION_SEPARATOR } from '../../ui/AppProperties';
import { createSubMask } from '../../../utils/maskUtils';
import { usePresets } from '../../../hooks/usePresets';
import Text from '../../ui/Text';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useProcessStore } from '../../../store/useProcessStore';
import { useAiMasking } from '../../../hooks/useAiMasking';
import { useEditorActions } from '../../../hooks/useEditorActions';
import { useUIStore } from '../../../store/useUIStore';
import { type DragData, MasksListRoot, NewMaskDropZone, DraggableGridItem } from './MasksPanel/DragAndDrop';
import { ContainerRow } from './MasksPanel/ContainerRow';
import { SettingsPanel } from './MasksPanel/SettingsPanel';

export default function MasksPanel() {
  const { t } = useTranslation();
  const { setAdjustments } = useEditorActions();
  const { handleGenerateAiDepthMask, handleGenerateAiForegroundMask, handleGenerateAiSkyMask } = useAiMasking();
  const setCustomEscapeHandler = useUIStore((s) => s.setCustomEscapeHandler);
  const { appSettings } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
    })),
  );

  const { aiModelDownloadStatus } = useProcessStore(
    useShallow((state) => ({
      aiModelDownloadStatus: state.aiModelDownloadStatus,
    })),
  );

  const {
    activeMaskContainerId,
    activeMaskId,
    adjustments,
    brushSettings,
    copiedMask,
    histogram,
    isGeneratingAiMask,
    selectedImage,
    setEditor,
  } = useEditorStore(
    useShallow((state) => ({
      activeMaskContainerId: state.activeMaskContainerId,
      activeMaskId: state.activeMaskId,
      adjustments: state.adjustments,
      brushSettings: state.brushSettings,
      copiedMask: state.copiedMask,
      histogram: state.histogram,
      isGeneratingAiMask: state.isGeneratingAiMask,
      selectedImage: state.selectedImage,
      setEditor: state.setEditor,
    })),
  );

  const setBrushSettings = useCallback(
    (updater: any) => {
      setEditor((state) => ({ brushSettings: typeof updater === 'function' ? updater(state.brushSettings) : updater }));
    },
    [setEditor],
  );
  const selectBrushToolForNewMask = useCallback(() => {
    setEditor((state) => ({
      brushSettings: {
        ...(state.brushSettings ?? { size: 50, feather: 50, tool: ToolType.Brush }),
        tool: ToolType.Brush,
      },
    }));
  }, [setEditor]);

  const setCopiedMask = useCallback((mask: MaskContainer) => setEditor({ copiedMask: mask }), [setEditor]);
  const setIsMaskControlHovered = useCallback(
    (hovered: boolean) => setEditor({ isMaskControlHovered: hovered }),
    [setEditor],
  );
  const onDragStateChange = useCallback(
    (isDragging: boolean) => setEditor({ isSliderDragging: isDragging }),
    [setEditor],
  );
  const onSelectContainer = useCallback((id: string | null) => setEditor({ activeMaskContainerId: id }), [setEditor]);
  const onSelectMask = useCallback((id: string | null) => setEditor({ activeMaskId: id }), [setEditor]);

  const [expandedContainers, setExpandedContainers] = useState<Set<string>>(new Set());
  const [activeDragItem, setActiveDragItem] = useState<DragData | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [tempName, setTempName] = useState('');
  const [copiedSubMask, setCopiedSubMask] = useState<SubMask | null>(null);
  const [collapsibleState, setCollapsibleState] = useState<any>({
    basic: true,
    curves: false,
    color: false,
    details: false,
    effects: false,
  });
  const [copiedSectionAdjustments, setCopiedSectionAdjustments] = useState<any | null>(null);
  const [isSettingsSectionOpen, setSettingsSectionOpen] = useState(true);
  const [isSettingsPanelEverOpened, setIsSettingsPanelEverOpened] = useState(false);
  const hasPerformedInitialSelection = useRef(false);
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);
  const [analyzingSubMaskId, setAnalyzingSubMaskId] = useState<string | null>(null);

  const { showContextMenu } = useContextMenu();
  const { presets } = usePresets(adjustments);

  const activeContainer = adjustments.masks?.find((m) => m.id === activeMaskContainerId);
  const activeSubMaskData = activeContainer?.subMasks?.find((sm) => sm.id === activeMaskId);
  const isAiMask =
    activeSubMaskData && [Mask.AiSubject, Mask.AiForeground, Mask.AiSky, Mask.AiDepth].includes(activeSubMaskData.type);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    if (isGeneratingAiMask && isAiMask) {
      timer = setTimeout(() => {
        setAnalyzingSubMaskId(activeMaskId);
      }, 200);
    } else {
      setAnalyzingSubMaskId(null);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [isGeneratingAiMask, isAiMask, activeMaskId]);

  useEffect(() => {
    if (activeMaskContainerId) {
      const containerExists = adjustments.masks?.some((m) => m.id === activeMaskContainerId);
      if (!containerExists) {
        onSelectContainer(null);
        onSelectMask(null);
      }
    }
  }, [adjustments.masks, activeMaskContainerId, onSelectContainer, onSelectMask]);

  useEffect(() => {
    if (!hasPerformedInitialSelection.current && !activeMaskContainerId && adjustments.masks?.length > 0) {
      const lastMask = adjustments.masks[adjustments.masks.length - 1];
      if (lastMask) {
        onSelectContainer(lastMask.id);
        onSelectMask(null);
      }
    }

    if (activeMaskContainerId) {
      const shouldAutoExpand = !hasPerformedInitialSelection.current || activeMaskId;

      if (shouldAutoExpand) {
        setExpandedContainers((prev) => {
          if (prev.has(activeMaskContainerId)) {
            return prev;
          }
          return new Set(prev).add(activeMaskContainerId);
        });
      }

      hasPerformedInitialSelection.current = true;
    }

    if (activeMaskContainerId || adjustments.masks?.length > 0) {
      setIsSettingsPanelEverOpened(true);
    }
  }, [activeMaskContainerId, activeMaskId, adjustments.masks, onSelectContainer, onSelectMask]);

  useEffect(() => {
    const handler = () => {
      if (renamingId) {
        setRenamingId(null);
        setTempName('');
      } else if (activeMaskId) onSelectMask(null);
      else if (activeMaskContainerId) onSelectContainer(null);
    };
    if (activeMaskContainerId || renamingId) setCustomEscapeHandler(() => handler);
    else setCustomEscapeHandler(null);
    return () => setCustomEscapeHandler(null);
  }, [activeMaskContainerId, activeMaskId, renamingId, onSelectContainer, onSelectMask, setCustomEscapeHandler]);

  const handleDeselect = () => {
    onSelectContainer(null);
    onSelectMask(null);
  };

  const handleToggleExpand = (id: string) => {
    setExpandedContainers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleResetAllMasks = () => {
    handleDeselect();
    setAdjustments((prev: any) => ({ ...prev, masks: [] }));
  };

  const createMaskLogic = (type: Mask, mode: SubMaskMode = SubMaskMode.Additive) => {
    if (!selectedImage) return createSubMask(type, {} as any, mode);
    const subMask = createSubMask(type, selectedImage, mode);

    const steps = adjustments?.orientationSteps || 0;
    const isRotated = steps === 1 || steps === 3;
    const imgW = isRotated ? selectedImage.height || 1000 : selectedImage.width || 1000;
    const imgH = isRotated ? selectedImage.width || 1000 : selectedImage.height || 1000;

    if (type === Mask.Linear && subMask.parameters) {
      subMask.parameters.range = Math.min(imgW, imgH) * 0.1;
    }

    if (type === Mask.Linear || type === Mask.Radial || type === Mask.Color || type === Mask.Luminance) {
      if (!subMask.parameters) subMask.parameters = {};
      subMask.parameters.isInitialDraw = true;
      if (type === Mask.Linear || type === Mask.Radial) {
        subMask.parameters.startX = -10000;
        subMask.parameters.startY = -10000;
        subMask.parameters.endX = -10000;
        subMask.parameters.endY = -10000;
        subMask.parameters.centerX = -10000;
        subMask.parameters.centerY = -10000;
        subMask.parameters.radiusX = 0;
        subMask.parameters.radiusY = 0;
      } else {
        subMask.parameters.targetX = -10000;
        subMask.parameters.targetY = -10000;
        subMask.parameters.tolerance = 20;
        subMask.parameters.feather = 35;
      }
    }

    if (type === Mask.AiDepth) {
      if (!subMask.parameters) subMask.parameters = {};
      subMask.parameters.minDepth = 20;
      subMask.parameters.maxDepth = 80;
      subMask.parameters.minFade = 15;
      subMask.parameters.maxFade = 15;
      subMask.parameters.feather = 10;
    }
    return subMask;
  };

  const handleAddMaskContainer = (type: Mask) => {
    const subMask = createMaskLogic(type);
    const count = (adjustments.masks?.length || 0) + 1;
    const newContainer = {
      ...INITIAL_MASK_CONTAINER,
      id: uuidv4(),
      name: t('editor.masks.patches.maskName', { count }),
      subMasks: [subMask],
    };
    setAdjustments((prev: Adjustments) => ({ ...prev, masks: [...(prev.masks || []), newContainer] }));
    onSelectContainer(newContainer.id);
    onSelectMask(subMask.id);
    setExpandedContainers((prev) => new Set(prev).add(newContainer.id));
    if (type === Mask.Brush || type === Mask.Flow) selectBrushToolForNewMask();
    if (type === Mask.AiForeground) handleGenerateAiForegroundMask(subMask.id);
    else if (type === Mask.AiSky) handleGenerateAiSkyMask(subMask.id);
    else if (type === Mask.AiDepth) handleGenerateAiDepthMask(subMask.id, subMask.parameters);
  };

  const handleAddSubMask = (
    containerId: string,
    type: Mask,
    mode: SubMaskMode = SubMaskMode.Additive,
    insertIndex: number = -1,
  ) => {
    const subMask = createMaskLogic(type, mode);
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      masks: prev.masks?.map((c: MaskContainer) => {
        if (c.id === containerId) {
          const newSubMasks = [...c.subMasks];
          if (insertIndex >= 0) {
            newSubMasks.splice(insertIndex, 0, subMask);
          } else {
            newSubMasks.push(subMask);
          }
          return { ...c, subMasks: newSubMasks };
        }
        return c;
      }),
    }));
    onSelectContainer(containerId);
    onSelectMask(subMask.id);
    setExpandedContainers((prev) => new Set(prev).add(containerId));
    if (type === Mask.Brush || type === Mask.Flow) selectBrushToolForNewMask();
    if (type === Mask.AiForeground) handleGenerateAiForegroundMask(subMask.id);
    else if (type === Mask.AiSky) handleGenerateAiSkyMask(subMask.id);
    else if (type === Mask.AiDepth) handleGenerateAiDepthMask(subMask.id, subMask.parameters);
  };

  const handleGridClick = (type: Mask, forceNewMaskContainer: boolean = false) => {
    if (!forceNewMaskContainer && activeMaskContainerId) handleAddSubMask(activeMaskContainerId, type);
    else handleAddMaskContainer(type);
  };

  const handleGridRightClick = (event: React.MouseEvent, type: Mask | null) => {
    if (event.button !== 2) return;
    event.preventDefault();
    event.stopPropagation();
    if (!type) return;
    handleGridClick(type, true);
  };

  const handleAddOthersMask = (event: React.MouseEvent) => {
    event.stopPropagation();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const options = OTHERS_MASK_TYPES.map((maskType) => ({
      label: getMaskTypeName(maskType),
      icon: maskType.icon,
      onClick: () => handleGridClick(maskType.type),
      onRightClick: () => handleGridClick(maskType.type, true),
    }));
    showContextMenu(rect.left, rect.bottom + 5, options);
  };

  const handleAddMaskContextMenu = (event: React.MouseEvent, targetContainerId?: string | null) => {
    event.preventDefault();
    event.stopPropagation();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();

    const buildMenu = (types: MaskType[], mode: SubMaskMode = SubMaskMode.Additive) =>
      types.map((maskType: MaskType) => ({
        label: getMaskTypeName(maskType),
        icon: maskType.icon,
        disabled: maskType.disabled,
        onClick: () => {
          if (targetContainerId) {
            handleAddSubMask(targetContainerId, maskType.type, mode);
          } else {
            handleAddMaskContainer(maskType.type);
          }
        },
      }));

    const container = targetContainerId ? adjustments.masks?.find((m) => m.id === targetContainerId) : null;
    const hasComponents = container && container.subMasks.length > 0;

    const buildModeSubmenu = (label: string, icon: any, mode: SubMaskMode) => ({
      label,
      icon,
      submenu: MASK_PANEL_CREATION_TYPES.map((maskType) => {
        if (maskType.id === 'others') {
          return {
            label: getMaskTypeName(maskType),
            icon: maskType.icon,
            submenu: buildMenu(OTHERS_MASK_TYPES, mode),
          };
        }
        return {
          label: getMaskTypeName(maskType),
          icon: maskType.icon,
          disabled: maskType.disabled,
          onClick: () => handleAddSubMask(targetContainerId!, maskType.type, mode),
        };
      }),
    });

    const options: any[] = buildMenu(
      MASK_PANEL_CREATION_TYPES.filter((m) => m.id !== 'others'),
      SubMaskMode.Additive,
    );
    const others = MASK_PANEL_CREATION_TYPES.find((m) => m.id === 'others');
    if (others) {
      options.push({
        label: getMaskTypeName(others),
        icon: others.icon,
        submenu: buildMenu(OTHERS_MASK_TYPES, SubMaskMode.Additive),
      });
    }

    if (targetContainerId && hasComponents) {
      options.push(
        { type: OPTION_SEPARATOR },
        buildModeSubmenu(t('editor.masks.actions.subtractFromMask'), Minus, SubMaskMode.Subtractive),
        buildModeSubmenu(t('editor.masks.actions.intersectMaskWith'), SquaresIntersect, SubMaskMode.Intersect),
      );
    }

    showContextMenu(rect.left, rect.bottom + 5, options);
  };

  const updateContainer = (id: string, data: any) =>
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      masks: prev.masks.map((m) => (m.id === id ? { ...m, ...data } : m)),
    }));
  const updateSubMask = (id: string, data: any) =>
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      masks: prev.masks.map((m) => ({
        ...m,
        subMasks: m.subMasks.map((sm) => (sm.id === id ? { ...sm, ...data } : sm)),
      })),
    }));

  const handleDeleteContainer = (id: string) => {
    if (activeMaskContainerId === id) handleDeselect();
    setAdjustments((prev: Adjustments) => ({ ...prev, masks: prev.masks.filter((m) => m.id !== id) }));
  };

  const handleDeleteSubMask = (containerId: string, subMaskId: string) => {
    if (activeMaskId === subMaskId) onSelectMask(null);
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      masks: prev.masks.map((m) =>
        m.id === containerId ? { ...m, subMasks: m.subMasks.filter((sm) => sm.id !== subMaskId) } : m,
      ),
    }));
  };

  const cloneMaskContainerData = (
    container: MaskContainer,
    options: { invert?: boolean; rename?: boolean; resetAdjustments?: boolean } = {},
  ): MaskContainer => {
    const clonedContainer = JSON.parse(JSON.stringify(container));

    clonedContainer.id = uuidv4();
    clonedContainer.invert = options.invert ? !clonedContainer.invert : clonedContainer.invert;
    clonedContainer.name =
      options.rename === false ? clonedContainer.name : t('editor.masks.patches.copyName', { name: container.name });
    clonedContainer.subMasks = clonedContainer.subMasks.map((subMask: SubMask) => ({
      ...subMask,
      id: uuidv4(),
    }));

    if (options.resetAdjustments) {
      clonedContainer.adjustments = JSON.parse(JSON.stringify(INITIAL_MASK_ADJUSTMENTS));
    }

    return clonedContainer;
  };

  const cloneSubMaskData = (subMask: SubMask, options: { invert?: boolean; rename?: boolean } = {}): SubMask => {
    const clonedSubMask = JSON.parse(JSON.stringify(subMask));

    clonedSubMask.id = uuidv4();
    clonedSubMask.invert = options.invert ? !clonedSubMask.invert : clonedSubMask.invert;
    clonedSubMask.name =
      options.rename === false
        ? clonedSubMask.name
        : t('editor.masks.patches.copyName', { name: getSubMaskName(subMask) });

    return clonedSubMask;
  };

  const copyMaskToClipboard = (container: MaskContainer) => {
    setCopiedMask(JSON.parse(JSON.stringify(container)));
  };

  const copySubMaskToClipboard = (subMask: SubMask) => {
    setCopiedSubMask(JSON.parse(JSON.stringify(subMask)));
  };

  const insertMaskContainer = (container: MaskContainer, insertIndex?: number) => {
    setAdjustments((prev: Adjustments) => {
      const newMasks = [...(prev.masks || [])];
      const targetIndex = Math.max(0, Math.min(insertIndex ?? newMasks.length, newMasks.length));

      newMasks.splice(targetIndex, 0, container);

      return { ...prev, masks: newMasks };
    });

    onSelectContainer(container.id);
    onSelectMask(null);
    setExpandedContainers((prev) => new Set(prev).add(container.id));
  };

  const insertSubMaskIntoContainer = (containerId: string, subMask: SubMask, insertIndex?: number) => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      masks: prev.masks.map((container) => {
        if (container.id !== containerId) {
          return container;
        }

        const newSubMasks = [...container.subMasks];
        const targetIndex = Math.max(0, Math.min(insertIndex ?? newSubMasks.length, newSubMasks.length));

        newSubMasks.splice(targetIndex, 0, subMask);

        return { ...container, subMasks: newSubMasks };
      }),
    }));

    onSelectContainer(containerId);
    onSelectMask(subMask.id);
    setExpandedContainers((prev) => new Set(prev).add(containerId));
  };

  const handleDuplicateContainer = (container: MaskContainer) => {
    const containerIndex = adjustments.masks.findIndex((mask) => mask.id === container.id);
    const duplicatedContainer = cloneMaskContainerData(container, { rename: true, resetAdjustments: true });

    insertMaskContainer(duplicatedContainer, containerIndex >= 0 ? containerIndex + 1 : undefined);
  };

  const handleDuplicateAndInvertContainer = (container: MaskContainer) => {
    const containerIndex = adjustments.masks.findIndex((mask) => mask.id === container.id);
    const duplicatedContainer = cloneMaskContainerData(container, {
      invert: true,
      rename: false,
      resetAdjustments: true,
    });
    duplicatedContainer.name = t('editor.masks.patches.invertedName', { name: container.name });

    insertMaskContainer(duplicatedContainer, containerIndex >= 0 ? containerIndex + 1 : undefined);
  };

  const handlePasteMask = (insertAfterContainerId?: string) => {
    if (!copiedMask) {
      return;
    }

    const pastedContainer = cloneMaskContainerData(copiedMask, { rename: false });
    const containerIndex = insertAfterContainerId
      ? adjustments.masks.findIndex((mask) => mask.id === insertAfterContainerId)
      : -1;

    insertMaskContainer(pastedContainer, containerIndex >= 0 ? containerIndex + 1 : undefined);
  };

  const handleDuplicateSubMask = (containerId: string, subMask: SubMask, insertIndex?: number) => {
    const duplicatedSubMask = cloneSubMaskData(subMask, { rename: true });

    insertSubMaskIntoContainer(containerId, duplicatedSubMask, insertIndex);
  };

  const handleDuplicateAndInvertSubMask = (containerId: string, subMask: SubMask) => {
    const parentContainer = adjustments.masks.find((m) => m.id === containerId);
    if (!parentContainer) return;

    const duplicatedSubMask = cloneSubMaskData(subMask, { invert: true, rename: false });
    const newContainer = cloneMaskContainerData(parentContainer, { rename: false, resetAdjustments: true });

    newContainer.name = t('editor.masks.patches.invertedName', { name: getSubMaskName(subMask) });
    newContainer.subMasks = [duplicatedSubMask];
    newContainer.invert = false;

    const parentIndex = adjustments.masks.findIndex((m) => m.id === containerId);
    insertMaskContainer(newContainer, parentIndex >= 0 ? parentIndex + 1 : undefined);
  };

  const handlePasteSubMask = (containerId: string, insertIndex?: number) => {
    if (!copiedSubMask) {
      return;
    }

    const pastedSubMask = cloneSubMaskData(copiedSubMask, { rename: false });

    insertSubMaskIntoContainer(containerId, pastedSubMask, insertIndex);
  };

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));

  const handleDragStart = (event: DragStartEvent) => {
    setActiveDragItem(event.active.data.current as DragData);
    if (onDragStateChange) onDragStateChange(true);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    const dragData = active.data.current as DragData;
    const overData = over?.data.current as DragData;

    if (dragData.type === 'Creation' && dragData.maskType) {
      const creationFn = () => {
        if (overData?.type === 'Container') {
          handleAddSubMask(overData.item!.id, dragData.maskType!);
        } else if (overData?.type === 'SubMask') {
          const container = adjustments.masks.find((m) => m.id === overData.parentId);
          const overId = over?.id;
          if (container && overId) {
            const targetIndex = container.subMasks.findIndex((sm) => sm.id === overId);
            handleAddSubMask(overData.parentId!, dragData.maskType!, SubMaskMode.Additive, targetIndex);
          }
        } else {
          handleAddMaskContainer(dragData.maskType!);
        }
      };

      if (adjustments.masks && adjustments.masks.length > 0) {
        setPendingAction(() => creationFn);
      } else {
        creationFn();
      }

      setActiveDragItem(null);
      if (onDragStateChange) onDragStateChange(false);
      return;
    }

    setActiveDragItem(null);
    if (onDragStateChange) onDragStateChange(false);

    if (dragData.type === 'Container') {
      const overId = over?.id;
      if (!overId || active.id === overId) return;

      setAdjustments((prev: Adjustments) => {
        const oldIndex = prev.masks.findIndex((m) => m.id === dragData.item!.id);
        let newIndex = -1;

        if (overId === 'mask-list-root') {
          newIndex = prev.masks.length - 1;
        } else if (overData?.type === 'Container') {
          newIndex = prev.masks.findIndex((m) => m.id === overId);
        } else if (overData?.type === 'SubMask') {
          newIndex = prev.masks.findIndex((m) => m.id === overData.parentId);
        }

        if (oldIndex !== -1 && newIndex !== -1 && oldIndex !== newIndex) {
          const newMasks = [...prev.masks];
          const [movedItem] = newMasks.splice(oldIndex, 1);
          newMasks.splice(newIndex, 0, movedItem);
          return { ...prev, masks: newMasks };
        }
        return prev;
      });
      return;
    }

    if (dragData.type === 'SubMask') {
      const sourceContainerId = dragData.parentId;
      if (!sourceContainerId) return;

      if (over?.id === 'mask-list-root' || !over) {
        setAdjustments((prev: Adjustments) => {
          const newMasks = JSON.parse(JSON.stringify(prev.masks));
          const sourceContainer = newMasks.find((m: MaskContainer) => m.id === sourceContainerId);
          if (!sourceContainer) return prev;
          const subMaskIndex = sourceContainer.subMasks.findIndex((sm: SubMask) => sm.id === dragData.item!.id);
          if (subMaskIndex === -1) return prev;
          const [movedSubMask] = sourceContainer.subMasks.splice(subMaskIndex, 1);

          const newContainer = {
            ...INITIAL_MASK_CONTAINER,
            id: uuidv4(),
            name: `Mask ${newMasks.length + 1}`,
            subMasks: [movedSubMask],
          };
          newMasks.push(newContainer);
          setTimeout(() => {
            onSelectContainer(newContainer.id);
            onSelectMask(movedSubMask.id);
            setExpandedContainers((p) => new Set(p).add(newContainer.id));
          }, 0);
          return { ...prev, masks: newMasks };
        });
        return;
      }

      if (!over) return;

      let targetContainerId: string | null = null;
      if (overData?.type === 'Container') targetContainerId = overData.item!.id;
      else if (overData?.type === 'SubMask' && overData.parentId) targetContainerId = overData.parentId;

      if (targetContainerId) {
        setAdjustments((prev: Adjustments) => {
          const newMasks = prev.masks.map((m) => ({ ...m, subMasks: [...m.subMasks] }));
          const sourceContainer = newMasks.find((m) => m.id === sourceContainerId);
          const targetContainer = newMasks.find((m) => m.id === targetContainerId);
          if (!sourceContainer || !targetContainer) return prev;

          const sourceSubMaskIndex = sourceContainer.subMasks.findIndex((sm) => sm.id === dragData.item!.id);
          if (sourceSubMaskIndex === -1) return prev;

          const [movedSubMask] = sourceContainer.subMasks.splice(sourceSubMaskIndex, 1);

          if (sourceContainerId === targetContainerId) {
            if (overData?.type === 'SubMask') {
              const overSubMaskIndex = sourceContainer.subMasks.findIndex((sm) => sm.id === over.id);
              const insertIndex = overSubMaskIndex >= 0 ? overSubMaskIndex : sourceContainer.subMasks.length;
              sourceContainer.subMasks.splice(insertIndex, 0, movedSubMask);
            } else {
              sourceContainer.subMasks.push(movedSubMask);
            }
          } else {
            if (overData?.type === 'SubMask') {
              const overSubMaskIndex = targetContainer.subMasks.findIndex((sm) => sm.id === over.id);
              const insertIndex = overSubMaskIndex >= 0 ? overSubMaskIndex : targetContainer.subMasks.length;
              targetContainer.subMasks.splice(insertIndex, 0, movedSubMask);
            } else {
              targetContainer.subMasks.push(movedSubMask);
            }
            setExpandedContainers((p) => new Set(p).add(targetContainerId!));
          }
          return { ...prev, masks: newMasks };
        });
      }
    }
  };

  const handlePanelContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    const allTypes = [...MASK_PANEL_CREATION_TYPES.filter((m) => m.id !== 'others'), ...OTHERS_MASK_TYPES];
    const newMaskSubMenu = allTypes.map((m) => ({
      label: getMaskTypeName(m),
      icon: m.icon,
      onClick: () => handleAddMaskContainer(m.type),
    }));
    showContextMenu(e.clientX, e.clientY, [
      {
        label: t('editor.masks.actions.pasteMask'),
        icon: ClipboardPaste,
        disabled: !copiedMask,
        onClick: () => handlePasteMask(),
      },
      { label: t('editor.masks.addNewMask'), icon: Plus, submenu: newMaskSubMenu },
    ]);
  };

  return (
    <DndContext
      id="masks-panel-dnd"
      sensors={sensors}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      collisionDetection={pointerWithin}
    >
      <div className="ui-panel-root select-none overflow-hidden" onContextMenu={handlePanelContextMenu}>
        <div className="ui-panel-header">
          <Text variant={TextVariants.heading}>{t('editor.masks.maskingTitle')}</Text>
          <div className="flex items-center gap-0.5">
            <button
              aria-label={t('editor.masks.resetMaskingTooltip')}
              className="ui-icon-button"
              onClick={handleResetAllMasks}
              data-tooltip={t('editor.masks.resetMaskingTooltip')}
            >
              <RotateCcw size={16} strokeWidth={1.8} />
            </button>
          </div>
        </div>

        <div className="ui-panel-body flex flex-col overflow-x-hidden">
          {selectedImage ? (
            <>
              <AnimatePresence mode="wait">
                {!adjustments.masks || adjustments.masks.length === 0 ? (
                  <motion.div
                    key="empty-masks-grid"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="z-10 shrink-0"
                    onClick={handleDeselect}
                  >
                    <Text variant={TextVariants.heading} className="mb-2">
                      {t('editor.masks.createNewTitle')}
                    </Text>
                    <div className="grid grid-cols-3 gap-2" onClick={(e) => e.stopPropagation()}>
                      {MASK_PANEL_CREATION_TYPES.map((maskType: MaskType) => (
                        <DraggableGridItem
                          key={maskType.type || maskType.id}
                          maskType={maskType}
                          onClick={(e: any) =>
                            maskType.id === 'others' ? handleAddOthersMask(e) : handleGridClick(maskType.type)
                          }
                          onRightClick={(e: React.MouseEvent) => handleGridRightClick(e, maskType.type)}
                          isDraggable={maskType.id !== 'others'}
                          activeMaskContainerId={activeMaskContainerId}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <MasksListRoot onClick={handleDeselect}>
                    <Text variant={TextVariants.heading} className="mb-2">
                      {t('editor.masks.masksTitle')}
                    </Text>

                    <AnimatePresence
                      initial={false}
                      mode="popLayout"
                      onExitComplete={() => {
                        if (pendingAction) {
                          pendingAction();
                          setPendingAction(null);
                        }
                      }}
                    >
                      {adjustments.masks.map((container) => (
                        <ContainerRow
                          key={container.id}
                          container={container}
                          isSelected={activeMaskContainerId === container.id && activeMaskId === null}
                          hasActiveChild={activeMaskContainerId === container.id && activeMaskId !== null}
                          isExpanded={expandedContainers.has(container.id)}
                          onToggle={() => handleToggleExpand(container.id)}
                          onSelect={() => {
                            onSelectContainer(container.id);
                            onSelectMask(null);
                          }}
                          renamingId={renamingId}
                          setRenamingId={setRenamingId}
                          tempName={tempName}
                          setTempName={setTempName}
                          updateContainer={updateContainer}
                          handleDelete={handleDeleteContainer}
                          handleDuplicate={handleDuplicateContainer}
                          handleDuplicateAndInvert={handleDuplicateAndInvertContainer}
                          handlePasteMask={handlePasteMask}
                          copyMaskToClipboard={copyMaskToClipboard}
                          copiedMask={copiedMask}
                          presets={presets}
                          setAdjustments={setAdjustments}
                          activeDragItem={activeDragItem}
                          activeMaskId={activeMaskId}
                          onSelectContainer={onSelectContainer}
                          onSelectMask={onSelectMask}
                          updateSubMask={updateSubMask}
                          handleDeleteSubMask={handleDeleteSubMask}
                          handleDuplicateSubMask={handleDuplicateSubMask}
                          handleDuplicateAndInvertSubMask={handleDuplicateAndInvertSubMask}
                          handlePasteSubMask={handlePasteSubMask}
                          copySubMaskToClipboard={copySubMaskToClipboard}
                          copiedSubMask={copiedSubMask}
                          analyzingSubMaskId={analyzingSubMaskId}
                          setIsMaskControlHovered={setIsMaskControlHovered}
                          onAddComponent={(e: React.MouseEvent) => handleAddMaskContextMenu(e, container.id)}
                        />
                      ))}
                    </AnimatePresence>

                    <AnimatePresence>
                      {activeDragItem?.type === 'Creation' && adjustments.masks.length > 0 && (
                        <NewMaskDropZone isOver={false} />
                      )}
                    </AnimatePresence>

                    <Text
                      as="div"
                      weight={TextWeights.medium}
                      className="flex items-center gap-2 p-2 rounded-md transition-colors transition-opacity opacity-70 hover:opacity-100 hover:bg-card-active cursor-pointer hover:text-text-primary"
                      onClick={(e) => handleAddMaskContextMenu(e, null)}
                    >
                      <div className="p-0.5">
                        <Plus size={18} />
                      </div>
                      <span>{t('editor.masks.addNewMask')}</span>
                    </Text>
                  </MasksListRoot>
                )}
              </AnimatePresence>

              <div className="h-4 shrink-0 w-full" onClick={handleDeselect} />

              <AnimatePresence>
                {isSettingsPanelEverOpened && (
                  <motion.div
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                    className="flex-1 min-h-0"
                  >
                    <Text variant={TextVariants.heading} className="mb-2">
                      {t('editor.masks.maskAdjustmentsTitle')}
                    </Text>
                    <SettingsPanel
                      container={activeContainer}
                      activeSubMask={activeSubMaskData || null}
                      aiModelDownloadStatus={aiModelDownloadStatus}
                      brushSettings={brushSettings}
                      setBrushSettings={setBrushSettings}
                      updateContainer={updateContainer}
                      updateSubMask={updateSubMask}
                      histogram={histogram}
                      appSettings={appSettings}
                      isGeneratingAiMask={isGeneratingAiMask}
                      setIsMaskControlHovered={setIsMaskControlHovered}
                      collapsibleState={collapsibleState}
                      setCollapsibleState={setCollapsibleState}
                      copiedSectionAdjustments={copiedSectionAdjustments}
                      setCopiedSectionAdjustments={setCopiedSectionAdjustments}
                      onDragStateChange={onDragStateChange}
                      isSettingsSectionOpen={isSettingsSectionOpen}
                      setSettingsSectionOpen={setSettingsSectionOpen}
                      presets={presets}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <Text
                variant={TextVariants.heading}
                color={TextColors.secondary}
                weight={TextWeights.normal}
                className="text-center"
              >
                {t('editor.ai.noImageSelected')}
              </Text>
            </div>
          )}
        </div>
      </div>

      <DragOverlay dropAnimation={{ duration: 150, easing: 'cubic-bezier(0.18, 0.67, 0.6, 1.22)' }}>
        {activeDragItem ? (
          <div className="w-(--sidebar-width,280px) pointer-events-none">
            {activeDragItem.type === 'Container' && activeDragItem.item && (
              <Text
                as="div"
                color={TextColors.primary}
                weight={TextWeights.medium}
                className="flex items-center gap-2 p-2 rounded-md bg-surface shadow-2xl opacity-90 ring-1 ring-black/10"
              >
                <FolderIcon size={18} className={TEXT_COLOR_KEYS[TextColors.secondary]} />
                <span className="flex-1 truncate">{(activeDragItem.item as MaskContainer).name}</span>
              </Text>
            )}

            {activeDragItem.type === 'SubMask' && activeDragItem.item && (
              <Text
                as="div"
                color={TextColors.primary}
                weight={TextWeights.medium}
                className="flex items-center gap-2 p-2 rounded-md bg-surface shadow-2xl opacity-90 ring-1 ring-black/10 ml-3.75"
              >
                {(() => {
                  const sm = activeDragItem.item as SubMask;
                  const Icon = MASK_ICON_MAP[sm.type] || Circle;
                  return <Icon size={16} className={`shrink-0 ml-1 ${TEXT_COLOR_KEYS[TextColors.secondary]}`} />;
                })()}
                <span className="flex-1 truncate">{getSubMaskName(activeDragItem.item as SubMask)}</span>
              </Text>
            )}

            {activeDragItem.type === 'Creation' && (
              <Text
                as="div"
                variant={TextVariants.small}
                color={TextColors.primary}
                className="bg-surface rounded-lg gap-2 p-2 flex flex-col items-center justify-center aspect-square w-20 shadow-xl opacity-90"
              >
                {(() => {
                  const maskType =
                    MASK_PANEL_CREATION_TYPES.find((m) => m.type === activeDragItem.maskType) ||
                    OTHERS_MASK_TYPES.find((m) => m.type === activeDragItem.maskType);
                  const Icon = maskType?.icon || Circle;
                  return (
                    <>
                      <Icon size={24} />
                      <span className="text-center">
                        {activeDragItem.maskType ? formatMaskTypeName(activeDragItem.maskType) : 'Mask'}
                      </span>
                    </>
                  );
                })()}
              </Text>
            )}
          </div>
        ) : null}
      </DragOverlay>
    </DndContext>
  );
}
