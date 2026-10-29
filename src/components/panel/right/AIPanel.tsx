import { useState, useEffect, useRef, useCallback } from 'react';
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
import { Circle, ClipboardPaste, Minus, Plus, RotateCcw, Wand2, SquaresIntersect } from 'lucide-react';

import { useContextMenu } from '../../../context/ContextMenuContext';
import {
  Mask,
  MaskType,
  SubMask,
  SubMaskMode,
  ToolType,
  MASK_ICON_MAP,
  AI_MANUAL_CLEANUP_TYPES,
  AI_SUB_MASK_COMPONENT_TYPES,
  formatMaskTypeName,
  getSubMaskName,
} from './Masks';
import { Adjustments, AiPatch } from '../../../utils/adjustments';
import { OPTION_SEPARATOR } from '../../ui/AppProperties';
import { createSubMask } from '../../../utils/maskUtils';
import Text from '../../ui/Text';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useProcessStore } from '../../../store/useProcessStore';
import { useUIStore } from '../../../store/useUIStore';
import { useEditorActions } from '../../../hooks/useEditorActions';
import { useAiMasking } from '../../../hooks/useAiMasking';
import { type DragData, AiListRoot, DraggableGridItem } from './AIPanel/DragAndDrop';
import { ContainerRow } from './AIPanel/ContainerRow';
import { ConnectionStatus } from './AIPanel/ConnectionStatus';
import { SUB_MASK_CONFIG } from './AIPanel/subMaskConfig';
import { SettingsPanel } from './AIPanel/SettingsPanel';

export default function AIPanel() {
  const { t } = useTranslation();
  const activePatchContainerId = useEditorStore((s) => s.activeAiPatchContainerId);
  const activeSubMaskId = useEditorStore((s) => s.activeAiSubMaskId);
  const adjustments = useEditorStore((s) => s.adjustments);
  const brushSettings = useEditorStore((s) => s.brushSettings);
  const isGeneratingAi = useEditorStore((s) => s.isGeneratingAi);
  const isGeneratingAiMask = useEditorStore((s) => s.isGeneratingAiMask);
  const selectedImage = useEditorStore((s) => s.selectedImage);
  const setEditor = useEditorStore((s) => s.setEditor);

  const aiModelDownloadStatus = useProcessStore((s) => s.aiModelDownloadStatus);
  const setCustomEscapeHandler = useUIStore((s) => s.setCustomEscapeHandler);

  const { setAdjustments } = useEditorActions();
  const { handleGenerativeReplace, handleDeleteAiPatch, handleGenerateAiForegroundMask } = useAiMasking();

  const setBrushSettings = useCallback(
    (updater: any) =>
      setEditor((state) => ({ brushSettings: typeof updater === 'function' ? updater(state.brushSettings) : updater })),
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

  const onSelectPatchContainer = useCallback(
    (id: string | null) => setEditor({ activeAiPatchContainerId: id }),
    [setEditor],
  );
  const onSelectSubMask = useCallback((id: string | null) => setEditor({ activeAiSubMaskId: id }), [setEditor]);
  const onDragStateChange = useCallback(
    (isDragging: boolean) => setEditor({ isSliderDragging: isDragging }),
    [setEditor],
  );

  const [expandedContainers, setExpandedContainers] = useState<Set<string>>(new Set());
  const [activeDragItem, setActiveDragItem] = useState<DragData | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [tempName, setTempName] = useState('');
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);
  const [isSettingsPanelEverOpened, setIsSettingsPanelEverOpened] = useState(false);
  const hasPerformedInitialSelection = useRef(false);
  const [analyzingSubMaskId, setAnalyzingSubMaskId] = useState<string | null>(null);
  const [copiedPatch, setCopiedPatch] = useState<AiPatch | null>(null);
  const [copiedSubMask, setCopiedSubMask] = useState<SubMask | null>(null);

  const [collapsibleState, setCollapsibleState] = useState({
    generative: true,
    properties: true,
  });

  const { showContextMenu } = useContextMenu();
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));

  const activeContainer = (adjustments.aiPatches || []).find((p) => p.id === activePatchContainerId);
  const activeSubMaskData = activeContainer?.subMasks.find((sm) => sm.id === activeSubMaskId);
  const isAiMask =
    activeSubMaskData && [Mask.AiSubject, Mask.AiForeground, Mask.AiSky].includes(activeSubMaskData.type);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    if (isGeneratingAiMask && isAiMask) {
      timer = setTimeout(() => {
        setAnalyzingSubMaskId(activeSubMaskId);
      }, 200);
    } else {
      setAnalyzingSubMaskId(null);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [isGeneratingAiMask, isAiMask, activeSubMaskId]);

  useEffect(() => {
    if (activePatchContainerId) {
      const patchExists = adjustments.aiPatches?.some((p) => p.id === activePatchContainerId);
      if (!patchExists) {
        onSelectPatchContainer(null);
        onSelectSubMask(null);
      } else if (!activeSubMaskId) {
        const container = adjustments.aiPatches?.find((p) => p.id === activePatchContainerId);
        if (
          container &&
          container.subMasks.length === 1 &&
          [Mask.Clone, Mask.Heal].includes(container.subMasks[0].type)
        ) {
          onSelectSubMask(container.subMasks[0].id);
        }
      }
    }
  }, [adjustments.aiPatches, activePatchContainerId, activeSubMaskId, onSelectPatchContainer, onSelectSubMask]);

  useEffect(() => {
    const hasPatches = (adjustments.aiPatches || []).length > 0;

    if (hasPatches) {
      setIsSettingsPanelEverOpened(true);
    }

    if (activePatchContainerId) {
      const shouldAutoExpand = !hasPerformedInitialSelection.current || activeSubMaskId;
      if (shouldAutoExpand) {
        setExpandedContainers((prev) => {
          if (prev.has(activePatchContainerId)) return prev;
          return new Set(prev).add(activePatchContainerId);
        });
      }
      hasPerformedInitialSelection.current = true;
      setIsSettingsPanelEverOpened(true);
    }
  }, [activePatchContainerId, activeSubMaskId, adjustments.aiPatches, onSelectPatchContainer, onSelectSubMask]);

  useEffect(() => {
    const handler = () => {
      if (renamingId) {
        setRenamingId(null);
        setTempName('');
      } else if (activeSubMaskId) onSelectSubMask(null);
      else if (activePatchContainerId) onSelectPatchContainer(null);
    };
    if (activePatchContainerId || renamingId) setCustomEscapeHandler(() => handler);
    else setCustomEscapeHandler(null);
    return () => setCustomEscapeHandler(null);
  }, [
    activePatchContainerId,
    activeSubMaskId,
    renamingId,
    onSelectPatchContainer,
    onSelectSubMask,
    setCustomEscapeHandler,
  ]);

  const handleDeselect = () => {
    onSelectPatchContainer(null);
    onSelectSubMask(null);
  };

  const handleToggleExpand = (id: string) => {
    setExpandedContainers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleResetAllAiEdits = () => {
    if (isGeneratingAi) return;
    handleDeselect();
    setAdjustments((prev: Adjustments) => ({ ...prev, aiPatches: [] }));
  };

  const createMaskLogic = (type: Mask, mode: SubMaskMode = SubMaskMode.Additive) => {
    if (!selectedImage) return createSubMask(type, {} as any, mode);
    const subMask = createSubMask(type, selectedImage, mode);

    const steps = adjustments?.orientationSteps || 0;
    const isRotated = steps === 1 || steps === 3;
    const imgW = isRotated ? selectedImage.height || 1000 : selectedImage.width || 1000;
    const imgH = isRotated ? selectedImage.width || 1000 : selectedImage.height || 1000;

    const config = SUB_MASK_CONFIG[type];
    if (config && config.parameters) {
      config.parameters.forEach((param: any) => {
        if (param.defaultValue !== undefined) {
          subMask.parameters[param.key] = param.defaultValue / (param.multiplier || 1);
        }
      });
    }

    if (type === Mask.Linear && subMask.parameters) {
      subMask.parameters.range = Math.min(imgW, imgH) * 0.1;
    }

    if (type === Mask.Linear || type === Mask.Radial) {
      if (!subMask.parameters) subMask.parameters = {};
      subMask.parameters.isInitialDraw = true;
      subMask.parameters.startX = -10000;
      subMask.parameters.startY = -10000;
      subMask.parameters.endX = -10000;
      subMask.parameters.endY = -10000;
      subMask.parameters.centerX = -10000;
      subMask.parameters.centerY = -10000;
      subMask.parameters.radiusX = 0;
      subMask.parameters.radiusY = 0;
    }
    return subMask;
  };

  const handleAddAiPatchContainer = (type: Mask) => {
    const subMask = createMaskLogic(type);

    let name: string;
    if (type === Mask.QuickEraser) {
      const count =
        (adjustments.aiPatches || []).filter((p: AiPatch) =>
          p.subMasks.some((sm: SubMask) => sm.type === Mask.QuickEraser),
        ).length + 1;
      name = t('editor.ai.patches.quickErase', { count });
    } else if (type === Mask.Clone) {
      const count =
        (adjustments.aiPatches || []).filter((p: AiPatch) => p.subMasks.some((sm: SubMask) => sm.type === Mask.Clone))
          .length + 1;
      name = t('editor.ai.patches.clone', { count });
    } else if (type === Mask.Heal) {
      const count =
        (adjustments.aiPatches || []).filter((p: AiPatch) => p.subMasks.some((sm: SubMask) => sm.type === Mask.Heal))
          .length + 1;
      name = t('editor.ai.patches.heal', { count });
    } else {
      const count = (adjustments.aiPatches || []).length + 1;
      name = t('editor.ai.patches.aiEdit', { count });
    }

    const newContainer: AiPatch = {
      id: uuidv4(),
      invert: false,
      isLoading: false,
      name: name,
      patchData: null,
      prompt: '',
      subMasks: [subMask],
      visible: true,
    };

    setAdjustments((prev: Adjustments) => ({ ...prev, aiPatches: [...(prev.aiPatches || []), newContainer] }));
    onSelectPatchContainer(newContainer.id);

    const isStandalone = [Mask.Clone, Mask.Heal].includes(type);

    onSelectSubMask(subMask.id);
    if (!isStandalone) {
      setExpandedContainers((prev) => new Set(prev).add(newContainer.id));
    }

    if (type === Mask.Brush || type === Mask.Clone || type === Mask.Heal) {
      selectBrushToolForNewMask();
    }

    if (type === Mask.AiForeground) handleGenerateAiForegroundMask(subMask.id);
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
      aiPatches: prev.aiPatches?.map((c: AiPatch) => {
        if (c.id === containerId) {
          const newSubMasks = [...c.subMasks];
          if (insertIndex >= 0) newSubMasks.splice(insertIndex, 0, subMask);
          else newSubMasks.push(subMask);
          return { ...c, subMasks: newSubMasks };
        }
        return c;
      }),
    }));
    onSelectPatchContainer(containerId);
    onSelectSubMask(subMask.id);
    setExpandedContainers((prev) => new Set(prev).add(containerId));

    if (type === Mask.Brush || type === Mask.Clone || type === Mask.Heal) {
      selectBrushToolForNewMask();
    }
    if (type === Mask.AiForeground) handleGenerateAiForegroundMask(subMask.id);
  };

  const handleAddAiContextMenu = (event: React.MouseEvent, targetContainerId?: string | null) => {
    event.preventDefault();
    event.stopPropagation();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();

    const container = targetContainerId ? adjustments.aiPatches.find((m) => m.id === targetContainerId) : null;
    const isStandalone =
      container && container.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(container.subMasks[0].type);

    if (isStandalone && targetContainerId) {
      return;
    }

    const buildMenu = (types: MaskType[], mode: SubMaskMode = SubMaskMode.Additive) =>
      types
        .filter((mt) => !mt.disabled && (!targetContainerId || ![Mask.Clone, Mask.Heal].includes(mt.type)))
        .map((maskType: MaskType) => ({
          label: formatMaskTypeName(maskType.type),
          icon: maskType.icon,
          onClick: () => {
            if (targetContainerId) {
              handleAddSubMask(targetContainerId, maskType.type, mode);
            } else {
              handleAddAiPatchContainer(maskType.type);
            }
          },
        }));

    const hasComponents = container && container.subMasks.length > 0;

    let options: any[];

    if (!targetContainerId) {
      options = buildMenu(AI_MANUAL_CLEANUP_TYPES, SubMaskMode.Additive);
    } else {
      options = buildMenu(AI_SUB_MASK_COMPONENT_TYPES, SubMaskMode.Additive);

      if (hasComponents) {
        options.push(
          { type: OPTION_SEPARATOR },
          {
            label: t('editor.ai.actions.subtractFromEdit'),
            icon: Minus,
            submenu: buildMenu(AI_SUB_MASK_COMPONENT_TYPES, SubMaskMode.Subtractive),
          },
          {
            label: t('editor.ai.actions.intersectEditWith'),
            icon: SquaresIntersect,
            submenu: buildMenu(AI_SUB_MASK_COMPONENT_TYPES, SubMaskMode.Intersect),
          },
        );
      }
    }

    showContextMenu(rect.left, rect.bottom + 5, options);
  };

  const updatePatch = (id: string, data: any) =>
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aiPatches: prev.aiPatches.map((p) => (p.id === id ? { ...p, ...data } : p)),
    }));

  const updateSubMask = (id: string, data: any) =>
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aiPatches: prev.aiPatches.map((p) => ({
        ...p,
        subMasks: p.subMasks.map((sm) => (sm.id === id ? { ...sm, ...data } : sm)),
      })),
    }));

  const handleDeleteContainer = (id: string) => {
    if (activePatchContainerId === id) handleDeselect();
    handleDeleteAiPatch(id);
  };

  const handleDeleteSubMask = (containerId: string, subMaskId: string) => {
    if (activeSubMaskId === subMaskId) onSelectSubMask(null);
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aiPatches: prev.aiPatches.map((p) =>
        p.id === containerId ? { ...p, subMasks: p.subMasks.filter((sm) => sm.id !== subMaskId) } : p,
      ),
    }));
  };

  const clonePatchData = (container: AiPatch, options: { invert?: boolean; rename?: boolean } = {}): AiPatch => {
    const clonedContainer = JSON.parse(JSON.stringify(container));

    clonedContainer.id = uuidv4();
    clonedContainer.invert = options.invert ? !clonedContainer.invert : clonedContainer.invert;
    clonedContainer.isLoading = false;
    clonedContainer.name = options.rename === false ? clonedContainer.name : `${container.name} Copy`;
    clonedContainer.patchData = null;
    clonedContainer.subMasks = clonedContainer.subMasks.map((subMask: SubMask) => ({
      ...subMask,
      id: uuidv4(),
    }));

    return clonedContainer;
  };

  const cloneSubMaskData = (subMask: SubMask, options: { invert?: boolean; rename?: boolean } = {}): SubMask => {
    const clonedSubMask = JSON.parse(JSON.stringify(subMask));

    clonedSubMask.id = uuidv4();
    clonedSubMask.invert = options.invert ? !clonedSubMask.invert : clonedSubMask.invert;
    clonedSubMask.name = options.rename === false ? clonedSubMask.name : `${getSubMaskName(subMask)} Copy`;

    return clonedSubMask;
  };

  const copyPatchToClipboard = (container: AiPatch) => {
    setCopiedPatch(JSON.parse(JSON.stringify(container)));
  };

  const copySubMaskToClipboard = (subMask: SubMask) => {
    setCopiedSubMask(JSON.parse(JSON.stringify(subMask)));
  };

  const insertPatchContainer = (container: AiPatch, insertIndex?: number) => {
    setAdjustments((prev: Adjustments) => {
      const newPatches = [...(prev.aiPatches || [])];
      const targetIndex = Math.max(0, Math.min(insertIndex ?? newPatches.length, newPatches.length));

      newPatches.splice(targetIndex, 0, container);
      return { ...prev, aiPatches: newPatches };
    });

    onSelectPatchContainer(container.id);
    const isStandalone =
      container.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(container.subMasks[0].type);

    if (isStandalone) {
      onSelectSubMask(container.subMasks[0].id);
    } else {
      onSelectSubMask(null);
      setExpandedContainers((prev) => new Set(prev).add(container.id));
    }
  };

  const insertSubMaskIntoContainer = (containerId: string, subMask: SubMask, insertIndex?: number) => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aiPatches: (prev.aiPatches || []).map((container) => {
        if (container.id !== containerId) {
          return container;
        }

        const newSubMasks = [...container.subMasks];
        const targetIndex = Math.max(0, Math.min(insertIndex ?? newSubMasks.length, newSubMasks.length));

        newSubMasks.splice(targetIndex, 0, subMask);
        return { ...container, subMasks: newSubMasks };
      }),
    }));

    onSelectPatchContainer(containerId);
    onSelectSubMask(subMask.id);
    setExpandedContainers((prev) => new Set(prev).add(containerId));
  };

  const handleDuplicatePatchContainer = (container: AiPatch) => {
    const patchIndex = (adjustments.aiPatches || []).findIndex((patch) => patch.id === container.id);
    const duplicatedContainer = clonePatchData(container, { rename: true });

    insertPatchContainer(duplicatedContainer, patchIndex >= 0 ? patchIndex + 1 : undefined);
  };

  const handleDuplicateAndInvertPatchContainer = (container: AiPatch) => {
    const patchIndex = (adjustments.aiPatches || []).findIndex((patch) => patch.id === container.id);
    const duplicatedContainer = clonePatchData(container, { invert: true, rename: false });
    duplicatedContainer.name = t('editor.ai.patches.invertedName', { name: container.name });

    insertPatchContainer(duplicatedContainer, patchIndex >= 0 ? patchIndex + 1 : undefined);
  };

  const handlePastePatch = (insertAfterContainerId?: string) => {
    if (!copiedPatch) {
      return;
    }

    const pastedContainer = clonePatchData(copiedPatch, { rename: false });
    const patchIndex = insertAfterContainerId
      ? (adjustments.aiPatches || []).findIndex((patch) => patch.id === insertAfterContainerId)
      : -1;

    insertPatchContainer(pastedContainer, patchIndex >= 0 ? patchIndex + 1 : undefined);
  };

  const handleDuplicateSubMask = (containerId: string, subMask: SubMask, insertIndex?: number) => {
    const duplicatedSubMask = cloneSubMaskData(subMask, { rename: true });
    insertSubMaskIntoContainer(containerId, duplicatedSubMask, insertIndex);
  };

  const handleDuplicateAndInvertSubMask = (containerId: string, subMask: SubMask) => {
    const parentContainer = (adjustments.aiPatches || []).find((p) => p.id === containerId);
    if (!parentContainer) return;

    const duplicatedSubMask = cloneSubMaskData(subMask, { invert: true, rename: false });
    const newContainer = clonePatchData(parentContainer, { rename: false });

    newContainer.name = t('editor.ai.patches.invertedName', { name: getSubMaskName(subMask) });
    newContainer.subMasks = [duplicatedSubMask];
    newContainer.invert = false;

    const parentIndex = (adjustments.aiPatches || []).findIndex((p) => p.id === containerId);
    insertPatchContainer(newContainer, parentIndex >= 0 ? parentIndex + 1 : undefined);
  };

  const handlePasteSubMask = (containerId: string, insertIndex?: number) => {
    if (!copiedSubMask) {
      return;
    }

    const pastedSubMask = cloneSubMaskData(copiedSubMask, { rename: false });
    insertSubMaskIntoContainer(containerId, pastedSubMask, insertIndex);
  };

  const handlePanelContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!selectedImage) {
      return;
    }

    const manualSubMenu = AI_MANUAL_CLEANUP_TYPES.filter((maskType) => !maskType.disabled).map((maskType) => ({
      label: formatMaskTypeName(maskType.type),
      icon: maskType.icon,
      onClick: () => handleAddAiPatchContainer(maskType.type),
    }));

    showContextMenu(e.clientX, e.clientY, [
      {
        label: t('editor.ai.actions.pasteEdit'),
        icon: ClipboardPaste,
        disabled: !copiedPatch,
        onClick: () => handlePastePatch(),
      },
      {
        label: t('editor.ai.addNewEdit'),
        icon: Plus,
        submenu: manualSubMenu,
      },
    ]);
  };

  const handleDragStart = (event: DragStartEvent) => {
    setActiveDragItem(event.active.data.current as DragData);
    if (onDragStateChange) onDragStateChange(true);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    const dragData = active.data.current as DragData;
    const overData = over?.data.current as DragData;

    setActiveDragItem(null);
    if (onDragStateChange) onDragStateChange(false);

    if (dragData.type === 'Creation' && dragData.maskType) {
      const creationFn = () => {
        const isCreationStandalone = [Mask.Clone, Mask.Heal].includes(dragData.maskType!);

        if (isCreationStandalone) {
          handleAddAiPatchContainer(dragData.maskType!);
        } else if (overData?.type === 'Container') {
          const overContainer = adjustments.aiPatches.find((p) => p.id === overData.item!.id);
          const isOverStandalone =
            overContainer?.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(overContainer.subMasks[0].type);

          if (isOverStandalone) {
            handleAddAiPatchContainer(dragData.maskType!);
          } else {
            handleAddSubMask(overData.item!.id, dragData.maskType!);
          }
        } else if (overData?.type === 'SubMask') {
          const container = adjustments.aiPatches.find((p) => p.id === overData.parentId);
          const isTargetStandalone =
            container?.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(container.subMasks[0].type);

          if (container && !isTargetStandalone) {
            const targetIndex = container.subMasks.findIndex((sm) => sm.id === over!.id);
            handleAddSubMask(overData.parentId!, dragData.maskType!, SubMaskMode.Additive, targetIndex);
          } else {
            handleAddAiPatchContainer(dragData.maskType!);
          }
        } else {
          handleAddAiPatchContainer(dragData.maskType!);
        }
      };

      if ((adjustments.aiPatches || []).length > 0) setPendingAction(() => creationFn);
      else creationFn();
      return;
    }

    if (dragData.type === 'Container') {
      const overId = over?.id;
      if (!overId || active.id === overId) return;

      setAdjustments((prev: Adjustments) => {
        const oldIndex = prev.aiPatches.findIndex((p) => p.id === dragData.item!.id);
        let newIndex = -1;

        if (overId === 'ai-list-root') newIndex = prev.aiPatches.length - 1;
        else if (overData?.type === 'Container') newIndex = prev.aiPatches.findIndex((p) => p.id === overId);
        else if (overData?.type === 'SubMask') newIndex = prev.aiPatches.findIndex((p) => p.id === overData.parentId);

        if (oldIndex !== -1 && newIndex !== -1 && oldIndex !== newIndex) {
          const newPatches = [...prev.aiPatches];
          const [movedItem] = newPatches.splice(oldIndex, 1);
          newPatches.splice(newIndex, 0, movedItem);
          return { ...prev, aiPatches: newPatches };
        }
        return prev;
      });
      return;
    }

    if (dragData.type === 'SubMask') {
      const sourceContainerId = dragData.parentId;
      if (!sourceContainerId) return;

      let targetContainerId: string | null = null;
      if (overData?.type === 'Container') targetContainerId = overData.item!.id;
      else if (overData?.type === 'SubMask') targetContainerId = overData.parentId || null;

      if (targetContainerId) {
        const targetContainer = adjustments.aiPatches.find((p) => p.id === targetContainerId);
        const isTargetStandalone =
          targetContainer?.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(targetContainer.subMasks[0].type);

        const isSourceStandalone = [Mask.Clone, Mask.Heal].includes((dragData.item as SubMask).type);

        if ((isTargetStandalone || isSourceStandalone) && sourceContainerId !== targetContainerId) {
          return;
        }
      }

      if (over?.id === 'ai-list-root' || !over || !targetContainerId) {
        setAdjustments((prev: Adjustments) => {
          const newPatches = JSON.parse(JSON.stringify(prev.aiPatches));
          const sourceContainer = newPatches.find((p: AiPatch) => p.id === sourceContainerId);
          if (!sourceContainer) return prev;
          const subMaskIndex = sourceContainer.subMasks.findIndex((sm: SubMask) => sm.id === dragData.item!.id);
          if (subMaskIndex === -1) return prev;

          const [movedSubMask] = sourceContainer.subMasks.splice(subMaskIndex, 1);

          const newContainer: AiPatch = {
            id: uuidv4(),
            invert: false,
            isLoading: false,
            name: t('editor.ai.patches.aiEdit', { count: newPatches.length + 1 }),
            patchData: null,
            prompt: '',
            subMasks: [movedSubMask],
            visible: true,
          };
          newPatches.push(newContainer);

          setTimeout(() => {
            onSelectPatchContainer(newContainer.id);
            onSelectSubMask(movedSubMask.id);
            setExpandedContainers((p) => new Set(p).add(newContainer.id));
          }, 0);
          return { ...prev, aiPatches: newPatches };
        });
        return;
      }

      if (targetContainerId) {
        setAdjustments((prev: Adjustments) => {
          const newPatches = prev.aiPatches.map((p) => ({ ...p, subMasks: [...p.subMasks] }));
          const sourceContainer = newPatches.find((p) => p.id === sourceContainerId);
          const targetContainer = newPatches.find((p) => p.id === targetContainerId);
          if (!sourceContainer || !targetContainer) return prev;

          const sourceIndex = sourceContainer.subMasks.findIndex((sm) => sm.id === dragData.item!.id);
          if (sourceIndex === -1) return prev;
          const [movedSubMask] = sourceContainer.subMasks.splice(sourceIndex, 1);

          if (sourceContainerId === targetContainerId) {
            if (overData?.type === 'SubMask') {
              const overIndex = sourceContainer.subMasks.findIndex((sm) => sm.id === over.id);
              const insertIndex = overIndex >= 0 ? overIndex : sourceContainer.subMasks.length;
              sourceContainer.subMasks.splice(insertIndex, 0, movedSubMask);
            } else {
              sourceContainer.subMasks.push(movedSubMask);
            }
          } else {
            if (overData?.type === 'SubMask') {
              const overIndex = targetContainer.subMasks.findIndex((sm) => sm.id === over.id);
              const insertIndex = overIndex >= 0 ? overIndex : targetContainer.subMasks.length;
              targetContainer.subMasks.splice(insertIndex, 0, movedSubMask);
            } else {
              targetContainer.subMasks.push(movedSubMask);
            }
            setExpandedContainers((p) => new Set(p).add(targetContainerId!));
          }
          return { ...prev, aiPatches: newPatches };
        });
      }
    }
  };

  return (
    <DndContext
      id="ai-panel-dnd"
      sensors={sensors}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      collisionDetection={pointerWithin}
    >
      <div className="ui-panel-root select-none overflow-hidden" onContextMenu={handlePanelContextMenu}>
        <div className="ui-panel-header">
          <Text variant={TextVariants.heading}>{t('editor.ai.inpaintingTitle')}</Text>
          <button
            className="ui-icon-button"
            onClick={handleResetAllAiEdits}
            data-tooltip={t('editor.ai.resetInpaintingTooltip')}
          >
            <RotateCcw size={18} />
          </button>
        </div>

        <div className="ui-panel-body flex flex-col overflow-x-hidden">
          {!selectedImage ? (
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
          ) : (
            <>
              <AnimatePresence mode="wait">
                {(adjustments.aiPatches || []).length === 0 ? (
                  <motion.div
                    key="ai-grid"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="z-10 shrink-0"
                    onClick={handleDeselect}
                  >
                    <ConnectionStatus />

                    <Text variant={TextVariants.heading} className="mb-2 mt-6">
                      {t('editor.ai.manualCleanupTitle')}
                    </Text>
                    <div className="grid grid-cols-3 gap-2 mb-6" onClick={(e) => e.stopPropagation()}>
                      {AI_MANUAL_CLEANUP_TYPES.map((maskType: MaskType) => (
                        <DraggableGridItem
                          key={maskType.type}
                          maskType={maskType}
                          isGenerating={isGeneratingAi}
                          onClick={() => handleAddAiPatchContainer(maskType.type)}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <AiListRoot
                    onClick={handleDeselect}
                    activeDragItem={activeDragItem}
                    hasPatches={(adjustments.aiPatches || []).length > 0}
                  >
                    <Text variant={TextVariants.heading} className="mb-2">
                      {t('editor.ai.editsTitle')}
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
                      {(adjustments.aiPatches || []).map((container) => (
                        <ContainerRow
                          key={container.id}
                          container={container}
                          isSelected={activePatchContainerId === container.id && activeSubMaskId === null}
                          hasActiveChild={activePatchContainerId === container.id && activeSubMaskId !== null}
                          isExpanded={expandedContainers.has(container.id)}
                          onToggle={() => handleToggleExpand(container.id)}
                          onSelect={() => {
                            onSelectPatchContainer(container.id);
                            onSelectSubMask(null);
                          }}
                          renamingId={renamingId}
                          setRenamingId={setRenamingId}
                          tempName={tempName}
                          setTempName={setTempName}
                          updateContainer={updatePatch}
                          handleDelete={handleDeleteContainer}
                          handleDuplicate={handleDuplicatePatchContainer}
                          handleDuplicateAndInvert={handleDuplicateAndInvertPatchContainer}
                          handlePastePatch={handlePastePatch}
                          copyPatchToClipboard={copyPatchToClipboard}
                          copiedPatch={copiedPatch}
                          setAdjustments={setAdjustments}
                          activeDragItem={activeDragItem}
                          activeSubMaskId={activeSubMaskId}
                          activePatchContainerId={activePatchContainerId}
                          onSelectContainer={onSelectPatchContainer}
                          onSelectSubMask={onSelectSubMask}
                          updateSubMask={updateSubMask}
                          handleDeleteSubMask={handleDeleteSubMask}
                          handleDuplicateSubMask={handleDuplicateSubMask}
                          handleDuplicateAndInvertSubMask={handleDuplicateAndInvertSubMask}
                          handlePasteSubMask={handlePasteSubMask}
                          copySubMaskToClipboard={copySubMaskToClipboard}
                          copiedSubMask={copiedSubMask}
                          analyzingSubMaskId={analyzingSubMaskId}
                          onAddComponent={(e: React.MouseEvent) => handleAddAiContextMenu(e, container.id)}
                        />
                      ))}
                    </AnimatePresence>

                    <Text
                      as="div"
                      weight={TextWeights.medium}
                      className="flex items-center gap-2 p-2 rounded-md transition-colors transition-opacity opacity-70 hover:opacity-100 hover:bg-card-active cursor-pointer hover:text-text-primary"
                      onClick={(e) => handleAddAiContextMenu(e, null)}
                    >
                      <div className="p-0.5">
                        <Plus size={18} />
                      </div>
                      <span>{t('editor.ai.addNewEdit')}</span>
                    </Text>
                  </AiListRoot>
                )}
              </AnimatePresence>

              <div className="h-4 shrink-0 w-full" onClick={handleDeselect} />

              <AnimatePresence>
                {isSettingsPanelEverOpened && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                    className="flex-1 min-h-0"
                  >
                    <Text variant={TextVariants.heading} className="mb-2">
                      {t('editor.ai.editSettingsTitle')}
                    </Text>
                    <SettingsPanel
                      container={activeContainer || null}
                      activeSubMask={activeSubMaskData || null}
                      aiModelDownloadStatus={aiModelDownloadStatus}
                      brushSettings={brushSettings}
                      setBrushSettings={setBrushSettings}
                      updateContainer={updatePatch}
                      updateSubMask={updateSubMask}
                      isGeneratingAi={isGeneratingAi}
                      isGeneratingAiMask={isGeneratingAiMask}
                      onGenerativeReplace={handleGenerativeReplace}
                      collapsibleState={collapsibleState}
                      setCollapsibleState={setCollapsibleState}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </>
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
                {(() => {
                  const item = activeDragItem.item as AiPatch;
                  const isStandalone =
                    item.subMasks.length === 1 && [Mask.Clone, Mask.Heal].includes(item.subMasks[0].type);
                  const Icon = isStandalone ? MASK_ICON_MAP[item.subMasks[0].type] || Circle : Wand2;
                  return <Icon size={18} className={TEXT_COLOR_KEYS[TextColors.secondary]} />;
                })()}
                <span className="flex-1 truncate">{(activeDragItem.item as AiPatch).name}</span>
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
                  const maskType = AI_SUB_MASK_COMPONENT_TYPES.find((m) => m.type === activeDragItem.maskType);
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
