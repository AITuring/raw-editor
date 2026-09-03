# RAW Editor design tokens

The UI uses three token layers in `tokens.css`, following the same separation used by token-based component systems:

1. **Primitive** — spacing, sizing, typography, radius, and elevation scales.
2. **Semantic** — panel inset, toolbar height, control border, chrome surface, and viewport inset.
3. **Component** — values that belong to a stable reusable pattern, such as the filter popover width.

Theme palettes remain in `src/utils/themes.ts` because they are switched at runtime. Those `--app-*` variables are the color seed tokens; `tokens.css` derives control, structural, and floating borders from them so every theme keeps the same hierarchy.

The main density controls are:

| Intent                 | Token                                                                                           | Default          |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ---------------- |
| General panel inset    | `--ui-panel-inset`                                                                              | `20px`           |
| Editor control inset   | `--ui-editor-panel-inset`                                                                       | `24px`           |
| Floating viewport gap  | `--ui-floating-viewport-inset`                                                                  | `24px`           |
| Dialog content gutter  | `--ui-dialog-inset`                                                                             | `24px`           |
| Dialog footer rhythm   | `--ui-dialog-footer-gap` / `--ui-dialog-footer-padding-block`                                   | shared           |
| Denoise dialog / grid  | `--ui-denoise-dialog-max-width` / `--ui-denoise-control-column` / `--ui-denoise-preview-height` | shared           |
| Standard action button | `--ui-button-height`                                                                            | `28px`           |
| Compact action button  | `--ui-button-height-sm`                                                                         | `24px`           |
| Button rest elevation  | `--ui-shadow-button-rest`                                                                       | shared           |
| Button hover elevation | `--ui-shadow-button-hover`                                                                      | shared           |
| Button press elevation | `--ui-shadow-button-pressed`                                                                    | shared           |
| Selected button state  | `--ui-shadow-button-selected`                                                                   | shared           |
| Quick-filter width     | `--ui-filter-popover-width`                                                                     | `160px`          |
| Quick-filter gap       | `--ui-filter-popover-trigger-gap`                                                               | `12px`           |
| Copy/paste dialog      | `--ui-copy-paste-dialog-width`                                                                  | `560px`          |
| Toolbar / panel header | `--ui-toolbar-height` / `--ui-panel-header-height`                                              | `44px`           |
| Bottom status bar      | `--ui-statusbar-height`                                                                         | `40px`           |
| Right tool rail        | `--ui-tool-rail-width`                                                                          | `48px`           |
| Standard control       | `--ui-size-control`                                                                             | `32px`           |
| Compact icon target    | `--ui-size-icon-hit`                                                                            | `28px`           |
| App panel gap          | `--ui-shell-gap`                                                                                | `8px`            |
| Message surface        | `--ui-message-min-height` / `--ui-message-max-width`                                            | `34px` / `420px` |
| Message elevation      | `--ui-shadow-message`                                                                           | shared           |

The adjustment panel has a second, denser rhythm layered on top of the common
panel scale. `--ui-editor-adjustment-inset` controls the safe content gutter;
`--ui-editor-field-label-column` and `--ui-editor-value-column` keep labels,
values, and slider rails aligned; the section and subsection height tokens keep
headings predictable as panels are reused. A container query narrows these
columns for compact right-panel widths without changing the underlying controls.
`--ui-editor-scrollbar-reserve` keeps the fixed header/profile actions on the
same right edge as the scrollable adjustment content.
Native selection fields in the adjustment panel use the same raised-surface
shadow and left-aligned value treatment, so a dropdown reads as a control
without adding a perimeter border.
Transient messages use the same viewport inset as other floating surfaces. They
are intentionally borderless, compact, and tone their status icon instead of
filling it with a saturated color; this keeps validation feedback visible
without covering the image or competing with editor controls.

Component styles should use semantic tokens. For example:

```css
.example-panel {
  padding: var(--ui-panel-inset);
  border: 1px solid var(--ui-border-structural);
}
```

Text actions use the shared `Button` component or one of the specialized
interaction classes. Use `primary` for the single commit action, `secondary`
for cancel/reset/select actions, `ui-segmented-option` for mutually exclusive
modes, and `ui-choice-button` for discrete options. Dense panel-header actions
use `size="sm"`; do not recreate buttons with local font and padding utilities.

Buttons never use perimeter borders. Filled or raised controls use the shared
button elevation tokens through `Button`, a specialized button class, or
`ui-surface-button`. Flat icon, tab, and inline actions may remain shadowless,
but they must also remain borderless. Focus outlines are retained for keyboard
accessibility. Borders still belong on structural surfaces and editable input
fields; the rule is specific to clickable buttons.

Use the shared layout classes when possible:

- `ui-panel-root` — full-height panel layout.
- `ui-panel-header` — standard 44px panel heading with the shared panel inset.
- `ui-panel-body` — scrollable panel content with the standard inset.
- `ui-panel-footer` — fixed panel footer with structural divider.
- `ui-toolbar` — standard application toolbar geometry.
- `ui-chrome-panel` — bordered application panel shell.
- `ui-icon-button` / `ui-icon-button--md` — reusable icon-only controls.
- `ui-select-trigger` / `ui-select-option` — shared dropdown geometry.
- `ui-segmented-control` — shared view and filter segmented controls.
- `app-modal-surface--padded` — inset form/dialog surface with a protected content gutter.
- `app-modal-surface--full-bleed` — explicit exception for image-workspace dialogs.
- `app-modal-footer` — aligned, tokenized modal action row.

Rules:

- Do not add local `p-3`/`p-4` values to panel shells; change `--ui-panel-inset` instead.
- Keep editor controls on `--ui-editor-panel-inset` and floating surfaces on
  `--ui-floating-viewport-inset`; they solve different boundary problems.
- Component-sized overlays use component tokens; do not grow them by changing
  the global panel or dialog inset.
- Use `--ui-border-control`, `--ui-border-structural`, or `--ui-border-floating` according to intent.
- Do not add `border*` utilities to `button` or `Button`; use the shared elevation states instead.
- Keep Lucide icons, callbacks, shortcuts, and state behavior separate from visual tokens.
- Add a new token only when a value is reused or represents a stable semantic role.
