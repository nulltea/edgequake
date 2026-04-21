/**
 * Custom sigma `defaultDrawNodeHover`. Sigma's built-in renderer paints
 * a hard-coded white hover box and then draws the label using
 * `settings.labelColor.color` — which in dark mode is a light slate
 * (#e2e8f0), so the label is invisible against white.
 *
 * This drawer keeps the same white box (matches sigma's look in light
 * mode and is more readable over arbitrary node colors than a dark
 * box) but forces a fixed dark text color for the label regardless of
 * the theme's normal label color.
 *
 * Logic is a direct port of sigma v3's `drawDiscNodeHover` at
 * `node_modules/sigma/dist/index-*.esm.js:658` — only the label-color
 * line is replaced.
 */
import type { Settings } from 'sigma/settings';
import type { NodeDisplayData, PartialButFor } from 'sigma/types';

const HOVER_LABEL_COLOR = '#111827'; // gray-900 — legible on white

export function drawNodeHoverReadable<
  N extends Record<string, unknown> = Record<string, unknown>,
  E extends Record<string, unknown> = Record<string, unknown>,
  G extends Record<string, unknown> = Record<string, unknown>,
>(
  context: CanvasRenderingContext2D,
  data: PartialButFor<NodeDisplayData, 'x' | 'y' | 'size' | 'label' | 'color'>,
  settings: Settings<N, E, G>,
): void {
  const size = settings.labelSize;
  const font = settings.labelFont;
  const weight = settings.labelWeight;
  context.font = `${weight} ${size}px ${font}`;

  context.fillStyle = '#FFF';
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.shadowBlur = 8;
  context.shadowColor = '#000';
  const PADDING = 2;

  if (typeof data.label === 'string') {
    const textWidth = context.measureText(data.label).width;
    const boxWidth = Math.round(textWidth + 5);
    const boxHeight = Math.round(size + 2 * PADDING);
    const radius = Math.max(data.size, size / 2) + PADDING;
    const angleRadian = Math.asin(boxHeight / 2 / radius);
    const xDeltaCoord = Math.sqrt(
      Math.abs(radius ** 2 - (boxHeight / 2) ** 2),
    );
    context.beginPath();
    context.moveTo(data.x + xDeltaCoord, data.y + boxHeight / 2);
    context.lineTo(data.x + radius + boxWidth, data.y + boxHeight / 2);
    context.lineTo(data.x + radius + boxWidth, data.y - boxHeight / 2);
    context.lineTo(data.x + xDeltaCoord, data.y - boxHeight / 2);
    context.arc(data.x, data.y, radius, angleRadian, -angleRadian);
    context.closePath();
    context.fill();
  } else {
    context.beginPath();
    context.arc(data.x, data.y, data.size + PADDING, 0, Math.PI * 2);
    context.closePath();
    context.fill();
  }

  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.shadowBlur = 0;

  if (!data.label) return;
  context.fillStyle = HOVER_LABEL_COLOR;
  context.font = `${weight} ${size}px ${font}`;
  context.fillText(
    data.label,
    data.x + data.size + 3,
    data.y + size / 3,
  );
}
