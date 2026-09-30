// Speaker colours. Seven categorical hues, fixed order, validated for colour-vision
// deficiency against white and the paper ground (dataviz validate_palette.js: no FAIL;
// the WARN pairs are covered by the speaker name always shown next to the colour).
// Brand orange is left out on purpose: it marks the playhead and actions.
// Speakers 8+ reuse the hues with a hatched fill, so no generated hue is ever needed.
export const HUES = ["#2a78d6", "#eda100", "#4a3aa7", "#1baf7a", "#e87ba4", "#008300", "#e34948"];

export const speakerColor = (i) => HUES[((i % HUES.length) + HUES.length) % HUES.length];
export const speakerHatched = (i) => i >= HUES.length;

export function swatchStyle(i) {
  const c = speakerColor(i);
  return speakerHatched(i)
    ? `background: repeating-linear-gradient(135deg, ${c} 0 3px, #ffffff 3px 5px)`
    : `background: ${c}`;
}
