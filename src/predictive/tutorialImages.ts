/**
 * Pictures for the getting-started tour, inlined as data URIs.
 *
 * They have to be inlined: Obsidian's community installer only ever fetches main.js,
 * manifest.json and styles.css, so a plugin cannot ship loose image files (the same reason the
 * language model is downloaded separately, and the same approach bmcQr.ts already uses).
 *
 * Each one is a real screenshot of the plugin working, cropped tight to the thing the step is
 * about. Keep them small - they are carried in main.js by every user - and keep them at 2x the
 * displayed width so they stay sharp on a retina display. The tour renders a step without its
 * picture if the entry here is undefined, so an empty slot degrades to text rather than
 * breaking.
 */
export const TUTORIAL_IMAGES: Record<"suggest" | "autocorrect" | "undo" | "links", string | undefined> = {
  suggest: undefined,
  autocorrect: undefined,
  undo: undefined,
  links: undefined,
};
