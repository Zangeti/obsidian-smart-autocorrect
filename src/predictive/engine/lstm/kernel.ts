/**
 * SIMD kernel for the LSTM hot loops, compiled from build_model/kernel.wat and
 * inlined here as base64 - deliberately NOT a separate .wasm file: Obsidian's
 * community-plugin installer only fetches main.js/manifest.json/styles.css, so a
 * sidecar binary would never survive a marketplace install. Inlining keeps the
 * plugin a single self-contained main.js with no toolchain on the user's machine.
 *
 * Exports (see kernel.wat): project_f32, matvec_acc_f32, quantise_vec, project_i8.
 * Regenerate with: node build_model/build_kernel.mjs  (dev-only, needs `wabt`).
 */
export const KERNEL_B64 =
  "AGFzbQEAAAABJARgBn9/f39/fwBgBX9/f39/AGAEf39/fQBgCH9/f39/f399AAMFBAABAgMFAwEAAQdCBQNtZW0CAAtwcm9qZWN0X2YzMgAADm1hdHZlY19hY2NfZjMyAAEMcXVhbnRpc2VfdmVjAAIKcHJvamVjdF9pOAADCvcEBLEBAwN/AXsBfUEAIQYCQANAIAYgBE8NASAAIAYgBWxBAnRqIQj9DAAAAAAAAAAAAAAAAAAAAAAhCUEAIQcCQANAIAcgBU8NASAJIAggB0ECdGr9AAQAIAIgB0ECdGr9AAQA/eYB/eQBIQkgB0EEaiEHDAALCyAJ/R8AIAn9HwGSIAn9HwIgCf0fA5KSIQogAyAGQQJ0aiABIAZBAnRqKgIAIAqSOAIAIAZBAWohBgwACwsLsQEDA38BewF9QQAhBQJAA0AgBSADTw0BIAAgBSAEbEECdGohB/0MAAAAAAAAAAAAAAAAAAAAACEIQQAhBgJAA0AgBiAETw0BIAggByAGQQJ0av0ABAAgASAGQQJ0av0ABAD95gH95AEhCCAGQQRqIQYMAAsLIAj9HwAgCP0fAZIgCP0fAiAI/R8DkpIhCSACIAVBAnRqIAIgBUECdGoqAgAgCZI4AgAgBUEBaiEFDAALCws3AQF/QQAhBAJAA0AgBCACTw0BIAEgBGogACAEQQJ0aioCACADlJD8ADoAACAEQQFqIQQMAAsLC9YBAwN/A3sBf0EAIQgCQANAIAggBU8NASAAIAggBmxqIQr9DAAAAAAAAAAAAAAAAAAAAAAhC0EAIQkCQANAIAkgBk8NASAKIAlq/QAEACEMIAMgCWr9AAQAIQ0gCyAM/YcBIA39hwH9ugEgDP2IASAN/YgB/boB/a4B/a4BIQsgCUEQaiEJDAALCyAL/RsAIAv9GwFqIAv9GwIgC/0bA2pqIQ4gBCAIQQJ0aiACIAhBAnRqKgIAIA6yIAEgCEECdGoqAgCUIAeUkjgCACAIQQFqIQgMAAsLCw==";
export const KERNEL_BYTES = 760;
