// Adapter between the emmu-nds WASM facade and Emmu's worker protocol.
//
// This file lives inside the crate and is copied to `src/wasm/nds/` by
// `npm run build:wasm`. The vendored dust core is never edited at runtime.

import initWasm, { NdsEmulator } from "./emmu_nds.js";

const FRAME_W = 256;
const FRAME_H = 384;
// dust-core emits 32768 Hz when built without the optional `xq-audio` feature.
const AUDIO_RATE = 32768;
const AUDIO_CHANNELS = 2;
const MAX_AUDIO_SAMPLES = 4096;

const BUTTON_MAP = {
  Up: 0,
  Down: 1,
  Left: 2,
  Right: 3,
  A: 4,
  B: 5,
  X: 6,
  Y: 7,
  L: 8,
  R: 9,
  Start: 10,
  Select: 11,
};

let wasmInit;

/** Initialize the wasm-bindgen module once per worker. */
export function initializeNdsWasm(input) {
  if (!wasmInit) {
    wasmInit = Promise.resolve(initWasm(input)).then(() => undefined);
  }
  return wasmInit;
}

/**
 * Worker-facing Nintendo DS core. Mirrors the object output contract used by
 * the other WASM cores (frame + audio metadata in a single result).
 */
export class NdsCoreAdapter {
  constructor(rom) {
    this.core = new NdsEmulator(rom);
    this.volume = 1;
  }

  async initialize() {}

  runFrame() {
    const frame = this.core.runFrame();
    const audioSamples = this.core.audio(MAX_AUDIO_SAMPLES);
    return {
      frame,
      width: FRAME_W,
      height: FRAME_H,
      audioSamples,
      audioSampleRate: AUDIO_RATE,
      audioChannels: AUDIO_CHANNELS,
    };
  }

  setButton(button, pressed, player) {
    const mapped = BUTTON_MAP[button];
    if (mapped === undefined) return;
    this.core.setButton(mapped, !!pressed, player || 1);
  }

  setVolume(volume) {
    this.volume = Math.max(0, Math.min(1, Number(volume) || 0));
    this.core.setVolume(this.volume);
  }

  setTouch(x, y) {
    this.core.setTouch(x, y);
  }

  endTouch() {
    this.core.endTouch();
  }

  restart() {
    this.core.restart();
  }

  saveState() {
    return this.core.saveState();
  }

  loadState(bytes) {
    this.core.loadState(bytes);
  }

  exportSave() {
    return this.core.exportSave();
  }

  importSave(bytes) {
    this.core.importSave(bytes);
  }

  free() {
    this.core.free();
  }
}
