# emmu-nds

Nintendo DS core for Emmu, wrapping [`dust`](https://github.com/Kelpsyberry/dust)
(GPL-3.0) behind the same worker protocol used by the other Emmu cores.

The core runs as WASM on **both web and desktop**: MesenCE has no Nintendo DS
implementation, so Tauri loads the same WASM module inside the WebView, exactly
like the N64 exception.

## API

- `new NdsEmulator(rom: Uint8Array)` — direct boot with generated firmware and
  HLE BIOS; no copyrighted system files required.
- `runFrame(): Uint8Array` — stacked 256×384 RGBA8888 image (top screen above
  bottom screen).
- `setButton(button, pressed, player)` — `DsButton` enum; player is accepted
  for protocol compatibility (the DS has one controller).
- `setTouch(x, y)` / `endTouch()` — lower-screen touch panel.
- `audio(maxSamples): Float32Array` — interleaved stereo PCM, 32.768 kHz (the native rate of `dust-core` without `xq-audio`).
- `setVolume(volume)` — 0.0..1.0.
- `saveState(): Uint8Array` — `EMMUNDS1` + u32 version + dust savestate + cartridge save RAM.
- `loadState(bytes)` — validates magic/version and restores the save RAM.
- `exportSave(): Uint8Array` / `importSave(bytes)` — cartridge save RAM.
- `restart()` — rebuilds the emulator preserving ROM, save RAM and renderers.

## Save states

Magic `EMMUNDS1` + little-endian u32 version (currently `2`) + two length-
prefixed sections: the savestate produced by `PersistentWriteSavestate` and the
raw cartridge save RAM. Version 1 states are still accepted for recovery; they
have no save RAM section. Invalid magic, version or lengths are rejected.

`dust-core` marks the raw save RAM fields with `#[savestate(skip)]`, so without
the appended section an in-game save would be lost whenever the state was
reloaded (new core instance, app reload, save-state history). The save RAM is
now captured and restored together with each state.

Cartridge save RAM uses the save type from `vendor/dust/game_db.json` (matched
by ROM game code). NAND saves are not supported by dust and fall back to no
save, matching upstream behavior. Game codes starting with `I` (Pokémon
Black/White and other IR cartridges) enable the IR-aware flash protocol, exactly
as dust's own desktop/web frontends do. Note that the pinned dust commit still
shows "The save data could not be accessed" with 256 MiB NDSi Enhanced dumps
such as Pokémon White in both upstream frontends; this is a core limitation,
not a save-state envelope problem.

## Rendering

The upstream web frontend renders 3D scanlines in a separate Web Worker using
`SharedArrayBuffer` and atomics, which requires COOP/COEP headers. `src/renderer_3d.rs`
renders the scanlines synchronously on the same thread instead, so Emmu works
on ordinary static hosting.

## Build

```sh
rustup toolchain install nightly-2025-12-15 --profile minimal --target wasm32-unknown-unknown
wasm-pack build --target web --out-dir ../../platforms/web/ui/src/wasm/nds --out-name emmu_nds
cp js/adapter.js ../../platforms/web/ui/src/wasm/nds/
```

`npm run build:wasm` in the main repository performs all of the above. The
generated module is imported directly from `src/wasm/nds/` by the worker and
bundled by Vite; unlike N64Wasm, NDS needs no separate runtime assets in
`public/`.

## Testing

```sh
cargo check --lib
cargo test --lib
```

Smoke test with a real ROM: open a `.nds` file in the Emmu UI and verify video
on both screens, audio, A/B/X/Y/L/R/Start/Select, touch, save/load state,
restart and volume.

## License

GPL-3.0-only. See `LICENSE` and `UPSTREAM.txt` for vendored code provenance.
