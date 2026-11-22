//! WASM facade for the Nintendo DS core.
//!
//! Wraps `dust-core` (GPL-3.0) behind the Emmu worker protocol: stacked
//! 256x384 RGBA8888 frames, interleaved f32 stereo audio, button/touch input,
//! save states with the `EMMUNDS1` magic, cartridge save RAM and reset.

use crate::{AUDIO_SAMPLE_RATE, FRAME_HEIGHT, SAVE_MAGIC, SAVE_VERSION, SCREEN_WIDTH};
use dust_core::{
    audio::OutputSample,
    cpu::interpreter::Interpreter,
    ds_slot,
    emu::{self, input::Keys, Emu},
    flash::Flash,
    gpu::{SCREEN_HEIGHT as CORE_SCREEN_HEIGHT, SCREEN_WIDTH as CORE_SCREEN_WIDTH},
    rtc,
    spi::firmware,
    utils::BoxedByteSlice,
    Model, SaveReloadContents,
};
use emu_utils::{PersistentReadSavestate, PersistentWriteSavestate, ReadSavestate, WriteSavestate};
use js_sys::{Float32Array, Uint8Array};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use wasm_bindgen::prelude::*;

use crate::renderer_3d;

const GAME_DB: &str = include_str!("../vendor/dust/game_db.json");

#[derive(Deserialize)]
struct GameDbEntry {
    code: u32,
    #[serde(rename = "save-type")]
    save_type: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SaveKind {
    None,
    Eeprom4k,
    EepromFram64k,
    EepromFram512k,
    EepromFram1m,
    Flash2m,
    Flash4m,
    Flash8m,
}

impl SaveKind {
    fn expected_len(self) -> Option<usize> {
        Some(match self {
            Self::None => return None,
            Self::Eeprom4k => 0x200,
            Self::EepromFram64k => 0x2000,
            Self::EepromFram512k => 0x1_0000,
            Self::EepromFram1m => 0x2_0000,
            Self::Flash2m => 0x4_0000,
            Self::Flash4m => 0x8_0000,
            Self::Flash8m => 0x10_0000,
        })
    }

    fn from_len(len: usize) -> Option<Self> {
        Some(match len {
            0x200 => Self::Eeprom4k,
            0x2000 => Self::EepromFram64k,
            0x1_0000 => Self::EepromFram512k,
            0x2_0000 => Self::EepromFram1m,
            0x4_0000 => Self::Flash2m,
            0x8_0000 => Self::Flash4m,
            0x10_0000 => Self::Flash8m,
            _ => return None,
        })
    }

    fn from_db(value: &str) -> Option<Self> {
        Some(match value {
            "eeprom-4k" => Self::Eeprom4k,
            "eeprom-fram-64k" => Self::EepromFram64k,
            "eeprom-fram-512k" => Self::EepromFram512k,
            "eeprom-fram-1m" => Self::EepromFram1m,
            "flash-2m" => Self::Flash2m,
            "flash-4m" => Self::Flash4m,
            "flash-8m" => Self::Flash8m,
            _ => return None,
        })
    }
}

/// Button actions accepted by `set_button`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsButton {
    Up,
    Down,
    Left,
    Right,
    A,
    B,
    X,
    Y,
    L,
    R,
    Start,
    Select,
}

impl DsButton {
    fn key(self) -> Keys {
        match self {
            Self::Up => Keys::UP,
            Self::Down => Keys::DOWN,
            Self::Left => Keys::LEFT,
            Self::Right => Keys::RIGHT,
            Self::A => Keys::A,
            Self::B => Keys::B,
            Self::X => Keys::X,
            Self::Y => Keys::Y,
            Self::L => Keys::L,
            Self::R => Keys::R,
            Self::Start => Keys::START,
            Self::Select => Keys::SELECT,
        }
    }
}

struct AudioBackend {
    buffer: Arc<Mutex<VecDeque<f32>>>,
}

impl dust_core::audio::Backend for AudioBackend {
    fn handle_sample_chunk(&mut self, samples: &mut Vec<[OutputSample; 2]>) {
        let mut queue = self.buffer.lock().unwrap();
        for [left, right] in samples.drain(..) {
            #[allow(clippy::unnecessary_cast)]
            queue.push_back(left as f32 * (1.0 / 512.0) - 1.0);
            #[allow(clippy::unnecessary_cast)]
            queue.push_back(right as f32 * (1.0 / 512.0) - 1.0);
        }
    }
}

/// Cartridges whose game code starts with `I` (Pokémon Black/White, etc.) use
/// the IR-aware SPI protocol: their commands are prefixed with a `0x00` byte.
/// Reporting `has_ir = false` makes the game read that prefix as a flash
/// command and show "The save data could not be accessed" before booting.
fn cartridge_has_ir(game_code: u32) -> bool {
    game_code as u8 == b'I'
}

/// Encode Emmu's save state envelope. `dust-core`'s persistent savestate skips
/// the cartridge save RAM (`#[savestate(skip)]`), so it is stored here too;
/// otherwise in-game saves are lost whenever the state is reloaded.
fn encode_state(body: &[u8], save: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SAVE_MAGIC.len() + 12 + body.len() + save.len());
    out.extend_from_slice(SAVE_MAGIC);
    out.extend_from_slice(&SAVE_VERSION.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&(save.len() as u32).to_le_bytes());
    out.extend_from_slice(save);
    out
}

struct DecodedState<'a> {
    body: &'a [u8],
    save: Option<&'a [u8]>,
}

/// Decode a save state envelope. Version 1 is accepted for recovery; it has no
/// cartridge save RAM section and no explicit body length.
fn decode_state(bytes: &[u8]) -> Result<DecodedState<'_>, String> {
    let header = SAVE_MAGIC.len() + 4;
    if bytes.len() < header || &bytes[..SAVE_MAGIC.len()] != SAVE_MAGIC {
        return Err("Save state DS inválido (magic)".to_string());
    }
    let version = u32::from_le_bytes([
        bytes[SAVE_MAGIC.len()],
        bytes[SAVE_MAGIC.len() + 1],
        bytes[SAVE_MAGIC.len() + 2],
        bytes[SAVE_MAGIC.len() + 3],
    ]);
    match version {
        1 => Ok(DecodedState {
            body: &bytes[header..],
            save: None,
        }),
        2 => {
            if bytes.len() < header + 4 {
                return Err("Save state DS truncado (tamanho do corpo)".to_string());
            }
            let body_len = u32::from_le_bytes([
                bytes[header],
                bytes[header + 1],
                bytes[header + 2],
                bytes[header + 3],
            ]) as usize;
            let body_start = header + 4;
            let save_len_pos = body_start
                .checked_add(body_len)
                .ok_or_else(|| "Save state DS inválido (tamanho do corpo)".to_string())?;
            let save_pos = save_len_pos
                .checked_add(4)
                .ok_or_else(|| "Save state DS inválido (tamanho do save)".to_string())?;
            if bytes.len() < save_pos {
                return Err("Save state DS truncado (corpo)".to_string());
            }
            let save_len = u32::from_le_bytes([
                bytes[save_len_pos],
                bytes[save_len_pos + 1],
                bytes[save_len_pos + 2],
                bytes[save_len_pos + 3],
            ]) as usize;
            let save_end = save_pos
                .checked_add(save_len)
                .ok_or_else(|| "Save state DS inválido (tamanho do save)".to_string())?;
            if save_end != bytes.len() {
                return Err("Save state DS inválido (tamanho do save)".to_string());
            }
            Ok(DecodedState {
                body: &bytes[body_start..save_len_pos],
                save: Some(&bytes[save_pos..save_end]),
            })
        }
        other => Err(format!(
            "Save state DS incompatível: versão {other} != {SAVE_VERSION}"
        )),
    }
}

fn make_ds_slot_spi(
    save_kind: SaveKind,
    contents: Option<BoxedByteSlice>,
    has_ir: bool,
) -> ds_slot::spi::Spi {
    let Some(kind) = (match save_kind {
        SaveKind::None => None,
        other => Some(other),
    }) else {
        return ds_slot::spi::Empty::new().into();
    };

    let expected = kind.expected_len().unwrap();
    let contents = match contents {
        Some(bytes) => {
            if bytes.len() == expected {
                dust_core::SaveContents::Existing(bytes)
            } else {
                let mut resized = BoxedByteSlice::new_zeroed(expected);
                let copy_len = bytes.len().min(expected);
                resized[..copy_len].copy_from_slice(&bytes[..copy_len]);
                dust_core::SaveContents::Existing(resized)
            }
        }
        None => dust_core::SaveContents::New(expected),
    };

    match kind {
        SaveKind::Eeprom4k => ds_slot::spi::eeprom_4k::Eeprom4k::new(contents, None)
            .expect("4 Kib EEPROM")
            .into(),
        SaveKind::EepromFram64k | SaveKind::EepromFram512k | SaveKind::EepromFram1m => {
            ds_slot::spi::eeprom_fram::EepromFram::new(contents, None)
                .expect("EEPROM/FRAM")
                .into()
        }
        SaveKind::Flash2m | SaveKind::Flash4m | SaveKind::Flash8m => {
            ds_slot::spi::flash::Flash::new(contents, [0; 20], has_ir)
                .expect("FLASH")
                .into()
        }
        SaveKind::None => unreachable!(),
    }
}

/// The Nintendo DS emulator wrapper.
#[wasm_bindgen]
pub struct NdsEmulator {
    emu: Option<Emu<Interpreter>>,
    frame: Vec<u8>,
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    volume: f32,
    keys: Keys,
    save_kind: SaveKind,
    model: Model,
}

#[wasm_bindgen]
impl NdsEmulator {
    /// Create a new emulator from ROM bytes. BIOS and firmware are provided by
    /// the core (HLE BIOS + generated default firmware), so no copyrighted
    /// system files are required.
    #[wasm_bindgen(constructor)]
    pub fn new(rom: &[u8]) -> NdsEmulator {
        console_error_panic_hook::set_once();

        let audio_buffer = Arc::new(Mutex::new(VecDeque::new()));
        let audio_backend = Box::new(AudioBackend {
            buffer: audio_buffer.clone(),
        });
        let model = Model::Lite;

        let game_code = u32::from_le_bytes([rom[0x0c], rom[0x0d], rom[0x0e], rom[0x0f]]);
        let has_ir = cartridge_has_ir(game_code);
        let save_kind = serde_json::from_str::<Vec<GameDbEntry>>(GAME_DB)
            .ok()
            .and_then(|db| db.into_iter().find(|entry| entry.code == game_code))
            .and_then(|entry| SaveKind::from_db(&entry.save_type))
            .unwrap_or(SaveKind::None);

        let rom_len = rom.len().next_power_of_two();
        let mut rom_bytes = BoxedByteSlice::new_zeroed(rom_len);
        rom_bytes[..rom.len()].copy_from_slice(rom);

        let (tx_3d, rx_3d) = renderer_3d::init();

        let mut builder = emu::Builder::new(
            Flash::new(
                dust_core::SaveContents::Existing(firmware::default(model)),
                firmware::id_for_model(model),
            )
            .expect("default firmware"),
            Some(Box::new(rom_bytes)),
            make_ds_slot_spi(save_kind, None, has_ir),
            audio_backend,
            None,
            Box::new(rtc::DummyBackend),
            Box::new(dust_soft_2d::sync::Renderer::new(Box::new(rx_3d))),
            Box::new(tx_3d),
            None,
        );
        builder.model = model;
        builder.direct_boot = true;

        let emu = match builder.build(Interpreter) {
            Ok(emu) => emu,
            Err(_) => panic!("Failed to build Nintendo DS emulator"),
        };

        NdsEmulator {
            emu: Some(emu),
            frame: vec![0u8; SCREEN_WIDTH * FRAME_HEIGHT * 4],
            audio_buffer,
            volume: 1.0,
            keys: Keys::empty(),
            save_kind,
            model,
        }
    }

    /// Advance one frame and return a stacked 256x384 RGBA8888 image.
    #[wasm_bindgen(js_name = runFrame)]
    pub fn run_frame(&mut self) -> Uint8Array {
        self.emu.as_mut().unwrap().run();

        let framebuffer = self.emu.as_ref().unwrap().gpu.renderer_2d().framebuffer();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                framebuffer.as_ptr() as *const u8,
                CORE_SCREEN_WIDTH as usize * CORE_SCREEN_HEIGHT as usize * 2 * 4,
            )
        };
        self.frame.copy_from_slice(bytes);
        Uint8Array::from(&self.frame[..])
    }

    pub fn width(&self) -> u32 {
        SCREEN_WIDTH as u32
    }

    pub fn height(&self) -> u32 {
        FRAME_HEIGHT as u32
    }

    /// Set a button state. `player` is accepted for protocol compatibility;
    /// the DS has a single controller.
    #[wasm_bindgen(js_name = setButton)]
    pub fn set_button(&mut self, button: DsButton, pressed: bool, _player: u32) {
        let bit = button.key();
        let next = if pressed {
            self.keys | bit
        } else {
            self.keys & !bit
        };
        let emu = self.emu.as_mut().unwrap();
        emu.release_keys(self.keys & !next);
        emu.press_keys(next & !self.keys);
        self.keys = next;
    }

    /// Touch the lower screen (DS LCD coordinates).
    #[wasm_bindgen(js_name = setTouch)]
    pub fn set_touch(&mut self, x: u16, y: u16) {
        self.emu.as_mut().unwrap().set_touch_pos([x, y]);
    }

    /// Release the touch screen.
    #[wasm_bindgen(js_name = endTouch)]
    pub fn end_touch(&mut self) {
        self.emu.as_mut().unwrap().end_touch();
    }

    #[wasm_bindgen(js_name = setVolume)]
    pub fn set_volume(&mut self, volume: f64) {
        self.volume = volume.clamp(0.0, 1.0) as f32;
    }

    /// Return up to `max_samples` interleaved stereo f32 samples.
    pub fn audio(&mut self, max_samples: u32) -> Float32Array {
        let max = (max_samples as usize).min(8192);
        let mut out = vec![0f32; max * 2];
        let mut queue = self.audio_buffer.lock().unwrap();
        let mut i = 0usize;
        while i + 1 < out.len() {
            let Some(left) = queue.pop_front() else { break };
            let right = queue.pop_front().unwrap_or(left);
            out[i] = left * self.volume;
            out[i + 1] = right * self.volume;
            i += 2;
        }
        drop(queue);
        Float32Array::from(&out[..i])
    }

    #[wasm_bindgen(js_name = audioSampleRate)]
    pub fn audio_sample_rate(&self) -> u32 {
        AUDIO_SAMPLE_RATE
    }

    #[wasm_bindgen(js_name = audioChannels)]
    pub fn audio_channels(&self) -> u32 {
        2
    }

    /// Reset the emulation, preserving ROM, save RAM and renderers.
    pub fn restart(&mut self) {
        let emu = self.emu.take().unwrap();
        let (renderer_2d, renderer_3d_tx) = emu.gpu.into_renderers();
        let mut builder = emu::Builder::new(
            emu.spi.firmware.reset(),
            emu.ds_slot.rom.into_contents(),
            emu.ds_slot.spi.reset(),
            emu.audio.backend,
            None,
            emu.rtc.backend,
            renderer_2d,
            renderer_3d_tx,
            None,
        );
        builder.model = self.model;
        builder.direct_boot = true;
        self.emu = Some(match builder.build(Interpreter) {
            Ok(emu) => emu,
            Err(_) => panic!("Failed to reset Nintendo DS emulator"),
        });
        self.keys = Keys::empty();
        self.audio_buffer.lock().unwrap().clear();
    }

    /// Serialize the emulation state with the `EMMUNDS1` magic prefix and the
    /// cartridge save RAM appended (dust-core skips it in its own savestate).
    #[wasm_bindgen(js_name = saveState)]
    pub fn save_state(&mut self) -> Result<Uint8Array, JsValue> {
        let mut body = Vec::new();
        PersistentWriteSavestate::new(&mut body)
            .store(self.emu.as_mut().unwrap())
            .map_err(|e| JsValue::from_str(&format!("Falha ao salvar estado DS: {e:?}")))?;
        let save = self.emu.as_ref().unwrap().ds_slot.spi.contents().to_vec();
        let out = encode_state(&body, &save);
        Ok(Uint8Array::from(&out[..]))
    }

    /// Load a state, validating magic and version before touching the core.
    /// Version 2 also restores the cartridge save RAM captured with the state.
    #[wasm_bindgen(js_name = loadState)]
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let decoded = decode_state(bytes).map_err(|message| JsValue::from_str(&message))?;
        PersistentReadSavestate::new(decoded.body)
            .and_then(|mut state| state.load_into(self.emu.as_mut().unwrap()).map_err(drop))
            .map_err(|e| JsValue::from_str(&format!("Falha ao carregar estado DS: {e:?}")))?;
        if let Some(save) = decoded.save {
            let expected = self.save_kind.expected_len().unwrap_or(save.len());
            let mut contents = BoxedByteSlice::new_zeroed(expected);
            let copy_len = contents.len().min(save.len());
            contents[..copy_len].copy_from_slice(&save[..copy_len]);
            self.emu
                .as_mut()
                .unwrap()
                .ds_slot
                .spi
                .reload_contents(SaveReloadContents::Existing(contents));
        }
        self.audio_buffer.lock().unwrap().clear();
        Ok(())
    }

    /// Export the cartridge save RAM (`.sav`).
    #[wasm_bindgen(js_name = exportSave)]
    pub fn export_save(&self) -> Uint8Array {
        let contents = self.emu.as_ref().unwrap().ds_slot.spi.contents();
        Uint8Array::from(contents)
    }

    /// Import a cartridge save RAM (`.sav`).
    #[wasm_bindgen(js_name = importSave)]
    pub fn import_save(&mut self, bytes: &[u8]) {
        let expected = self
            .save_kind
            .expected_len()
            .or_else(|| SaveKind::from_len(bytes.len()).and_then(|kind| kind.expected_len()))
            .unwrap_or(bytes.len());
        let mut contents = BoxedByteSlice::new_zeroed(expected);
        let copy_len = contents.len().min(bytes.len());
        contents[..copy_len].copy_from_slice(&bytes[..copy_len]);
        self.emu
            .as_mut()
            .unwrap()
            .ds_slot
            .spi
            .reload_contents(SaveReloadContents::Existing(contents));
    }

    #[wasm_bindgen(js_name = setPaused)]
    pub fn set_paused(&mut self, _paused: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_cartridges_are_detected_from_game_code() {
        // "IRAO" (Pokémon White), stored little-endian at ROM offset 0x0C.
        assert!(cartridge_has_ir(0x4F41_5249));
        // "IRBJ" (Pokémon Black) and "IRBO" also start with 'I'.
        assert!(cartridge_has_ir(0x4A42_5249));
        // "A2DE" (New Super Mario Bros.) must keep the plain flash protocol.
        assert!(!cartridge_has_ir(0x4544_3241));
    }

    #[test]
    fn state_envelope_round_trips_body_and_save_ram() {
        let bytes = encode_state(&[1, 2, 3, 4], &[0xAA, 0xBB]);
        let decoded = decode_state(&bytes).expect("decode state");
        assert_eq!(decoded.body, &[1, 2, 3, 4]);
        assert_eq!(decoded.save, Some(&[0xAA, 0xBB][..]));
    }

    #[test]
    fn legacy_state_without_save_ram_is_accepted() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SAVE_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[9, 8, 7]);
        let decoded = decode_state(&bytes).expect("decode legacy state");
        assert_eq!(decoded.body, &[9, 8, 7]);
        assert_eq!(decoded.save, None);
    }

    #[test]
    fn invalid_states_are_rejected() {
        let mut bad_magic = encode_state(&[1], &[2]);
        bad_magic[0] = b'X';
        assert!(decode_state(&bad_magic).is_err());

        let mut bad_version = encode_state(&[1], &[2]);
        bad_version[SAVE_MAGIC.len()] = 9;
        assert!(decode_state(&bad_version).is_err());

        let mut truncated = encode_state(&[1, 2, 3], &[4, 5]);
        truncated.pop();
        assert!(decode_state(&truncated).is_err());
    }
}
