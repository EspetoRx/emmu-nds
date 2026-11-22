//! Synchronous 3D renderer bridge for the dust core.
//!
//! The upstream web frontend renders 3D scanlines in a dedicated Web Worker
//! using shared memory and atomics. Emmu's worker is already isolated and must
//! also work without COOP/COEP headers, so this module renders the scanlines
//! inline on the same thread. The core calls `swap_buffers`/`start_rendering`
//! and then reads scanlines through `SoftRendererRx` in the same `emu.run()`.

use dust_core::{
    gpu::{
        engine_3d::{
            Polygon, RendererTx, RenderingState as CoreRenderingState, ScreenVertex, SoftRendererRx,
        },
        Scanline, SCREEN_HEIGHT,
    },
    utils::Bytes,
};
use dust_soft_3d::{Renderer as Renderer3d, RenderingData};
use std::{cell::RefCell, rc::Rc};

pub struct RenderState {
    rendering_data: Box<RenderingData>,
    scanlines: Box<[Scanline<u32>; SCREEN_HEIGHT]>,
    renderer: Renderer3d,
}

impl RenderState {
    fn new() -> Self {
        Self {
            rendering_data: unsafe { Box::new_zeroed().assume_init() },
            scanlines: unsafe { Box::new_zeroed().assume_init() },
            renderer: Renderer3d::new(),
        }
    }

    fn render_all(&mut self) {
        let data = &*self.rendering_data;
        self.renderer.start_frame(data);
        self.renderer.render_line(0, data);
        for y in 0..SCREEN_HEIGHT {
            if y < SCREEN_HEIGHT - 1 {
                self.renderer.render_line(y as u8 + 1, data);
            }
            self.renderer
                .postprocess_line(y as u8, &mut self.scanlines[y], data);
        }
    }
}

pub struct Tx {
    state: Rc<RefCell<RenderState>>,
}

impl RendererTx for Tx {
    fn set_capture_enabled(&mut self, _capture_enabled: bool) {}

    fn swap_buffers(
        &mut self,
        vert_ram: &[ScreenVertex],
        poly_ram: &[Polygon],
        state: &CoreRenderingState,
    ) {
        self.state
            .borrow_mut()
            .rendering_data
            .prepare(vert_ram, poly_ram, state);
    }

    fn repeat_last_frame(&mut self, state: &CoreRenderingState) {
        self.state
            .borrow_mut()
            .rendering_data
            .repeat_last_frame(state);
    }

    fn start_rendering(
        &mut self,
        texture: &Bytes<0x8_0000>,
        tex_pal: &Bytes<0x1_8000>,
        state: &CoreRenderingState,
    ) {
        let mut render = self.state.borrow_mut();
        render.rendering_data.copy_vram(texture, tex_pal, state);
        render.render_all();
    }

    fn skip_rendering(&mut self) {}
}

pub struct Rx {
    state: Rc<RefCell<RenderState>>,
    next_scanline: usize,
}

impl SoftRendererRx for Rx {
    fn start_frame(&mut self) {
        self.next_scanline = 0;
    }

    fn read_scanline(&mut self) -> &Scanline<u32> {
        let index = self.next_scanline;
        self.next_scanline += 1;
        // Safe: the core reads exactly one scanline at a time and the render
        // state is only mutated by `Tx` between frames.
        let state = self.state.borrow();
        unsafe {
            let ptr = state.scanlines.as_ptr().add(index);
            &*ptr
        }
    }

    fn skip_scanline(&mut self) {
        self.next_scanline += 1;
    }
}

pub fn init() -> (Tx, Rx) {
    let state = Rc::new(RefCell::new(RenderState::new()));
    (
        Tx {
            state: state.clone(),
        },
        Rx {
            state,
            next_scanline: 0,
        },
    )
}
