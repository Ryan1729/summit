#![deny(unused)]
#![deny(bindings_with_variant_name)]

extern crate alloc;
use alloc::vec::Vec;

struct Storage<A>(Vec<A>);

impl <A> game::ClearableStorage<A> for Storage<A> {
    fn clear(&mut self) {
        self.0.clear();
    }

    fn push(&mut self, a: A) {
        self.0.push(a);
    }
}

const SAMPLING_SHADER: &str = include_str!("../assets/sampling.fs");

const SPRITESHEET_BYTES: &[u8] = include_bytes!("../assets/spritesheet.png");

const SPRITE_PIXELS_PER_TILE_SIDE: f32 = 16.0;

use game::{SpriteKind, ArrowKind, Dir};

struct SourceSpec {
    x: f32,
    y: f32,
}

fn source_spec(sprite: SpriteKind) -> SourceSpec {
    use ArrowKind::*;
    use Dir::*;
    use SpriteKind::*;

    let sx = match sprite {
        NeutralEye
        | Arrow(_, Red)
        | SmallPupilEye
        | NarrowLeftEye
        | NarrowCenterEye
        | NarrowRightEye
        | ClosedEye
        | HalfLidEye => 0.,
        Arrow(_, Green)| DirEye(_) => 1.
    };

    let sy = match sprite {
        Arrow(Up, _) => 0.,
        Arrow(UpRight, _) => 1.,
        Arrow(Right, _) => 2.,
        Arrow(DownRight, _) => 3.,
        Arrow(Down, _) => 4.,
        Arrow(DownLeft, _) => 5.,
        Arrow(Left, _) => 6.,
        Arrow(UpLeft, _) => 7.,
        DirEye(Up) => 8.,
        DirEye(UpRight) => 9.,
        DirEye(Right) => 10.,
        DirEye(DownRight) => 11.,
        DirEye(Down) => 12.,
        DirEye(DownLeft) => 13.,
        DirEye(Left) => 14.,
        DirEye(UpLeft) => 15.,
        ClosedEye => 8.,
        HalfLidEye => 9.,
        NeutralEye => 10.,
        NarrowCenterEye => 11.,
        NarrowRightEye => 12.,
        NarrowLeftEye => 13.,
        SmallPupilEye => 14.,
    };

    SourceSpec {
        x: sx * SPRITE_PIXELS_PER_TILE_SIDE,
        y: sy * SPRITE_PIXELS_PER_TILE_SIDE,
    }
}

const WINDOW_TITLE: &str = "summit";

fn main() {
    raylib_rs_platform::inner_main();
}

/// Let's keep all the raylib specific stuff in one module to make it easier to add
/// any different backends later.
mod raylib_rs_platform {
    use super::{
        Storage,
        source_spec,
        SPRITE_PIXELS_PER_TILE_SIDE,
        SPRITESHEET_BYTES,
        SAMPLING_SHADER,
        WINDOW_TITLE
    };
    use raylib::prelude::{
        *,
        KeyboardKey::*,
        ffi::{
            LoadImageFromMemory,
            MouseButton::MOUSE_LEFT_BUTTON,
        },
        core::{
            drawing::{RaylibTextureModeExt, RaylibShaderModeExt},
            logging
        }
    };

    use ::core::{
        convert::TryInto,
    };

    fn draw_wh(rl: &RaylibHandle) -> game::DrawWH {
        game::DrawWH {
            w: rl.get_screen_width() as game::DrawW,
            h: rl.get_screen_height() as game::DrawH,
        }
    }

    pub fn inner_main() {
        let start_paused = {
            let mut args = std::env::args();

            args.next(); // exe name

            args.next()
                .map(|arg| arg.to_lowercase().contains("pause"))
                .unwrap_or(false)
        };

        let (mut rl, thread) = {
            // TODO: Read display size ourselves, since while raylib tries to figure
            // out the right size if `0, 0` is passed, it sometimes gets the wrong
            // answer. In particular, on my current dev setup which uses Linux.
            // Since as of this writing, I'm unable to find a small cross-platform
            // crate for this, one option to get the info is to copy what the
            // `winit` crate does. However, that turns out to be rather complicated,
            // even just for x11, and hardcoding it for Linux currently seems
            // preferable to either including winit as a dependency without using
            // most of it, or spending the time to whittle away all the parts we
            // don't need, since we just want the current monitor's size.
            #[cfg(target_os = "linux")]
            const W: i32 = 1920;
            #[cfg(target_os = "linux")]
            const H: i32 = 1080;

            #[cfg(not(target_os = "linux"))]
            const W: i32 = 0;
            #[cfg(not(target_os = "linux"))]
            const H: i32 = 0;

            raylib::init()
            .size(W, H)
            .resizable()
            .title(WINDOW_TITLE)
            .build()
        };

        if cfg!(debug_assertions) {
            logging::set_trace_log(TraceLogLevel::LOG_WARNING);
        }

        rl.set_target_fps(60);
        rl.toggle_fullscreen();

        // We need a reference to this so we can use `draw_text_rec`
        let font = rl.get_font_default();

        let spritesheet_img = {
            let byte_count: i32 = SPRITESHEET_BYTES.len()
                .try_into()
                .expect("(2^31)-1 bytes ought to be enough for anybody!");

            let bytes = SPRITESHEET_BYTES.as_ptr();

            let file_type = b".png\0" as *const u8 as *const i8;

            unsafe {
                Image::from_raw(LoadImageFromMemory(
                    file_type,
                    bytes,
                    byte_count
                ))
            }
        };

        let spritesheet = rl.load_texture_from_image(
            &thread,
            &spritesheet_img
        ).expect(
            "Embedded spritesheet could not be loaded!"
        );

        // This call currently (sometimes?) produces warnings about not being able
        // to find shader attributes/uniforms. These warnings seem harmless at the
        // moment. I think the cause is that the unused parts are being optimized
        // out by the GPU when it interprets the shader, as mentioned here:
        // https://github.com/raysan5/raylib/issues/2211
        let grid_shader = rl.load_shader_from_memory(
            &thread,
            None,
            Some(SAMPLING_SHADER)
        );

        // This seems like a safe texture size, with wide GPU support.
        // TODO What we should do is query GL_MAX_TEXTURE_SIZE and figure
        // out what to do if we get a smaller value than this.
//        const RENDER_TARGET_SIZE: u32 = 8192;
        // On the other hand, 8192 makes my old intergrated graphics laptop overheat
        // Maybe it would be faster/less hot to avoiding clearing the whole thing
        // each frame?
        const RENDER_TARGET_SIZE: u32 = 2048;

        // We'll let the OS reclaim the memory when the game closes.
        let mut render_target = rl.load_render_texture(
            &thread,
            RENDER_TARGET_SIZE,
            RENDER_TARGET_SIZE
        ).unwrap();

        let seed: u128 = {
            use std::time::SystemTime;

            let duration = match
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
            {
                Ok(d) => d,
                Err(err) => err.duration(),
            };

            let _ = duration.as_nanos();

            1643795368538561068
        };
        println!("{}", seed);

        let mut state = game::State::from_seed(seed.to_le_bytes());
        let mut commands = Storage(Vec::with_capacity(1024));

        macro_rules! get_cursor_xy {
            () => {{
                let pos = rl.get_mouse_position();

                game::CursorXY {
                    x: pos.x,
                    y: pos.y,
                }
            }}
        }


        // generate the commands for the first frame
        game::update(
            &mut state,
            &mut commands,
            0,
            get_cursor_xy!(),
            draw_wh(&rl),
            rl.get_frame_time(),
        );

        const BACKGROUND: Color = Color{ r: 0x22, g: 0x22, b: 0x22, a: 255 };
        const WHITE: Color = Color{ r: 0xee, g: 0xee, b: 0xee, a: 255 };
        const STONE: Color = Color{ r: 0x5a, g: 0x7d, b: 0x8b, a: 255 };
        const POLE: Color = Color{ r: 0x33, g: 0x52, b: 0xe1, a: 255 };
        const FLAG: Color = Color{ r: 0xde, g: 0x49, b: 0x49, a: 255 };
        const ARROW: Color = Color{ r: 0x30, g: 0xb0, b: 0x6e, a: 255 };
        const TEXT: Color = WHITE;
        const NO_TINT: Color = WHITE;
        const OUTLINE: Color = WHITE;

        let mut show_stats = false;
        use std::time::Instant;
        struct TimeSpan {
            start: Instant,
            end: Instant,
        }

        impl Default for TimeSpan {
            fn default() -> Self {
                let start = Instant::now();
                Self {
                    start,
                    end: start,
                }
            }
        }

        impl std::fmt::Display for TimeSpan {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{: >6.3} ms",
                    (self.end - self.start).as_micros() as f32 / 1000.0
                )
            }
        }

        const STEP_KEY: KeyboardKey   = KEY_F8;
        const RESUME_KEY: KeyboardKey = KEY_F9;
        const STATS_KEY: KeyboardKey  = KEY_F10;

        enum StepState {
            Running,
            Paused,
            Stepping,
        }

        let mut step_state = if start_paused {
            StepState::Paused
        } else {
            StepState::Running
        };

        #[derive(Default)]
        struct FrameStats {
            loop_body: TimeSpan,
            input_gather: TimeSpan,
            update: TimeSpan,
            render: TimeSpan,
        }

        let mut prev_stats = FrameStats::default();

        while !rl.window_should_close() {
            let mut current_stats = FrameStats::default();
            current_stats.loop_body.start = Instant::now();
            current_stats.input_gather.start = current_stats.loop_body.start;

            if rl.is_key_pressed(STEP_KEY) {
                step_state = match step_state {
                    StepState::Running => StepState::Paused,
                    StepState::Paused
                    | StepState::Stepping => StepState::Stepping,
                };
            }

            if rl.is_key_pressed(RESUME_KEY) {
                step_state = StepState::Running;
            }

            if rl.is_key_pressed(STATS_KEY) {
                show_stats = !show_stats;
            }

            if rl.is_key_pressed(KEY_F11) {
                rl.toggle_fullscreen();
            }

            let mut input_flags = 0;

            if rl.is_key_pressed(KEY_SPACE) || rl.is_key_pressed(KEY_ENTER) {
                input_flags |= game::INPUT_INTERACT_PRESSED;
            }

            if rl.is_key_down(KEY_UP) || rl.is_key_down(KEY_W) {
                input_flags |= game::INPUT_UP_DOWN;
            }

            if rl.is_key_down(KEY_DOWN) || rl.is_key_down(KEY_S) {
                input_flags |= game::INPUT_DOWN_DOWN;
            }

            if rl.is_key_down(KEY_LEFT) || rl.is_key_down(KEY_A) {
                input_flags |= game::INPUT_LEFT_DOWN;
            }

            if rl.is_key_down(KEY_RIGHT) || rl.is_key_down(KEY_D) {
                input_flags |= game::INPUT_RIGHT_DOWN;
            }

            if rl.is_key_pressed(KEY_UP) || rl.is_key_pressed(KEY_W) {
                input_flags |= game::INPUT_UP_PRESSED;
            }

            if rl.is_key_pressed(KEY_DOWN) || rl.is_key_pressed(KEY_S) {
                input_flags |= game::INPUT_DOWN_PRESSED;
            }

            if rl.is_key_pressed(KEY_LEFT) || rl.is_key_pressed(KEY_A) {
                input_flags |= game::INPUT_LEFT_PRESSED;
            }

            if rl.is_key_pressed(KEY_RIGHT) || rl.is_key_pressed(KEY_D) {
                input_flags |= game::INPUT_RIGHT_PRESSED;
            }

            if rl.is_mouse_button_pressed(MOUSE_LEFT_BUTTON)
            || rl.is_mouse_button_released(MOUSE_LEFT_BUTTON) {
                input_flags |= game::INPUT_LEFT_MOUSE_CHANGED;
            }

            if rl.is_mouse_button_down(MOUSE_LEFT_BUTTON) {
                input_flags |= game::INPUT_LEFT_MOUSE_DOWN;
            }

            current_stats.input_gather.end = Instant::now();
            current_stats.update.start = current_stats.input_gather.end;

            let should_update = match step_state {
                StepState::Running => true,
                StepState::Paused => false,
                StepState::Stepping => {
                    step_state = StepState::Paused;
                    true
                },
            };

            if should_update {
                game::update(
                    &mut state,
                    &mut commands,
                    input_flags,
                    get_cursor_xy!(),
                    draw_wh(&rl),
                    rl.get_frame_time(),
                );
            }

            current_stats.update.end = Instant::now();
            current_stats.render.start = current_stats.update.end;

            let screen_render_rect = Rectangle {
                x: 0.,
                y: 0.,
                width: rl.get_screen_width() as _,
                height: rl.get_screen_height() as _
            };

            let sizes = game::sizes(&state);

            let mut d = rl.begin_drawing(&thread);

            d.clear_background(BACKGROUND);

            {
                let mut texture_d = d.begin_texture_mode(
                    &thread,
                    &mut render_target
                );

                let mut shader_d = texture_d.begin_shader_mode(
                    &grid_shader
                );

                shader_d.clear_background(BACKGROUND);

                // the -1 and +2 business makes the border lie just outside the actual
                // play area
                shader_d.draw_rectangle_lines(
                    sizes.play_xywh.x as i32 - 1,
                    sizes.play_xywh.y as i32 - 1,
                    sizes.play_xywh.w as i32 + 2,
                    sizes.play_xywh.h as i32 + 2,
                    OUTLINE
                );

                let tile_base_source_rect = Rectangle {
                    x: 0.,
                    y: 0.,
                    width: SPRITE_PIXELS_PER_TILE_SIDE,
                    height: SPRITE_PIXELS_PER_TILE_SIDE,
                };

                let tile_base_render_rect = Rectangle {
                    x: 0.,
                    y: 0.,
                    width: sizes.tile_side_length,
                    height: sizes.tile_side_length,
                };

                // I don't know why the texture lookup seems to be offset by these
                // amounts, but it seems to be.
                const X_SOURCE_FUDGE: f32 = -2.;
                const Y_SOURCE_FUDGE: f32 = -1.;

                for cmd in commands.0.iter() {
                    use game::draw::{DrawXY, Command::*};
                    macro_rules! convert_colour {
                        ($game_colour: expr) => {{
                            use game::draw::Colour::*;
                            match $game_colour {
                                Stone => STONE,
                                Pole => POLE,
                                Flag => FLAG,
                                Arrow => ARROW,
                            }
                        }}
                    }

                    match cmd {
                        Sprite(s) => {
                            let spec = source_spec(s.sprite);

                            let origin = Vector2 {
                                x: (tile_base_render_rect.width / 2.).round(),
                                y: (tile_base_render_rect.height / 2.).round(),
                            };

                            let render_rect = Rectangle {
                                x: s.xy.x + origin.x,
                                y: s.xy.y + origin.y,
                                ..tile_base_render_rect
                            };

                            let source_rect = Rectangle {
                                x: spec.x + X_SOURCE_FUDGE,
                                y: spec.y + Y_SOURCE_FUDGE,
                                ..tile_base_source_rect
                            };

                            shader_d.draw_texture_pro(
                                &spritesheet,
                                source_rect,
                                render_rect,
                                origin,
                                0.0, // Rotation
                                NO_TINT
                            );
                        }
                        Text(t) => {
                            use game::draw::TextKind;
                            match t.kind {
                                TextKind::UI => {
                                    shader_d.draw_text_rec(
                                        &font,
                                        &t.text,
                                        Rectangle {
                                            x: t.xy.x,
                                            y: t.xy.y,
                                            width: t.wh.w,
                                            height: t.wh.h,
                                        },
                                        // Constant arrived at through trial and error.
                                        sizes.draw_wh.w * (1./48.),
                                        1.,
                                        true, // word_wrap
                                        TEXT
                                    );
                                },
                            };
                        },
                        TriangleStrip(strip, colour) => {
                            let vectors: Vec<_> = strip
                                .iter()
                                .map(|DrawXY{x, y}| Vector2{x: *x, y: *y})
                                .collect();

                            shader_d.draw_triangle_strip(
                                &vectors,
                                convert_colour!(colour),
                            );
                        }
                        Line((xy1, xy2), colour) => {
                            shader_d.draw_line_ex(
                                Vector2 { x: xy1.x, y: xy1.y },
                                Vector2 { x: xy2.x, y: xy2.y },
                                256. / game::CAMERA_SCALE_FACTOR,
                                convert_colour!(colour),
                            );
                        }
                    }
                }

                if show_stats {
                    shader_d.draw_text_rec(
                        &font,
                        &format!(
                            "{:?}:step {:?}:resume {:?}:stats | loop {} input {} update {} render {}",
                            STEP_KEY,
                            RESUME_KEY,
                            STATS_KEY,
                            prev_stats.loop_body,
                            prev_stats.input_gather,
                            prev_stats.update,
                            prev_stats.render,
                        ),
                        Rectangle {
                            x: 0.,
                            y: 0.,
                            width: sizes.play_xywh.x,
                            height: sizes.play_xywh.h,
                        },
                        // Constant arrived at through trial and error.
                        sizes.draw_wh.w * (1./96.),
                        1.,
                        true, // word_wrap
                        TEXT
                    );
                }
            }

            let render_target_source_rect = Rectangle {
                x: 0.,
                y: (RENDER_TARGET_SIZE as f32) - screen_render_rect.height,
                width: screen_render_rect.width,
                // y flip for openGL
                height: -screen_render_rect.height
            };

            d.draw_texture_pro(
                &render_target,
                render_target_source_rect,
                screen_render_rect,
                Vector2::default(),
                0.0,
                NO_TINT
            );

            current_stats.render.end = Instant::now();
            current_stats.loop_body.end = current_stats.render.end;

            prev_stats = current_stats;
        }
    }

}
