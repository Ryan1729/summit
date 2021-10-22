#![deny(unused)]

#[allow(unused)]
macro_rules! compile_time_assert {
    ($assertion: expr) => {{
        #[allow(unknown_lints, eq_op)]
        // Based on the const_assert macro from static_assertions;
        const _: [(); 0 - !{$assertion} as usize] = [];
    }}
}

// In case we decide that we care about no_std/not directly allocating ourself
pub trait ClearableStorage<A> {
    fn clear(&mut self);

    fn push(&mut self, a: A);
}

pub type Seed = [u8; 16];

type Xs = [core::num::Wrapping<u32>; 4];

fn xorshift(xs: &mut Xs) -> u32 {
    let mut t = xs[3];

    xs[3] = xs[2];
    xs[2] = xs[1];
    xs[1] = xs[0];

    t ^= t << 11;
    t ^= t >> 8;
    xs[0] = t ^ xs[0] ^ (xs[0] >> 19);

    xs[0].0
}

#[allow(unused)]
fn xs_u32(xs: &mut Xs, min: u32, one_past_max: u32) -> u32 {
    (xorshift(xs) % (one_past_max - min)) + min
}

#[allow(unused)]
fn xs_shuffle<A>(rng: &mut Xs, slice: &mut [A]) {
    for i in 1..slice.len() as u32 {
        // This only shuffles the first u32::MAX_VALUE - 1 elements.
        let r = xs_u32(rng, 0, i + 1) as usize;
        let i = i as usize;
        slice.swap(i, r);
    }
}

#[allow(unused)]
fn new_seed(rng: &mut Xs) -> Seed {
    let s0 = xorshift(rng).to_le_bytes();
    let s1 = xorshift(rng).to_le_bytes();
    let s2 = xorshift(rng).to_le_bytes();
    let s3 = xorshift(rng).to_le_bytes();

    [
        s0[0], s0[1], s0[2], s0[3],
        s1[0], s1[1], s1[2], s1[3],
        s2[0], s2[1], s2[2], s2[3],
        s3[0], s3[1], s3[2], s3[3],
    ]
}

fn xs_from_seed(mut seed: Seed) -> Xs {
    // 0 doesn't work as a seed, so use this one instead.
    if seed == [0; 16] {
        seed = 0xBAD_5EED_u128.to_le_bytes();
    }

    macro_rules! wrap {
        ($i0: literal, $i1: literal, $i2: literal, $i3: literal) => {
            core::num::Wrapping(
                u32::from_le_bytes([
                    seed[$i0],
                    seed[$i1],
                    seed[$i2],
                    seed[$i3],
                ])
            )
        }
    }

    [
        wrap!( 0,  1,  2,  3),
        wrap!( 4,  5,  6,  7),
        wrap!( 8,  9, 10, 11),
        wrap!(12, 13, 14, 15),
    ]
}

/// These type aliases make adding a custom newtype easy.
pub type X = f32;
pub type Y = f32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XY {
    pub x: X,
    pub y: Y,
}

pub mod draw;

pub use draw::{
    DrawLength,
    DrawX,
    DrawY, 
    DrawXY,
    DrawW,
    DrawH,
    DrawWH,
    SpriteKind
};

#[derive(Copy, Clone, Debug)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right
}

#[derive(Debug, Default)]
struct Board {
    rng: Xs,
}

impl Board {
    fn from_seed(seed: Seed) -> Self {
        let rng = xs_from_seed(seed);

        Self {
            rng,
            ..<_>::default()
        }
    }
}

/// 64k animation frames ought to be enough for anybody!
type AnimationTimer = u16;

/// We use this because it has a lot more varied factors than 65536.
const ANIMATION_TIMER_LENGTH: AnimationTimer = 60 * 60 * 18;

#[derive(Debug, Default)]
pub struct State {
    sizes: draw::Sizes,
    board: Board,
    animation_timer: AnimationTimer
}

impl State {
    pub fn from_seed(seed: Seed) -> Self {
        Self {
            board: Board::from_seed(seed),
            ..<_>::default()
        }
    }
}

pub fn sizes(state: &State) -> draw::Sizes {
    state.sizes.clone()
}

pub type InputFlags = u16;

pub const INPUT_UP_PRESSED: InputFlags              = 0b0000_0000_0000_0001;
pub const INPUT_DOWN_PRESSED: InputFlags            = 0b0000_0000_0000_0010;
pub const INPUT_LEFT_PRESSED: InputFlags            = 0b0000_0000_0000_0100;
pub const INPUT_RIGHT_PRESSED: InputFlags           = 0b0000_0000_0000_1000;

pub const INPUT_UP_DOWN: InputFlags                 = 0b0000_0000_0001_0000;
pub const INPUT_DOWN_DOWN: InputFlags               = 0b0000_0000_0010_0000;
pub const INPUT_LEFT_DOWN: InputFlags               = 0b0000_0000_0100_0000;
pub const INPUT_RIGHT_DOWN: InputFlags              = 0b0000_0000_1000_0000;

pub const INPUT_INTERACT_PRESSED: InputFlags        = 0b0000_0001_0000_0000;

#[derive(Clone, Copy, Debug)]
enum Input {
    NoChange,
    Up,
    Down,
    Left,
    Right,
    Interact,
}

impl Input {
    fn from_flags(flags: InputFlags) -> Self {
        use Input::*;
        if INPUT_INTERACT_PRESSED & flags != 0 {
            Interact
        } else if INPUT_UP_DOWN & flags != 0 {
            Up
        } else if INPUT_DOWN_DOWN & flags != 0 {
            Down
        } else if INPUT_LEFT_DOWN & flags != 0 {
            Left
        } else if INPUT_RIGHT_DOWN & flags != 0 {
            Right
        } else {
            NoChange
        }
    }
}

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<draw::Command>,
    input_flags: InputFlags,
    draw_wh: DrawWH,
) {
    use draw::{TextSpec, TextKind, Command::*};

    if draw_wh != state.sizes.draw_wh {
        state.sizes = draw::fresh_sizes(draw_wh);
    }

    commands.clear();

    let input = Input::from_flags(input_flags);

    let board_xywh = &state.sizes.board_xywh;
    let left_text_x = state.sizes.play_xywh.x + MARGIN;

    const MARGIN: f32 = 16.;

    let small_section_h = state.sizes.draw_wh.h / 8. - MARGIN;

    {
        let mut y = MARGIN;

        commands.push(Text(TextSpec{
            text: format!(
                "input: {:?}",
                input
            ),
            xy: DrawXY { x: left_text_x, y },
            wh: DrawWH {
                w: board_xywh.x - left_text_x,
                h: small_section_h
            },
            kind: TextKind::UI,
        }));

        y += small_section_h;

        commands.push(Text(TextSpec{
            text: format!(
                "{:?}",
                state,
            ),
            xy: DrawXY { x: left_text_x, y },
            wh: DrawWH {
                w: board_xywh.x - left_text_x,
                h: small_section_h
            },
            kind: TextKind::UI,
        }));
    }

    state.animation_timer += 1;
    if state.animation_timer >= ANIMATION_TIMER_LENGTH {
        state.animation_timer = 0;
    }
}
