#![deny(unused)]
#![deny(bindings_with_variant_name)]

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

use core::ops::Range;

#[allow(unused)]
fn xs_range(xs: &mut Xs, range: Range<u32>) -> u32 {
    xs_u32(xs, range.start, range.end)
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

/// This type alias makes adding a custom newtype easy.
pub type X = f32;
/// This type alias makes adding a custom newtype easy.
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
    SpriteKind,
    SpriteSpec,
    Sizes,
};

macro_rules! from_rng_enum_def {
    ($name: ident { $( $variants: ident ),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $( $variants ),+
        }

        impl $name {
            pub const COUNT: usize = {
                let mut count = 0;

                $(
                    // Some reference to the vars is needed to use
                    // the repetitions.
                    let _ = Self::$variants;

                    count += 1;
                )+

                count
            };

            pub const ALL: [Self; Self::COUNT] = [
                $(Self::$variants,)+
            ];

            pub fn from_rng(rng: &mut Xs) -> Self {
                Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
            }
        }
    }
}

from_rng_enum_def!{
    ArrowKind {
        Red,
        Green
    }
}

impl Default for ArrowKind {
    fn default() -> Self {
        Self::Red
    }
}

from_rng_enum_def!{
    Dir {
        Up,
        UpRight,
        Right,
        DownRight,
        Down,
        DownLeft,
        Left,
        UpLeft,
    }
}

impl Default for Dir {
    fn default() -> Self {
        Self::Up
    }
}

mod tile {
    use crate::{Xs, xs_u32};

    pub type Count = u32;

    pub type Coord = u8;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct X(Coord);

    impl X {
        pub const MAX: Coord = 0b1111;
        pub const COUNT: Count = (X::MAX as Count) + 1;

        pub fn from_rng(rng: &mut Xs) -> Self {
            Self(xs_u32(rng, 0, Self::COUNT) as Coord)
        }

        pub fn saturating_add_one(&self) -> Self {
            Self(core::cmp::min(self.0.saturating_add(1), Self::MAX))
        }

        pub fn saturating_sub_one(&self) -> Self {
            Self(self.0.saturating_sub(1))
        }
    }

    impl From<X> for Coord {
        fn from(X(c): X) -> Self {
            c
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Y(Coord);

    impl Y {
        pub const MAX: Coord = 0b1111;
        pub const COUNT: Count = (Y::MAX as Count) + 1;

        pub fn from_rng(rng: &mut Xs) -> Self {
            Self(xs_u32(rng, 0, Self::COUNT) as Coord)
        }

        pub fn saturating_add_one(&self) -> Self {
            Self(core::cmp::min(self.0.saturating_add(1), Self::MAX))
        }

        pub fn saturating_sub_one(&self) -> Self {
            Self(self.0.saturating_sub(1))
        }
    }

    impl From<Y> for Coord {
        fn from(Y(c): Y) -> Self {
            c
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct XY {
        pub x: X,
        pub y: Y,
    }

    impl XY {
        pub const COUNT: Count = X::COUNT * Y::COUNT;

        pub fn from_rng(rng: &mut Xs) -> Self {
            Self {
                x: X::from_rng(rng),
                y: Y::from_rng(rng),
            }
        }

        pub fn move_up(&mut self) {
            self.y = self.y.saturating_sub_one();
        }

        pub fn move_down(&mut self) {
            self.y = self.y.saturating_add_one();
        }

        pub fn move_left(&mut self) {
            self.x = self.x.saturating_sub_one();
        }

        pub fn move_right(&mut self) {
            self.x = self.x.saturating_add_one();
        }
    }

    #[allow(unused)]
    pub fn xy_to_i(xy: XY) -> usize {
        xy_to_i_usize((usize::from(xy.x.0), usize::from(xy.y.0)))
    }

    pub fn xy_to_i_usize((x, y): (usize, usize)) -> usize {
        y * Y::COUNT as usize + x
    }

    pub fn i_to_xy(index: usize) -> XY {
        XY {
            x: X(to_coord_or_default(
                (index % X::COUNT as usize) as Count
            )),
            y: Y(to_coord_or_default(
                ((index % (XY::COUNT as usize) as usize)
                / X::COUNT as usize) as Count
            )),
        }
    }

    fn to_coord_or_default(n: Count) -> Coord {
        core::convert::TryFrom::try_from(n).unwrap_or_default()
    }
}

fn draw_xy_from_tile(sizes: &Sizes, txy: tile::XY) -> DrawXY {
    DrawXY {
        x: sizes.board_xywh.x + sizes.board_xywh.w * (tile::Coord::from(txy.x) as DrawLength / tile::X::COUNT as DrawLength),
        y: sizes.board_xywh.y + sizes.board_xywh.h * (tile::Coord::from(txy.y) as DrawLength / tile::Y::COUNT as DrawLength),
    }
}

/// A Tile should always be at a particular position, but that position should be
/// derivable from the tiles location in the tiles array, so it doesn't need to be
/// stored. But, we often want to get the tile's data and it's location as a single
/// thing. This is why we have both `Tile` and `TileData`
#[derive(Copy, Clone, Debug, Default)]
struct TileData {
    dir: Dir,
    arrow_kind: ArrowKind,
}

impl TileData {
    fn from_rng(rng: &mut Xs) -> Self {
        Self {
            dir: Dir::from_rng(rng),
            arrow_kind: ArrowKind::from_rng(rng),
        }
    }

    fn sprite(&self) -> SpriteKind {
        SpriteKind::Arrow(self.dir, self.arrow_kind)
    }
}

pub const TILES_LENGTH: usize = tile::XY::COUNT as _;

type TileDataArray = [TileData; TILES_LENGTH as _];

#[derive(Clone, Debug)]
pub struct Tiles {
    tiles: TileDataArray,
}

impl Default for Tiles {
    fn default() -> Self {
        Self {
            tiles: [TileData::default(); TILES_LENGTH as _],
        }
    }
}

impl Tiles {
    fn from_rng(rng: &mut Xs) -> Self {
        let mut tiles = [TileData::default(); TILES_LENGTH as _];

        for tile_data in tiles.iter_mut() {
            *tile_data = TileData::from_rng(rng);
        }

        Self {
            tiles
        }
    }
}

#[derive(Debug)]
enum EyeState {
    Idle,
    Moved(Dir),
    NarrowAnimLeft,
    NarrowAnimCenter,
    NarrowAnimRight,
    SmallPupil,
    Closed,
    HalfLid,
}

impl Default for EyeState {
    fn default() -> Self {
        Self::Idle
    }
}

impl EyeState {
    fn sprite(&self) -> SpriteKind {
        use EyeState::*;
        match self {
            Idle => SpriteKind::NeutralEye,
            Moved(dir) => SpriteKind::DirEye(*dir),
            NarrowAnimLeft => SpriteKind::NarrowLeftEye,
            NarrowAnimCenter => SpriteKind::NarrowCenterEye,
            NarrowAnimRight => SpriteKind::NarrowRightEye,
            SmallPupil => SpriteKind::SmallPupilEye,
            Closed => SpriteKind::ClosedEye,
            HalfLid => SpriteKind::HalfLidEye,
        }
    }
}

#[derive(Debug, Default)]
struct Eye {
    xy: tile::XY,
    state: EyeState,
}

// Short for "zero-one", which seemed better than `_01`.
mod zo {
    /// Values outside the range [0, 1] are expected, but they are expected to be
    /// clipped later.
    pub type Zo = f32;

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct X(pub(crate) Zo);

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct Y(pub(crate) Zo);

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct XY {
        pub x: X,
        pub y: Y,
    }

    #[macro_export]
    macro_rules! zo_xy {
        ($x: expr, $y: expr $(,)?) => {
            XY {
                x: X(
                    $x,
                ),
                y: Y(
                    $y,
                ),
            }
        }
    }
}

type Triangles = Vec<zo::XY>;
type Edge = Vec<zo::XY>;

pub const TOP_Y: f32 = 1.0;
pub const BOTTOM_Y: f32 = 0.0;

fn push_with_floor_point(
    triangles: &mut Triangles,
    xy: zo::XY,
) {
    use zo::{XY, X, Y};

    triangles.push(zo_xy!(xy.x.0, BOTTOM_Y));
    triangles.push(xy);
}

fn push_floor_triangles_from_edge(
    triangles: &mut Triangles,
    edge: &[zo::XY],
) {
    if edge.is_empty() {
        return;
    }

    triangles.push(edge[0]);

    for &xy in &edge[1..] {
        push_with_floor_point(triangles, xy);
    }
}

#[test]
fn push_floor_triangles_from_edge_works_on_this_small_example() {
    use zo::{XY, X, Y};

    let mut triangles = Triangles::with_capacity(5);

    push_floor_triangles_from_edge(
        &mut triangles,
        &[zo_xy!(0., BOTTOM_Y), zo_xy!(0.55, 0.45)]
    );

    assert_eq!(
        triangles,
        vec![
            zo_xy!(0., BOTTOM_Y),
            zo_xy!(0.55, BOTTOM_Y),
            zo_xy!(0.55, 0.45),
        ]
    );
}

#[test]
fn push_floor_triangles_from_edge_works_on_the_initial_example() {
    use zo::{XY, X, Y};

    let mut triangles = Triangles::with_capacity(5);

    push_floor_triangles_from_edge(
        &mut triangles,
        &[zo_xy!(0., BOTTOM_Y), zo_xy!(0.55, 0.45), zo_xy!(1., 0.)]
    );

    assert_eq!(
        triangles,
        vec![
            zo_xy!(0., BOTTOM_Y),
            zo_xy!(0.55, BOTTOM_Y),
            zo_xy!(0.55, 0.45),
            zo_xy!(1., BOTTOM_Y),
            zo_xy!(1., 0.),
        ]
    );
}

fn push_spiky_edge_points(
    edge: &mut Edge,
    rng: &mut Xs,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    if count == 0 {
        return;
    }

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) });

    if count == 1 {
        return;
    }

    let (min_x, max_x) = if start.x.0 < end.x.0 {
        (start.x.0, end.x.0)
    } else {
        // NaN ends up here.
        (end.x.0, start.x.0)
    };

    let (min_y, max_y) = if start.y.0 < end.y.0 {
        (start.y.0, end.y.0)
    } else {
        // NaN ends up here.
        (end.y.0, start.y.0)
    };

    let x_delta = (end.x.0 - start.x.0) / count as f32;
    let y_delta = (end.y.0 - start.y.0) / count as f32;

    x_base += x_delta;
    y_base += y_delta;

    const SCALE: u32 = 65536;

    macro_rules! minus_one_to_one {
        () => {
            (xs_range(rng, 0..SCALE * 2) as f32 / SCALE as f32) - 1.
        }
    }

    // `y` can be spikier.
    const Y_SPIKE_FACTOR: f32 = 8.;

    for _ in 1..count {
        let mut x = x_base + x_delta * minus_one_to_one!();
        let mut y = y_base + y_delta * Y_SPIKE_FACTOR * minus_one_to_one!();

        if x < min_x { x = min_x; }
        if y < min_y { y = min_y; }

        if x > max_x { x = max_x; }
        if y > max_y { y = max_y; }

        edge.push(zo::XY{ x: zo::X(x), y: zo::Y(y) });

        x_base += x_delta;
        y_base += y_delta;
    }
}

fn push_evenly_spaced_edge_points(
    edge: &mut Edge,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    if count == 0 {
        return;
    }

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) });

    if count == 1 {
        return;
    }

    let max_x = if start.x.0 < end.x.0 {
        end.x.0
    } else {
        // NaN ends up here.
        start.x.0
    };

    let max_y = if start.y.0 < end.y.0 {
        end.y.0
    } else {
        // NaN ends up here.
        start.y.0
    };

    let x_delta = (end.x.0 - start.x.0) / count as f32;
    let y_delta = (end.y.0 - start.y.0) / count as f32;

    x_base += x_delta;
    y_base += y_delta;

    for _ in 1..count {
        let mut x = x_base + x_delta;
        let mut y = y_base + y_delta;

        if x > max_x { x = max_x; }
        if y > max_y { y = max_y; }

        edge.push(zo::XY{ x: zo::X(x), y: zo::Y(y) });

        x_base += x_delta;
        y_base += y_delta;
    }
}

fn push_simplest_overhang_edge_points(
    edge: &mut Edge,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    if count == 0 {
        return;
    }

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    let start_len = edge.len();

    edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) });

    if count == 1 {
        return;
    }

    // TODO refactor to pass around triangles vec instead, so we can get a real
    // overhang

    if count > 4 {
        let x_delta = (end.x.0 - start.x.0) / count as f32;
        let y_delta = (end.y.0 - start.y.0) / count as f32;

        x_base += x_delta;
        y_base += y_delta;

        edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(BOTTOM_Y) });
        edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) });

        x_base += x_delta;
        y_base += y_delta;

        edge.push(zo::XY{ x: zo::X(x_base), y: zo::Y(BOTTOM_Y) });
        edge.push(zo::XY{ x: zo::X(x_base - x_delta * 2.), y: zo::Y(y_base) });
    }

    push_evenly_spaced_edge_points(
        edge,
        zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) }..end,
        count - (edge.len() - start_len),
    )
}

#[derive(Debug, Default)]
struct Board {
    tiles: Tiles,
    eye: Eye,
    triangles: Triangles,
    summit: zo::XY,
}

impl Board {
    fn from_seed(seed: Seed) -> Self {
        let mut rng = xs_from_seed(seed);

        let tiles = Tiles::from_rng(&mut rng);

        let triangle_count = 128;
        let overall_count = 2 + triangle_count;

        let mut triangles = Vec::with_capacity(overall_count as usize);

        use zo::{XY, X, Y};

        #[cfg(any())] {
            triangles.push(zo_xy!(0., 1.));
            triangles.push(zo_xy!(1., 1.));
            triangles.push(zo_xy!(1., 0.));
        }

        let edge_count = (triangle_count / 2) + 1;

        let mut edge = Vec::with_capacity(edge_count as usize);

        let supposed_summit = zo_xy!(
            0.5,
            0.625,
        );

        assert!(edge_count >= 2);
        let per_slope = edge_count / 2;

        dbg!(per_slope);

        /*push_spiky_edge_points(
            &mut edge,
            &mut rng,
            zo_xy!(0., 0.)..supposed_summit,
            per_slope,
        );*/

        push_simplest_overhang_edge_points(
            &mut edge,
            zo_xy!(0., 0.)..supposed_summit,
            per_slope,
        );

        let mut max_x = 0.;
        let mut max_y = 0.;

        for xy in &edge {
            let x = xy.x.0;
            let y = xy.y.0;

            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }

        let summit = zo_xy!(
            max_x,
            // The summit must be the highest point
            max_y + 0.0015,
        );

        push_spiky_edge_points(
            &mut edge,
            &mut rng,
            summit..zo_xy!(1., 0.),
            per_slope,
        );
        edge.push(zo_xy!(1., 0.));

        // TODO consider fixing up the summit after the second half is generated.

        // TODO make a function that deterministically produces the simplest
        // overhang.

        // TODO make a function that generates the a random variation on the
        // simplest overhang.

        // TODO make a function that deterministically produces a spiral overhang.

        dbg!(&edge);

        assert_eq!(edge.len(), edge_count);

        push_floor_triangles_from_edge(
            &mut triangles,
            &edge
        );

        Self {
            tiles,
            eye: Eye {
                xy: tile::XY::from_rng(&mut rng),
                ..<_>::default()
            },
            triangles,
            summit,
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
    Dir(Dir),
    Interact,
}

impl Input {
    fn from_flags(flags: InputFlags) -> Self {
        use Input::*;
        use crate::Dir::*;
        if INPUT_INTERACT_PRESSED & flags != 0 {
            Interact
        } else if (INPUT_UP_DOWN | INPUT_RIGHT_DOWN) & flags == (INPUT_UP_DOWN | INPUT_RIGHT_DOWN) {
            Dir(UpRight)
        } else if (INPUT_DOWN_DOWN | INPUT_RIGHT_DOWN) & flags == (INPUT_DOWN_DOWN | INPUT_RIGHT_DOWN) {
            Dir(DownRight)
        } else if (INPUT_DOWN_DOWN | INPUT_LEFT_DOWN) & flags == (INPUT_DOWN_DOWN | INPUT_LEFT_DOWN) {
            Dir(DownLeft)
        } else if (INPUT_UP_DOWN | INPUT_LEFT_DOWN) & flags == (INPUT_UP_DOWN | INPUT_LEFT_DOWN) {
            Dir(UpRight)
        } else if INPUT_UP_DOWN & flags != 0 {
            Dir(Up)
        } else if INPUT_DOWN_DOWN & flags != 0 {
            Dir(Down)
        } else if INPUT_LEFT_DOWN & flags != 0 {
            Dir(Left)
        } else if INPUT_RIGHT_DOWN & flags != 0 {
            Dir(Right)
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

    use EyeState::*;
    use Input::*;
    use crate::Dir::*;

    const HOLD_FRAMES: AnimationTimer = 30;

    match input {
        NoChange => match state.board.eye.state {
            Idle => {
                if state.animation_timer % (HOLD_FRAMES * 3) == 0 {
                    state.board.eye.state = NarrowAnimCenter;
                }
            },
            Moved(_) => {
                if state.animation_timer % HOLD_FRAMES == 0 {
                    state.board.eye.state = Idle;
                }
            },
            SmallPupil => {
                if state.animation_timer % (HOLD_FRAMES * 3) == 0 {
                    state.board.eye.state = Closed;
                }
            },
            Closed => {
                if state.animation_timer % (HOLD_FRAMES) == 0 {
                    state.board.eye.state = HalfLid;
                }
            },
            HalfLid => {
                if state.animation_timer % (HOLD_FRAMES * 5) == 0 {
                    state.board.eye.state = Idle;
                }
            },
            NarrowAnimCenter => {
                let modulus = state.animation_timer % (HOLD_FRAMES * 4);
                if modulus == 0 {
                    state.board.eye.state = NarrowAnimRight;
                } else if modulus == HOLD_FRAMES * 2 {
                    state.board.eye.state = NarrowAnimLeft;
                }
            },
            NarrowAnimLeft | NarrowAnimRight => {
                if state.animation_timer % HOLD_FRAMES == 0 {
                    state.board.eye.state = NarrowAnimCenter;
                }
            },
        },
        Dir(Up) => {
            state.board.eye.state = Moved(Up);
            state.board.eye.xy.move_up();
        },
        Dir(UpRight) => {
            state.board.eye.state = Moved(UpRight);
            state.board.eye.xy.move_up();
            state.board.eye.xy.move_right();
        },
        Dir(Right) => {
            state.board.eye.state = Moved(Right);
            state.board.eye.xy.move_right();
        },
        Dir(DownRight) => {
            state.board.eye.state = Moved(DownRight);
            state.board.eye.xy.move_down();
            state.board.eye.xy.move_right();
        },
        Dir(Down) => {
            state.board.eye.state = Moved(Down);
            state.board.eye.xy.move_down();
        },
        Dir(DownLeft) => {
            state.board.eye.state = Moved(DownLeft);
            state.board.eye.xy.move_down();
            state.board.eye.xy.move_left();
        },
        Dir(Left) => {
            state.board.eye.state = Moved(Left);
            state.board.eye.xy.x = state.board.eye.xy.x.saturating_sub_one();
        },
        Dir(UpLeft) => {
            state.board.eye.state = Moved(UpLeft);
            state.board.eye.xy.move_up();
            state.board.eye.xy.move_left();
        },
        Interact => {
            state.board.eye.state = SmallPupil;
        },
    }

    for i in 0..TILES_LENGTH {
        let tile_data = state.board.tiles.tiles[i];

        let txy = tile::i_to_xy(i);

        commands.push(Sprite(SpriteSpec{
            sprite: tile_data.sprite(),
            xy: draw_xy_from_tile(&state.sizes, txy),
        }));
    }

    commands.push(Sprite(SpriteSpec{
        sprite: state.board.eye.state.sprite(),
        xy: draw_xy_from_tile(&state.sizes, state.board.eye.xy),
    }));

    fn zo_to_draw_xy(sizes: &Sizes, xy: zo::XY) -> DrawXY {
        DrawXY {
            x: sizes.board_xywh.x + sizes.board_xywh.w * xy.x.0,
            y: sizes.board_xywh.y + sizes.board_xywh.h * (TOP_Y - xy.y.0),
        }
    }

    // TODO avoid this per-frame allocation or merge it with others.
    let mountain: draw::TriangleStrip = state.board.triangles
        .iter()
        .map(|xy| zo_to_draw_xy(&state.sizes, *xy))
        .collect();

    commands.push(TriangleStrip(mountain, draw::Colour::Stone));

    let summit_xy = state.board.summit;

    const POLE_HALF_W: f32 = 1./1024.;
    const POLE_H: f32 = POLE_HALF_W * 32.;//8.;

    let pole_top_y = summit_xy.y.0 + POLE_H;
    let pole_min_x = summit_xy.x.0 - POLE_HALF_W;
    let pole_max_x = summit_xy.x.0 + POLE_HALF_W;

    use zo::{X, Y, XY};

    // TODO avoid this per-frame allocation or merge it with others.
    let pole = vec![
        zo_xy!{ pole_min_x, summit_xy.y.0 },
        zo_xy!{ pole_max_x, summit_xy.y.0 },
        zo_xy!{ pole_min_x, pole_top_y },
        zo_xy!{ pole_max_x, pole_top_y },
    ]
        .into_iter()
        .map(|xy| zo_to_draw_xy(&state.sizes, xy))
        .collect();

    commands.push(TriangleStrip(pole, draw::Colour::Pole));

    const FLAG_H: f32 = POLE_H / 4.;
    const FLAG_W: f32 = FLAG_H;

    // TODO Animate the flag blowing in the wind.

    let flag = vec![
        zo_xy!{ pole_max_x, pole_top_y },
        zo_xy!{ pole_max_x, pole_top_y - FLAG_H },
        zo_xy!{ pole_max_x + FLAG_W, pole_top_y - FLAG_H / 2. },
    ]
        .into_iter()
        .map(|xy| zo_to_draw_xy(&state.sizes, xy))
        .collect();

    commands.push(TriangleStrip(flag, draw::Colour::Flag));

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
                w: state.sizes.play_xywh.w,
                h: small_section_h
            },
            kind: TextKind::UI,
        }));

        y += small_section_h;

        commands.push(Text(TextSpec{
            text: format!(
                "sizes: {:?}\nanimation_timer: {:?}",
                state.sizes,
                state.animation_timer
            ),
            xy: DrawXY { x: left_text_x, y },
            wh: DrawWH {
                w: state.sizes.play_xywh.w,
                h: state.sizes.play_xywh.h - y
            },
            kind: TextKind::UI,
        }));
    }

    state.animation_timer += 1;
    if state.animation_timer >= ANIMATION_TIMER_LENGTH {
        state.animation_timer = 0;
    }
}
