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

const XS_SCALE: u32 = 1 << f32::MANTISSA_DIGITS;

#[allow(unused)]
fn xs_zero_to_one(xs: &mut Xs) -> f32 {
    (xs_u32(xs, 0, XS_SCALE + 1) as f32 / XS_SCALE as f32) - 1.
}

#[allow(unused)]
fn xs_minus_one_to_one(xs: &mut Xs) -> f32 {
    (xs_u32(xs, 0, (XS_SCALE * 2) + 1) as f32 / XS_SCALE as f32) - 1.
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

    #[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
    pub struct X(pub(crate) Zo);

    #[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
    pub struct Y(pub(crate) Zo);

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct XY {
        pub x: X,
        pub y: Y,
    }

    use core::fmt;

    impl fmt::Display for XY {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "({}, {})", self.x.0, self.y.0)
        }
    }

    #[macro_export]
    macro_rules! zo_xy {
        ($x: expr, $y: expr $(,)?) => {
            zo::XY {
                x: zo::X(
                    $x,
                ),
                y: zo::Y(
                    $y,
                ),
            }
        }
    }

    pub fn minimums((a, b): (XY, XY)) -> (X, Y) {
        let min_x = if a.x < b.x {
            a.x
        } else {
            // NaN ends up here.
            b.x
        };

        let min_y = if a.y < b.y {
            a.y
        } else {
            // NaN ends up here.
            b.y
        };

        (min_x, min_y)
    }

    pub fn maximums((a, b): (XY, XY)) -> (X, Y) {
        let max_x = if a.x < b.x {
            b.x
        } else {
            // NaN ends up here.
            a.x
        };

        let max_y = if a.y < b.y {
            b.y
        } else {
            // NaN ends up here.
            a.y
        };

        (max_x, max_y)
    }
}

fn zo_to_draw_xy(sizes: &Sizes, xy: zo::XY) -> DrawXY {
    DrawXY {
        x: sizes.board_xywh.x + sizes.board_xywh.w * xy.x.0,
        y: sizes.board_xywh.y + (sizes.board_xywh.h * TOP_Y - sizes.board_xywh.h * xy.y.0),
    }
}

fn draw_to_zo_xy(sizes: &Sizes, xy: DrawXY) -> zo::XY {
    zo_xy!{
        (xy.x - sizes.board_xywh.x) / sizes.board_xywh.w,
        TOP_Y - ((xy.y - sizes.board_xywh.y) / sizes.board_xywh.h),
    }
}

#[cfg(test)]
const ACCEPTABLE_EPSILON: f32 = f32::EPSILON;

#[test]
fn zo_to_draw_to_zo_round_trips_on_these_examples() {
    let sizes = draw::fresh_sizes(draw::EXAMPLE_WH);

    // Short for assert.
    macro_rules! a {
        ($x: expr, $y: expr) => {
            let example = zo_xy!{ $x, $y };
            let round_tripped = draw_to_zo_xy(&sizes, zo_to_draw_xy(&sizes, example));

            assert!((round_tripped.x.0 - example.x.0).abs() <= ACCEPTABLE_EPSILON, "{round_tripped} !~= {example} (x)");
            assert!((round_tripped.y.0 - example.y.0).abs() <= ACCEPTABLE_EPSILON, "{round_tripped} !~= {example} (y)");
        }
    }

    a!(0., 0.);
    a!(0.5, 0.5);
    a!(1., 0.);
    a!(0., 1.);

    a!(-0., 0.);
    a!(-0.5, 0.5);
    a!(-1., 0.);
    a!(-0., 1.);

    a!(0., -0.);
    a!(0.5, -0.5);
    a!(1., -0.);
    a!(0., -1.);

    a!(-0., -0.);
    a!(-0.5, -0.5);
    a!(-1., -0.);
    a!(-0., -1.);
}

#[test]
fn draw_to_zo_to_draw_round_trips_on_these_examples() {
    let sizes = draw::fresh_sizes(draw::EXAMPLE_WH);

    // Short for assert.
    macro_rules! a {
        ($x: expr, $y: expr) => {
            let example = DrawXY{ x: $x * draw::EXAMPLE_WH.w, y: $y * draw::EXAMPLE_WH.h };
            let round_tripped = zo_to_draw_xy(&sizes, draw_to_zo_xy(&sizes, example));

            assert!((round_tripped.x - example.x).abs() <= ACCEPTABLE_EPSILON, "{round_tripped} !~= {example} (x)");
            assert!((round_tripped.y - example.y).abs() <= ACCEPTABLE_EPSILON, "{round_tripped} !~= {example} (y)");
        }
    }

    a!(0., 0.);
    a!(0.5, 0.5);
    a!(1., 0.);
    a!(0., 1.);

    a!(-0., 0.);
    a!(-0.5, 0.5);
    a!(-1., 0.);
    a!(-0., 1.);

    a!(0., -0.);
    a!(0.5, -0.5);
    a!(1., -0.);
    a!(0., -1.);

    a!(-0., -0.);
    a!(-0.5, -0.5);
    a!(-1., -0.);
    a!(-0., -1.);
}

type Triangles = Vec<zo::XY>;

pub const TOP_Y: f32 = 1.0;
pub const BOTTOM_Y: f32 = 0.0;

fn push_with_floor_point(
    triangles: &mut Triangles,
    xy: zo::XY,
) {
    triangles.push(zo_xy!(xy.x.0, BOTTOM_Y));
    triangles.push(xy);
}

type MountainSection = fn (
    triangles: &mut Triangles,
    rng: &mut Xs,
    range: Range<zo::XY>,
    count: usize
);

fn push_spiky_triangles_with_floor_points(
    triangles: &mut Triangles,
    rng: &mut Xs,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    if count == 0 {
        return;
    }

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    if triangles.is_empty() {
        triangles.push(zo_xy!{x_base, y_base});

        if count == 1 {
            return;
        }
    }

    let (min_x, min_y) = zo::minimums((start, end));
    let (min_x, min_y) = (min_x.0, min_y.0);
    let (max_x, max_y) = zo::maximums((start, end));
    let (max_x, max_y) = (max_x.0, max_y.0);

    let x_delta = (end.x.0 - start.x.0) / count as f32;
    let y_delta = (end.y.0 - start.y.0) / count as f32;

    x_base += x_delta;
    y_base += y_delta;

    // `y` can be spikier.
    const Y_SPIKE_FACTOR: f32 = 8.;

    for _ in 1..count {
        let mut x = x_base + x_delta * xs_minus_one_to_one(rng);
        let mut y = y_base + y_delta * Y_SPIKE_FACTOR * xs_minus_one_to_one(rng);

        if x < min_x { x = min_x; }
        if y < min_y { y = min_y; }

        if x > max_x { x = max_x; }
        if y > max_y { y = max_y; }

        push_with_floor_point(triangles, zo_xy!{x, y});

        x_base += x_delta;
        y_base += y_delta;
    }
}

/// Jump discontinuously to another point using degenerate (zero area) triangles.
fn push_dengenerate_to(
    triangles: &mut Triangles,
    (next1, next2): (zo::XY, zo::XY),
) {
    let len = triangles.len();

    match (triangles.get(len - 2), triangles.get(len - 1)) {
        (None, None) // Don't need extra triangle
        | (Some(_), None) // Should be impossible
        => {},
        // An unusual/unexpected case we nevertheless try to handle
        (None, Some(_prev)) => {
            triangles.extend_from_slice(&[
                // Make a degenerate triangle with this weird single point
                next1,
                next1,
                // Next tri but degenerate
                next1,
                next2,
                next2,
            ]);
        },
        // The expected usual case
        (Some(&prev2), Some(&prev1)) => {
            triangles.extend_from_slice(&[
                // Last tri but degenerate
                prev2,
                prev1,
                prev1,
                // Make the jump we want to make
                prev1,
                next1,
                next1,
                // Next tri but degenerate
                next1,
                next2,
                next2,
            ]);
        },
    }
}

fn push_evenly_spaced_triangles(
    triangles: &mut Triangles,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    if count == 0 {
        return;
    }

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    if triangles.is_empty() {
        triangles.push(zo_xy!{x_base, y_base});

        if count == 1 {
            return;
        }
    }

    let (max_x, max_y) = zo::maximums((start, end));
    let (max_x, max_y) = (max_x.0, max_y.0);

    let x_delta = (end.x.0 - start.x.0) / count as f32;
    let y_delta = (end.y.0 - start.y.0) / count as f32;

    x_base += x_delta;
    y_base += y_delta;

    for _ in 1..count {
        let mut x = x_base + x_delta;
        let mut y = y_base + y_delta;

        if x > max_x { x = max_x; }
        if y > max_y { y = max_y; }

        push_with_floor_point(triangles, zo_xy!{x, y});

        x_base += x_delta;
        y_base += y_delta;
    }
}

fn push_evenly_spaced_triangles_section(
    triangles: &mut Triangles,
    _: &mut Xs,
    range: Range<zo::XY>,
    count: usize,
) {
    push_evenly_spaced_triangles(triangles, range, count)
}

fn push_overhang_triangles(
    triangles: &mut Triangles,
    rng: &mut Xs,
    Range { start, end }: Range<zo::XY>,
    count: usize
) {
    let mut remaining_count = count;

    macro_rules! early_out {
        () => { if remaining_count == 0 { return; } }
    }

    early_out!();

    let mut x_base = start.x.0;
    let mut y_base = start.y.0;

    let start_len = triangles.len();

    if start_len == 0 {
        triangles.push(zo::XY{ x: zo::X(x_base), y: zo::Y(y_base) });

        remaining_count -= 1;

        early_out!();
    }

    let (min_x, min_y) = zo::minimums((start, end));
    let (min_x, min_y) = (min_x.0, min_y.0);
    let (max_x, max_y) = zo::maximums((start, end));
    let (max_x, max_y) = (max_x.0, max_y.0);

    const PER_OVERHANG: usize = 4;
    const POINTS_PER_DELTA: f32 = 2.;

    let x_delta = (end.x.0 - start.x.0) / (count as f32 / POINTS_PER_DELTA);
    let y_delta = (end.y.0 - start.y.0) / (count as f32 / POINTS_PER_DELTA);

    while remaining_count >= PER_OVERHANG {
        macro_rules! gen_xy {
            () => {{
                let mut x = x_base + x_delta * xs_minus_one_to_one(rng);
                let mut y = y_base + y_delta * xs_minus_one_to_one(rng);

                if x < min_x { x = min_x; }
                if y < min_y { y = min_y; }

                if x > max_x { x = max_x; }
                if y > max_y { y = max_y; }

                (x, y)
            }}
        }

        x_base += x_delta;
        y_base += y_delta;

        let (x, y) = gen_xy!();

        push_with_floor_point(triangles, zo_xy!{x, y});

        x_base += x_delta;
        y_base += y_delta;

        let (x, y) = gen_xy!();

        // This is similar, but not identical to, an inlined version of
        // the `push_with_floor_point` procedure.
        triangles.push(zo::XY{ x: zo::X(x), y: zo::Y(BOTTOM_Y) });
        let mut overhang_x = x - x_delta * 2.;
        if overhang_x < min_x { overhang_x = min_x; }
        triangles.push(zo::XY{ x: zo::X(overhang_x), y: zo::Y(y) });

        remaining_count -= PER_OVERHANG;
    }

    early_out!();

    push_evenly_spaced_triangles(
        triangles,
        zo_xy!{ x_base, y_base }..end,
        count - (triangles.len() - start_len),
    )
}

type Radians = f32;

const PI: Radians = core::f32::consts::PI;

#[derive(Debug, Default)]
struct Player {
    xy: zo::XY,
    angle: Radians,
}

#[derive(Debug, Default)]
struct Board {
    tiles: Tiles,
    eye: Eye,
    triangles: Triangles,
    summit: zo::XY,
    player: Player,
}

impl Board {
    fn from_seed(seed: Seed) -> Self {
        let mut rng = xs_from_seed(seed);

        let tiles = Tiles::from_rng(&mut rng);

        let triangle_count = 254;
        let overall_count = 2 + triangle_count;

        let mut triangles = Vec::with_capacity(overall_count as usize);

        #[cfg(any())] {
            triangles.push(zo_xy!(0., 1.));
            triangles.push(zo_xy!(1., 1.));
            triangles.push(zo_xy!(1., 0.));
        }

        let edge_count = (triangle_count / 2) + 1;

        let supposed_summit = zo_xy!(
            0.5,
            0.625,
        );

        assert!(edge_count >= 2);
        let per_slope = edge_count / 2;

        const SECTIONS: [MountainSection; 4] = [
            push_evenly_spaced_triangles_section,
            push_spiky_triangles_with_floor_points,
            push_overhang_triangles,

            // TODO make a function that deterministically produces a spiral overhang.

            // TODO does including this in here make a meaningful differnce?
            random_sections_across,
        ];

        const INITIAL_POINT: zo::XY = zo_xy!(0., 0.);
        const FINAL_POINT: zo::XY = zo_xy!(1., 0.);

        fn random_sections_across(
            triangles: &mut Triangles,
            rng: &mut Xs,
            Range { start, end }: Range<zo::XY>,
            count: usize,
        ) {
            let (min_x, min_y) = zo::minimums((start, end));
            let (min_x, min_y) = (min_x.0, min_y.0);

            let (max_x, max_y) = zo::maximums((start, end));
            let (max_x, max_y) = (max_x.0, max_y.0);

            let x_delta = (end.x.0 - start.x.0) / count as f32;
            let y_delta = (end.y.0 - start.y.0) / count as f32;

            let mut previous_end_point = start;

            let mut remaining = count;
            loop {
                let mut count_for_this_section = xs_range(rng, 0..8) as usize;

                if remaining < count_for_this_section {
                    count_for_this_section = remaining;
                }

                remaining -= count_for_this_section;

                compile_time_assert!(SECTIONS.len() <= u32::MAX as usize);

                let section = SECTIONS[xs_range(rng, 0..SECTIONS.len() as u32) as usize];

                let mut x = previous_end_point.x.0 + x_delta * count_for_this_section as f32;
                let mut y = previous_end_point.y.0 + y_delta * count_for_this_section as f32;

                if x < min_x { x = min_x; }
                if y < min_y { y = min_y; }

                if x > max_x { x = max_x; }
                if y > max_y { y = max_y; }

                let next_point = zo_xy!(x, y);

                section(
                    triangles,
                    rng,
                    previous_end_point..next_point,
                    count_for_this_section
                );

                if remaining == 0 { break }

                previous_end_point = next_point;
            }
        }

        random_sections_across(
            &mut triangles,
            &mut rng,
            INITIAL_POINT..supposed_summit,
            per_slope,
        );

        random_sections_across(
            &mut triangles,
            &mut rng,
            supposed_summit..FINAL_POINT,
            per_slope,
        );
        triangles.push(FINAL_POINT);

        let mut max_y = 0.;
        let mut summit_index = 0;

        for (i, xy) in triangles.iter().enumerate() {
            let y = xy.y.0;

            if y > max_y {
                max_y = y;
                summit_index = i;
            }
        }

        let mut summit = triangles[summit_index];

        // The summit must be the highest point
        summit.y.0 = max_y + 0.0015;

        triangles[summit_index] = summit;

        Self {
            tiles,
            eye: Eye {
                xy: tile::XY::from_rng(&mut rng),
                ..<_>::default()
            },
            triangles,
            summit,
            player: Player {
                // Something non-default for inital testing
                xy: zo_xy!{0., summit.y.0},
                angle: PI,
            }, // <_>::default(),
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

/// A 3 by 3 homogeneous affine trasformation matrix, with the bottom row
/// of `0 0 1` left implicit.
type Transform = [f32; 6];

fn apply_transform(xys: &mut [zo::XY], transform: Transform) {
    for xy in xys.iter_mut() {
        *xy = zo_xy!{
            xy.x.0 * transform[0]
            + xy.y.0 * transform[1]
            + /* 1. * */ transform[2],
            xy.x.0 * transform[3]
            + xy.y.0 * transform[4]
            + /* 1. * */ transform[5],
        }
    }
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

pub type CursorXY = DrawXY;

pub type DeltaTimeInSeconds = f32;

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<draw::Command>,
    input_flags: InputFlags,
    cursor_xy: CursorXY,
    draw_wh: DrawWH,
    dt: DeltaTimeInSeconds
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

    state.board.player.angle += PI * dt;

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

    macro_rules! convert_strip {
        ($strip: expr) => {
            $strip
                .into_iter()
                .map(|xy| zo_to_draw_xy(&state.sizes, xy))
                .collect()
        }
    }

    // TODO avoid this per-frame allocation or merge it with others.
    let pole = vec![
        zo_xy!{ pole_min_x, summit_xy.y.0 },
        zo_xy!{ pole_max_x, summit_xy.y.0 },
        zo_xy!{ pole_min_x, pole_top_y },
        zo_xy!{ pole_max_x, pole_top_y },
    ];

    commands.push(TriangleStrip(convert_strip!(pole), draw::Colour::Pole));

    const FLAG_H: f32 = POLE_H / 4.;
    const FLAG_W: f32 = FLAG_H;

    // TODO Animate the flag blowing in the wind.

    let flag = vec![
        zo_xy!{ pole_max_x, pole_top_y },
        zo_xy!{ pole_max_x, pole_top_y - FLAG_H },
        zo_xy!{ pole_max_x + FLAG_W, pole_top_y - FLAG_H / 2. },
    ];

    commands.push(TriangleStrip(convert_strip!(flag), draw::Colour::Flag));

    let player = {
        const LEG_WIDTH: f32 = 1./64.;//1024.;
        const BETWEEN_LEGS_HALF_WIDTH: f32 = LEG_WIDTH;
        const LEG_HEIGHT: f32 = LEG_WIDTH * 4.;

        const TORSO_HEIGHT: f32 = LEG_HEIGHT * 1.25;
        const HEAD_HEIGHT: f32 = LEG_HEIGHT * 0.5;

        const PLAYER_HEIGHT: f32 = LEG_HEIGHT + TORSO_HEIGHT + HEAD_HEIGHT;

        // We set these offsets so that we can rotate around the player's center.
        let x = 0.0;
        let y = -PLAYER_HEIGHT / 2.;

        let left_leg_min_x = x - (BETWEEN_LEGS_HALF_WIDTH + LEG_WIDTH);
        let left_leg_max_x = x - (BETWEEN_LEGS_HALF_WIDTH);

        let right_leg_min_x = x + (BETWEEN_LEGS_HALF_WIDTH);
        let right_leg_max_x = x + (BETWEEN_LEGS_HALF_WIDTH + LEG_WIDTH);

        let leg_max_y = y + LEG_HEIGHT;

        // TODO avoid this per-frame allocation or merge it with others.
        let mut player = Vec::with_capacity(64);

        player.extend_from_slice(&[
            // Left leg
            zo_xy!{left_leg_min_x, y},
            zo_xy!{left_leg_max_x, y},
            zo_xy!{left_leg_min_x, leg_max_y},
            zo_xy!{left_leg_max_x, leg_max_y},
        ]);

        // Right leg
        push_dengenerate_to(
            &mut player,
            (
                zo_xy!{right_leg_max_x, y},
                zo_xy!{right_leg_min_x, y},
            )
        );
        player.extend_from_slice(&[
            zo_xy!{right_leg_max_x, y},
            zo_xy!{right_leg_min_x, y},
            zo_xy!{right_leg_max_x, leg_max_y},
            zo_xy!{right_leg_min_x, leg_max_y},
        ]);

        // Might want an extended hip or something later.
        let torso_min_x = left_leg_min_x;
        let torso_min_y = leg_max_y;

        let torso_max_x = right_leg_max_x;
        let torso_max_y = torso_min_y + TORSO_HEIGHT;

        // Torso
        push_dengenerate_to(
            &mut player,
            (
                zo_xy!{torso_min_x, torso_min_y},
                zo_xy!{torso_max_x, torso_min_y},
            )
        );
        player.extend_from_slice(&[
            zo_xy!{torso_min_x, torso_min_y},
            zo_xy!{torso_max_x, torso_min_y},
            zo_xy!{torso_min_x, torso_max_y},
            zo_xy!{torso_max_x, torso_max_y},
        ]);

        let head_min_x = left_leg_max_x;
        let head_min_y = torso_max_y;

        let head_max_x = right_leg_min_x;
        let head_max_y = head_min_y + HEAD_HEIGHT;

        let head_mid_x = (head_min_x + head_max_x) / 2.;
        let head_mid_y = (head_min_y + head_max_y) / 2.;

        let head_radius = head_max_x - head_mid_x;

        // Head

        // Based on https://stackoverflow.com/a/15296912
        let mut angle = -PI/2.;
        let step = PI/16.;
        angle += step;

        let head_point_1 = zo_xy!{head_min_x, head_mid_y};

        // Pull an iteration out of th below loop so we have the first two points:
        let (head_point_2, head_point_3) = {
            let x = head_radius * angle.sin();
            let y = head_radius * angle.cos();

            angle += step;

            (
                zo_xy!{head_mid_x + x, head_mid_y + y},
                zo_xy!{head_mid_x + x, head_mid_y - y},
            )
        };

        push_dengenerate_to(
            &mut player,
            (
                head_point_1,
                head_point_2,
            )
        );

        player.push(head_point_1);
        player.push(head_point_2);
        player.push(head_point_3);

        while angle < PI/2. {
            // TODO can we pull the trig functions out of the loop?
            let x = head_radius * angle.sin();
            let y = head_radius * angle.cos();

            player.push(zo_xy!{head_mid_x + x, head_mid_y + y});
            player.push(zo_xy!{head_mid_x + x, head_mid_y - y});

            angle += step;
        }

        player.push(zo_xy!{head_max_x, head_mid_y});

        let xy = state.board.player.xy;
        let angle = state.board.player.angle;

        let (sin_of, cos_of) = angle.sin_cos();

        apply_transform(
            &mut player,
            [
                cos_of, -sin_of, xy.x.0,
                sin_of, cos_of, xy.y.0,
            ]
        );

        player
    };

    commands.push(TriangleStrip(convert_strip!(player), draw::Colour::Stone));

    fn move_along_angle_with_pre_sin_cos(
        (sin, cos): (Radians, Radians),
        radius: f32,
        at: zo::XY,
    ) -> zo::XY {
        // To move along the angle, we find the x and y of an axis-aligned right
        // triangle, where the hypotenuse starts at the current point, and ends
        // at the desired point.
        //   /|
        // H/ |O
        // /__|
        //   A

        // cos(angle) = A / H, so A = cos(angle) * H.
        let x = at.x.0 + cos * radius;

        // sin(angle) = O / H, so O = sin(angle) * H.
        let y = at.y.0 + sin * radius;

        zo_xy!{x, y}
    }

    let jump_arrow = {
        const JUMP_ARROW_HALF_W: f32 = 1./32.;//1024.;
        const JUMP_ARROW_HALF_H: f32 = JUMP_ARROW_HALF_W * 2.;

        const JUMP_ARROW_MIN_X: f32 = -JUMP_ARROW_HALF_W;
        const JUMP_ARROW_MAX_X: f32 = JUMP_ARROW_HALF_W;

        const JUMP_ARROW_MIN_Y: f32 = -JUMP_ARROW_HALF_H;
        const JUMP_ARROW_MAX_Y: f32 = JUMP_ARROW_HALF_H;

        let mut arrow = vec![
            zo_xy!{ JUMP_ARROW_MIN_X, JUMP_ARROW_MAX_Y },
            zo_xy!{ 0.0, 0.0 },
            zo_xy!{ JUMP_ARROW_MAX_X, 0.0 },
            zo_xy!{ JUMP_ARROW_MIN_X, JUMP_ARROW_MIN_Y },
        ];

        let cursor_zo_xy: zo::XY = draw_to_zo_xy(&state.sizes, cursor_xy);

        let px = state.board.player.xy.x.0;
        let py = state.board.player.xy.y.0;

        let rel_x = cursor_zo_xy.x.0 - px;
        let rel_y = cursor_zo_xy.y.0 - py;

        let angle = f32::atan2(rel_y, rel_x);

        const ARROW_RADIUS: f32 = JUMP_ARROW_HALF_W * 8.;

        let (sin_of, cos_of) = angle.sin_cos();

        let xy = move_along_angle_with_pre_sin_cos(
            (sin_of, cos_of),
            ARROW_RADIUS,
            state.board.player.xy,
        );

        let x = xy.x.0;
        let y = xy.y.0;

        apply_transform(
            &mut arrow,
            [
                cos_of, -sin_of, x,
                sin_of, cos_of, y,
            ]
        );

        arrow
    };

    commands.push(TriangleStrip(convert_strip!(jump_arrow), draw::Colour::Arrow));

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
