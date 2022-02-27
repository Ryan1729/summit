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
    use core::ops::{AddAssign, Add, SubAssign, Sub, MulAssign, Mul};

    /// Values outside the range [0, 1] are expected, but they are expected to be
    /// clipped later.
    pub type Zo = f32;

    #[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
    pub struct X(pub(crate) Zo);

    impl AddAssign for X {
        fn add_assign(&mut self, other: Self) {
            self.0 += other.0;
        }
    }

    impl Add for X {
        type Output = Self;

        fn add(mut self, other: Self) -> Self::Output {
            self += other;
            self
        }
    }

    impl SubAssign for X {
        fn sub_assign(&mut self, other: Self) {
            self.0 -= other.0;
        }
    }

    impl Sub for X {
        type Output = Self;

        fn sub(mut self, other: Self) -> Self::Output {
            self -= other;
            self
        }
    }

    impl MulAssign<Zo> for X {
        fn mul_assign(&mut self, scale: Zo) {
            self.0 *= scale;
        }
    }

    impl Mul<Zo> for X {
        type Output = Self;

        fn mul(mut self, scale: Zo) -> Self::Output {
            self *= scale;
            self
        }
    }

    impl Mul<X> for Zo {
        type Output = X;

        fn mul(self, mut other: X) -> Self::Output {
            other *= self;
            other
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
    pub struct Y(pub(crate) Zo);

    impl AddAssign for Y {
        fn add_assign(&mut self, other: Self) {
            self.0 += other.0;
        }
    }

    impl Add for Y {
        type Output = Self;

        fn add(mut self, other: Self) -> Self::Output {
            self += other;
            self
        }
    }

    impl SubAssign for Y {
        fn sub_assign(&mut self, other: Self) {
            self.0 -= other.0;
        }
    }

    impl Sub for Y {
        type Output = Self;

        fn sub(mut self, other: Self) -> Self::Output {
            self -= other;
            self
        }
    }

    impl MulAssign<Zo> for Y {
        fn mul_assign(&mut self, scale: Zo) {
            self.0 *= scale;
        }
    }

    impl Mul<Zo> for Y {
        type Output = Self;

        fn mul(mut self, scale: Zo) -> Self::Output {
            self *= scale;
            self
        }
    }

    impl Mul<Y> for Zo {
        type Output = Y;

        fn mul(self, mut other: Y) -> Self::Output {
            other *= self;
            other
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    pub struct XY {
        pub x: X,
        pub y: Y,
    }

    impl AddAssign for XY {
        fn add_assign(&mut self, other: Self) {
            self.x += other.x;
            self.y += other.y;
        }
    }

    impl Add for XY {
        type Output = Self;

        fn add(mut self, other: Self) -> Self::Output {
            self += other;
            self
        }
    }

    impl SubAssign for XY {
        fn sub_assign(&mut self, other: Self) {
            self.x -= other.x;
            self.y -= other.y;
        }
    }

    impl Sub for XY {
        type Output = Self;

        fn sub(mut self, other: Self) -> Self::Output {
            self -= other;
            self
        }
    }

    impl MulAssign<Zo> for XY {
        fn mul_assign(&mut self, scale: Zo) {
            self.x *= scale;
            self.y *= scale;
        }
    }

    impl Mul<Zo> for XY {
        type Output = Self;

        fn mul(mut self, scale: Zo) -> Self::Output {
            self *= scale;
            self
        }
    }

    impl Mul<XY> for Zo {
        type Output = XY;

        fn mul(self, mut other: XY) -> Self::Output {
            other *= self;
            other
        }
    }

    use core::fmt;

    impl fmt::Display for XY {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if f.alternate() {
                write!(f, "(\n{},\n{}\n)\n", self.x.0, self.y.0)
            } else {
                write!(f, "({}, {})", self.x.0, self.y.0)
            }
        }
    }

    impl XY {
        pub fn distance_sq(self, other: XY) -> Zo {
            let x = self.x.0 - other.x.0;
            let y = self.y.0 - other.y.0;

            x * x + y * y
        }

        pub fn dot(self, other: XY) -> Zo {
            self.x.0 * other.x.0 + self.y.0 * other.y.0
        }
    }

    #[macro_export]
    macro_rules! zo_xy {
        () => {
            zo_xy!{0., 0.}
        };
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

type Line = (zo::XY, zo::XY);

fn lines_collide(l1: Line, l2: Line) -> bool {
    if l1 == l2 { return true }
    // This is based on this wikipedia page:
    // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment

    // Two terms, `t` and `u` are defined there, including formulae for them.

    // For the intersection point, (if any) between the lines `t` and `u` are the
    // fraction of the way from `l1.0` to `l1.1`, and `l2.0` to `l2.1` respectively.
    // That is, for varying values of `t`, (for example) from 0 to 1,
    // `l1.0 * t(l1.1 - l1.0)` takes on all points that are a part of `l1`.

    // From these definitions, we can apparently conclude that the will be an
    // intersection if and only if `t` and `u` are in the range [0, 1].

    let x1 = l1.0.x.0;
    let y1 = l1.0.y.0;

    let x2 = l1.1.x.0;
    let y2 = l1.1.y.0;

    let x3 = l2.0.x.0;
    let y3 = l2.0.y.0;

    let x4 = l2.1.x.0;
    let y4 = l2.1.y.0;

    // The formulae given are fractions with the same denominator, which we will
    // call `d`. We can avoid dividing by that denominator by checking that
    // `t * d` and `u * d` are in the range [0, `d`].

    let d = ((x1 - x2) * (y3 - y4)) - ((x3 - x4) * (y1 - y2));

    let t_times_d = ((x1 - x3) * (y3 - y4)) - ((x3 - x4) * (y1 - y3));
    let u_times_d = ((x1 - x3) * (y1 - y2)) - ((x1 - x2) * (y1 - y3));

    /*
    if d >= 0. {
        0. <= t_times_d && t_times_d <= d && 0. <= u_times_d && u_times_d <= d
    } else {
        0. >= t_times_d && t_times_d >= d && 0. >= u_times_d && u_times_d >= d
    }
    */
    // ^^^ A more readable, but also more branchy equivalent of the below code.
    // See comparision at: https://rust.godbolt.org/z/8rYYhn5W9
    // Fair warning, I have not done any real benchmarking of this code.

    let d_signum = d.signum();
    let t_times_d_signum = t_times_d.signum();
    let u_times_d_signum = u_times_d.signum();

    let d_abs = d.abs();
    let t_times_d_abs = t_times_d.abs();
    let u_times_d_abs = u_times_d.abs();

    d_signum == t_times_d_signum && t_times_d_signum == u_times_d_signum
    && 0. <= t_times_d_abs && t_times_d_abs <= d_abs
    && 0. <= u_times_d_abs && u_times_d_abs <= d_abs
}

#[cfg(test)]
mod lines_collide_returns_true_for_these_observed_values {
    use super::*;

    // We observed what should be a collision go undetected.
    // The following diagram describes the scenario:
    // |    |/ <-- slope
    // |    /
    // |   /|
    // |__/_|  <-- player-avatar's right leg
    //   /

    const SLOPE_TOP: zo::XY = zo_xy!{0.014025899, 0.015630051};
    const SLOPE_BOTTOM: zo::XY = zo_xy!{};

    const SLOPE: Line = (SLOPE_TOP, SLOPE_BOTTOM);

    const RIGHT_LEG_RIGHT_BOTTOM: zo::XY = zo_xy!{0.001953125, 0.0020731823};

    #[test]
    fn leg_bottom() {
        let right_leg_left_bottom = zo_xy!{0.0009765625, 0.0020731823};

        let leg_bottom = (RIGHT_LEG_RIGHT_BOTTOM, right_leg_left_bottom);

        assert!(lines_collide(leg_bottom, SLOPE));
    }

    #[test]
    fn leg_right_side() {
        let right_leg_right_top = zo_xy!{0.001953125, 0.0059794323};

        let leg_right_side = (right_leg_right_top, RIGHT_LEG_RIGHT_BOTTOM);

        assert!(lines_collide(leg_right_side, SLOPE));
    }
}

#[test]
fn lines_collide_acts_as_expected_on_these_origin_crossing_examples() {
    // Short for half-length;
    const H: zo::Zo = 1.;

    const L1: Line = (zo_xy!{-H, 0.}, zo_xy!{ H, 0.});
    const L2: Line = (zo_xy!{-H, -H}, zo_xy!{ H,  H});
    const L3: Line = (zo_xy!{0., -H}, zo_xy!{0.,  H});
    const L4: Line = (zo_xy!{ H, -H}, zo_xy!{-H,  H});

    for l1 in [L1, L2, L3, L4] {
        for l2 in [L1, L2, L3, L4] {
            assert!(lines_collide(l1, l2), "{:?} {:?}", l1, l2);
        }
    }
}

#[test]
fn lines_collide_detects_this_co_incident_line_example() {
    // Short for half-length;
    const H: zo::Zo = 1.;

    const L1: Line = (zo_xy!{   -H,   -H}, zo_xy!{   H,   H});
    const L2: Line = (zo_xy!{-H/2.,-H/2.}, zo_xy!{H/2.,H/2.});

    assert!(lines_collide(L1, L2), "{:?} {:?}", L1, L2);
}

#[test]
fn lines_collide_detects_this_identical_line_example() {
    // Short for half-length;
    const H: zo::Zo = 1.;

    const L1: Line = (zo_xy!{0.,-H}, zo_xy!{0., H});

    assert!(lines_collide(L1, L1));
}

#[cfg(test)]
mod lines_collide_can_return_both_values_when_the_collision_values_are {
    use super::*;

    macro_rules! with_str {
        ($e: expr) => {
            ($e, stringify!($e))
        }
    }

    macro_rules! a {
        (
            $(
                (($x1: expr, $y1: expr), ($x2: expr, $y2: expr))
                (($x3: expr, $y3: expr), ($x4: expr, $y4: expr))
                $expected: literal
            )+
            ;
            $d_op: tt $t_times_d_op: tt $u_times_d_op: tt
        ) => {{
            $(
                let l1 = (zo_xy!{$x1, $y1}, zo_xy!{$x2, $y2});
                let l2 = (zo_xy!{$x3, $y3}, zo_xy!{$x4, $y4});

                // These tests assume that the implementation of `lines_collide`
                // does the same calculations. We could use some shared macro in
                // the implementation, but that makes both this and the
                // implementation harder to read, and honestly I would be surprised
                // if we changed either after we get it all working.

                let (d, d_str) = with_str!(
                    (($x1 - $x2) * ($y3 - $y4)) - (($x3 - $x4) * ($y1 - $y2))
                );
                let d_str_lit = "(($x1 - $x2) * ($y3 - $y4)) - (($x3 - $x4) * ($y1 - $y2))";

                let (t_times_d, t_times_d_str) = with_str!(
                    (($x1 - $x3) * ($y3 - $y4)) - (($x3 - $x4) * ($y1 - $y3))
                );
                let t_times_d_str_lit = "(($x1 - $x3) * ($y3 - $y4)) - (($x3 - $x4) * ($y1 - $y3))";

                let (u_times_d, u_times_d_str) = with_str!(
                    (($x1 - $x3) * ($y1 - $y2)) - (($x1 - $x2) * ($y1 - $y3))
                );
                let u_times_d_str_lit = "(($x1 - $x3) * ($y3 - $y4)) - (($x3 - $x4) * ($y1 - $y3))";

                let expected = $expected;

                assert!(d $d_op 0., "\n\nexpected: {expected}\nd: {}\n = {} {}\n\n", d_str_lit, d_str, stringify!($d_op 0.));
                assert!(t_times_d $t_times_d_op 0., "\n\nexpected: {expected}\nt_times_d: {}\n = {} {}\n\n", t_times_d_str_lit, t_times_d_str, stringify!($t_times_d_op 0.));
                assert!(u_times_d $u_times_d_op 0., "\n\nexpected: {expected}\nu_times_d: {}\n = {} {}\n\n", u_times_d_str_lit, u_times_d_str, stringify!($u_times_d_op 0.));

                let actual = lines_collide(l1, l2);
                assert_eq!(actual, $expected, "expected {}, got {}", $expected, actual);
            )+
        }}
    }

    #[test]
    fn pos_pos_pos() {
        a!(
            ((-1., 1.), ( 1., 0.)) (( 0.,-1.), ( 0., 1.)) true
            ((-1., 1.), ( 1., 0.)) (( 0.,-1.), ( 0., 0.)) false
            // via SMT
            ((1./8., 1./2.), ( 1., 0.)) (( -1./2., -1./2.), (1., -1.)) false
            // Modified from above: flip sign of y4
            ((1./8., 1./2.), ( 1., 0.)) (( -1./2., -1./2.), (1., 1.)) true
            ;
            > > >
        );
    }

    #[test]
    fn neg_pos_pos() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 0.)) ((-1./2., -1./2.), ( -1., -3.)) false
            // SMT reports true case is unsatisfiable
            ;
            < > >
        );
    }

    #[test]
    fn pos_neg_pos() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 0.)) ((-1./2., -1./2.), (-1., 1.)) false
            // SMT reports true case is unsatisfiable
            ;
            > < >
        );
    }

    #[test]
    fn neg_neg_pos() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 0.)) ((-1./2., -1./2.), ( -1., -1.)) false
            // SMT reports true case is unsatisfiable
            ;
            < < >
        );
    }

    #[test]
    fn pos_pos_neg() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 3.)) ((-1./2., -1./2.), ( -1., -3./2.)) false
            // SMT reports true case is unsatisfiable
            ;
            > > <
        );
    }

    #[test]
    fn neg_pos_neg() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 3.)) ((-1./2., -1./2.), (-1., -3.)) false
            // SMT reports true case is unsatisfiable
            ;
            < > <
        );
    }

    #[test]
    fn pos_neg_neg() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 3.)) ((-1./2., -1./2.), ( -1., 0.)) false
            // SMT reports true case is unsatisfiable
            ;
            > < <
        );
    }

    #[test]
    fn neg_neg_neg() {
        a!(
            // via SMT
            (( 1./8., 1./2.), ( 1., 3.)) ((-1./2., -1./2.), ( 1., 2.)) true
            // via SMT, by invering intersection check
            (( 1./8., 1./2.), ( 1., 3.)) ((-1./2., -1./2.), ( 1., 7./2.)) false
            ;
            < < <
        );
    }

    // The test cases labeled "Via SMT" above were found by passing the following
    // code into the Z3 SMT solver, and then manually flipping the `>` to `<`,
    // and/or commenting out the intersection check, as appropriate for each test
    // case. The particular online version I used raised an "unsupported" error when
    // I tried to use `let`, which is why I repeated subexpressions so much.
    /*

    ; Variable declarations
    (declare-fun x1 () Real)
    (declare-fun y1 () Real)
    (declare-fun x2 () Real)
    (declare-fun y2 () Real)

    (declare-fun x3 () Real)
    (declare-fun y3 () Real)
    (declare-fun x4 () Real)
    (declare-fun y4 () Real)

    ;
    ; Constraints
    ;

    ; Lines not points
    (assert (not
        (and
            (= x1 x2)
            (= y1 y2)
        )
    ))
    (assert (not
        (and
            (= x3 x4)
            (= y3 y4)
        )
    ))

    ; d has right sign
    (assert (>
        (-
            (*
                (- x1 x2)
                (- y3 y4)
            )
            (*
                (- x3 x4)
                (- y1 y2)
            )
        )
        0
    ))

    ; t * d has right sign
    (assert (>
        (-
            (*
                (- x1 x3)
                (- y3 y4)
            )
            (*
                (- y1 y3)
                (- x3 x4)
            )
        )
        0
    ))

    ; u * d has right sign
    (assert (>
        (-
            (*
                (- x1 x3)
                (- y1 y2)
            )
            (*
                (- y1 y3)
                (- x1 x2)
            )
        )
        0
    ))

    ; the lines intersect if 0 <= t <= 1 and 0 <= u <= 1
    ; 0 <= t
    (assert (<=
        0
        (/
           ; t * d
           (-
              (*
                (- x1 x3)
                (- y3 y4)
              )
              (*
                (- y1 y3)
                (- x3 x4)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
    ))
    ; t <= 1
    (assert (<=
        (/
           ; t * d
           (-
              (*
                (- x1 x3)
                (- y3 y4)
              )
              (*
                (- y1 y3)
                (- x3 x4)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
        1
    ))

    ; 0 <= u
    (assert (<=
        0
        (/
           ; u * d
           (-
              (*
                (- x1 x3)
                (- y1 y2)
              )
              (*
                (- y1 y3)
                (- x1 x2)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
    ))
    ; u <= 1
    (assert (<=
        (/
           ; u * d
           (-
              (*
                (- x1 x3)
                (- y1 y2)
              )
              (*
                (- y1 y3)
                (- x1 x2)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
        1
    ))

    ; Solve
    (check-sat)
    (get-model)

    */

    // I needed to invert the intersection as well. Here's the code for that check
    // which replaces the asserts checking for an intersection.
    /*
    ; the lines intersect if 0 <= t <= 1 and 0 <= u <= 1
    ; assert at least on intersection condition is violated
    (assert (or
      (>
        0
        (/
           ; t * d
           (-
              (*
                (- x1 x3)
                (- y3 y4)
              )
              (*
                (- y1 y3)
                (- x3 x4)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
      )
    (>
        (/
           ; t * d
           (-
              (*
                (- x1 x3)
                (- y3 y4)
              )
              (*
                (- y1 y3)
                (- x3 x4)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
        1
    )
    (>
        0
        (/
           ; u * d
           (-
              (*
                (- x1 x3)
                (- y1 y2)
              )
              (*
                (- y1 y3)
                (- x1 x2)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
    )
    (>
        (/
           ; u * d
           (-
              (*
                (- x1 x3)
                (- y1 y2)
              )
              (*
                (- y1 y3)
                (- x1 x2)
              )
           )
           ; d
           (-
              (*
                (- x1 x2)
                (- y3 y4)
              )
              (*
                (- x3 x4)
                (- y1 y2)
              )
           )
        )
        1
    )
    ))
    */
}

fn bounce_vector_if_overlapping(
    player: &Player,
    mountain_triangles: &Triangles
) -> Option<zo::XY> {
    let player_triangles = player.get_triangles();

    for pw in player_triangles.windows(3) {
        let player_lines = [
            (pw[0], pw[1]),
            (pw[1], pw[2]),
            (pw[2], pw[0]),
        ];

        // TODO Use a spatial partition to reduce the amount of mountain lines
        // we need to test.
        for mw in mountain_triangles.windows(3) {
            let mountain_lines = [
                (mw[0], mw[1]),
                (mw[1], mw[2]),
                (mw[2], mw[0]),
            ];

            for player_line in player_lines {
                for mountain_line in mountain_lines {
                    let is_colliding = lines_collide(player_line, mountain_line);

                    if is_colliding {
                        // We want a vector that will separate the player from the
                        // collison. Since the player will enter a collision edge first,
                        // we point the vector towards the player's center.

                        // That's one point for the line that we derive the vector from,
                        // but we need another one. A natural other point for the vector
                        // is the center of the colliding line.
                        let line_center = zo_xy!{
                            (player_line.0.x.0 + player_line.1.x.0) / 2.,
                            (player_line.0.y.0 + player_line.1.y.0) / 2.,
                        };

                        // We have two potential normals here: we can either rotate by
                        // pi/2 radians, or -pi/2 radians. Recall that roating looks
                        // uses this transform for given angle:
                        // [
                        //    cos(angle), -sin(angle), 0.,
                        //    sin(angle), cos(angle), 0.,
                        // ]


                        // When the angle is pi/2, then the transform is the same as:
                        // [
                        //    0., -1., 0.,
                        //    1., 0., 0.,
                        // ]
                        // Inlining that gives us:
                        let line_normal_a = zo_xy!{-player_line.0.y.0, player_line.0.x.0};
                        // When the angle is -pi/2, then the transform is the same as:
                        // [
                        //    0., 1., 0.,
                        //    -1., 0., 0.,
                        // ]
                        // Inlining that gives us:
                        let line_normal_b = zo_xy!{player_line.0.y.0, -player_line.0.x.0};

                        // Pick the one that points the closest to the player, so we
                        // bounce off of the mountain.
                        let line_normal = if zo::XY::distance_sq(
                            line_center + line_normal_a,
                            player.xy,
                        ) < zo::XY::distance_sq(
                            line_center + line_normal_b,
                            player.xy,
                        ) {
                            line_normal_a
                        } else {
                            line_normal_b
                        };

                        let mut bounce_vector = 2.
                            * line_normal
                            * line_normal.dot(player.velocity);

                        // We want the bounce to be forceful enough that the collision
                        // stops, so we arbitrairaly scale it up to get that effect.
                        // TODO Derive a value here in a principled way?
                        bounce_vector *= 16.;

                        return Some(bounce_vector)
                    }
                }
            }
        }
    }

    None
}

#[test]
fn bounce_vector_if_overlapping_detects_this_overlap() {
    let player = Player {
        // We unrealisitcally place the middle of the player on the line.
        xy: zo_xy!{0.25, 0.25},
        ..<_>::default()
    };

    // Note that the overlapping part of the mountain is an implied line
    // that does not show up in in the `windows` iterator.
    let mountain = vec![
        zo_xy!{},
        zo_xy!{0.5, 0.},
        zo_xy!{0.5, 0.5},
    ];

    assert!(
        bounce_vector_if_overlapping(&player, &mountain).is_some()
    );
}

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

const OVERHANG_MIN_X: zo::X = zo::X(1. / 64.);

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

        if overhang_x < OVERHANG_MIN_X.0 { overhang_x = OVERHANG_MIN_X.0; }
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

fn push_random_length_overhang_triangles(
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

    const PER_OVERHANG_MINIMUM: u32 = 4;

    compile_time_assert!(usize::BITS >= u32::BITS);
    if remaining_count >= PER_OVERHANG_MINIMUM as usize {
        let available_to_use: u32 = xs_range(
            rng,
            PER_OVERHANG_MINIMUM..(remaining_count.saturating_add(1)) as u32
        );

        const POINTS_PER_DELTA: f32 = 2.;

        let x_delta = (end.x.0 - start.x.0) / (count as f32 / POINTS_PER_DELTA);
        let y_delta = (end.y.0 - start.y.0) / (count as f32 / POINTS_PER_DELTA);

        compile_time_assert!(usize::BITS >= u32::BITS);

        while remaining_count >= PER_OVERHANG_MINIMUM as usize {
            let mut used = 0;

            const USED_IN_THIS_LOOP: u32 = 2;
            while available_to_use - used >= USED_IN_THIS_LOOP + PER_OVERHANG_MINIMUM {
                x_base += x_delta;
                y_base += y_delta;

                let x = x_base + x_delta * xs_zero_to_one(rng);

                triangles.push(zo::XY{ x: zo::X(x), y: zo::Y(BOTTOM_Y) });
                triangles.push(zo::XY{ x: zo::X(x), y: zo::Y(y_base + y_delta) });
                used += USED_IN_THIS_LOOP;
            }

            x_base += x_delta;
            y_base += y_delta;

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

            // PER_OVERHANG_MINIMUM corresponds to the amount used in this block
            {
                let (x, y) = gen_xy!();

                push_with_floor_point(triangles, zo_xy!{x, y});
                used += 2;

                x_base += x_delta;
                y_base += y_delta;

                let (x, y) = gen_xy!();

                triangles.push(zo::XY{ x: zo::X(x), y: zo::Y(BOTTOM_Y) });
                used += 1;

                let mut overhang_x = x - x_delta * (used as f32);

                if overhang_x < OVERHANG_MIN_X.0 { overhang_x = OVERHANG_MIN_X.0; }
                if overhang_x < min_x { overhang_x = min_x; }
                triangles.push(zo::XY{ x: zo::X(overhang_x), y: zo::Y(y) });
                used += 1;
            }

            compile_time_assert!(usize::BITS >= u32::BITS);
            remaining_count = remaining_count.saturating_sub(used as usize);
        }
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

#[derive(Clone, Copy, Debug, Default)]
struct Player {
    xy: zo::XY,
    angle: Radians,
    velocity: zo::XY,
}

const PLAYER_SCALE: f32 = 1./1024.;

impl Player {
    fn get_triangles(&self) -> Triangles {
        const LEG_WIDTH: f32 = PLAYER_SCALE;
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

        let xy = self.xy;
        let angle = self.angle;

        let (sin_of, cos_of) = angle.sin_cos();

        let transform = [
            cos_of, -sin_of, xy.x.0,
            sin_of, cos_of, xy.y.0,
        ];

        apply_transform(&mut player, transform);

        player
    }
}

#[derive(Debug, Default)]
struct Board {
    tiles: Tiles,
    eye: Eye,
    triangles: Triangles,
    summit: zo::XY,
    player: Player,
}

const SUMMIT_EXTRA: f32 = 0.0015;

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

        const SECTIONS: [MountainSection; 5] = [
            push_evenly_spaced_triangles_section,
            push_spiky_triangles_with_floor_points,
            push_overhang_triangles,
            push_random_length_overhang_triangles,

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
                let mut count_for_this_section = xs_range(rng, 0..12) as usize;

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
        summit.y.0 = max_y + SUMMIT_EXTRA;

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
                // Read off of the screen while paused
                xy: zo_xy!{0., 0.007625},
                ..<_>::default()
            },
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

#[allow(unused)]
const IDENTITY_TRANSFORM: Transform = [
    1., 0., 0.,
    0., 1., 0.,
  /*0., 0., 1.,*/
];

/// Combine two transforms into a single transform that when applied to a point will
/// apply t1 then t2.
/// AKA: t2 matrix multiplied by t1
#[allow(unused)]
fn merge_transforms(t1: Transform, t2: Transform) -> Transform {
    [
        t2[0] * t1[0] + t2[1] * t1[3] /* + t2[2] * 0. */,
        t2[0] * t1[1] + t2[1] * t1[4] /* + t2[2] * 0. */,
        t2[0] * t1[2] + t2[1] * t1[5] + t2[2] /* * 1. */,

        t2[3] * t1[0] + t2[4] * t1[3] /* + t2[5] * 0. */,
        t2[3] * t1[1] + t2[4] * t1[4] /* + t2[5] * 0. */,
        t2[3] * t1[2] + t2[4] * t1[5] + t2[5] /* * 1. */,
    ]
}

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

#[cfg(test)]
mod merge_transforms_then_apply_is_equivalent_to_sequential_applies {
    use super::*;

    #[test]
    fn on_this_example() {
        let t1 = [
            1., 2., 3.,
            4., 5., 6.,
        ];

        let t2 = [
            7., 8., 9.,
            0., 1., 2.,
        ];

        let merged = merge_transforms(t1, t2);

        macro_rules! a {
            ($($token: tt)*) => {{
                let point = zo_xy!{$($token)*};

                let mut merged_point_array = [point];

                apply_transform(&mut merged_point_array, merged);

                let mut sequential_point_array = [point];

                apply_transform(&mut sequential_point_array, t1);
                apply_transform(&mut sequential_point_array, t2);

                assert_eq!(merged_point_array, sequential_point_array);
            }}
        }

        a!();
        a!(1., 1.);
        a!(1., 2.);
        a!(1., -2.);
        a!(-1., 2.);
        a!(-1., -2.);
        a!(std::f32::consts::PI, 1./std::f32::consts::PI);
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
pub const INPUT_INTERACT_DOWN: InputFlags           = 0b0000_0010_0000_0000;

/// Should be set if the mouse button was pressed or released this frame.
pub const INPUT_LEFT_MOUSE_CHANGED: InputFlags      = 0b0000_0100_0000_0000;
pub const INPUT_LEFT_MOUSE_DOWN: InputFlags         = 0b0000_1000_0000_0000;

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

pub const CAMERA_SCALE_FACTOR: f32 = 24.;//256.;

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<draw::Command>,
    input_flags: InputFlags,
    cursor_xy: CursorXY,
    draw_wh: DrawWH,
    dt: DeltaTimeInSeconds
) {
    #[cfg(feature = "fake-fixed-dt")]
    let dt = {
        drop(dt); // Hush unused lint
        1./60.
    };

    use draw::{TextSpec, TextKind, Command::*};

    if draw_wh != state.sizes.draw_wh {
        state.sizes = draw::fresh_sizes(draw_wh);
    }

    commands.clear();

    let left_mouse_button_down = input_flags & INPUT_LEFT_MOUSE_DOWN != 0;

    let left_mouse_button_pressed =
        input_flags & INPUT_LEFT_MOUSE_CHANGED != 0
        && left_mouse_button_down;
    let left_mouse_button_released =
        input_flags & INPUT_LEFT_MOUSE_CHANGED != 0
        && !left_mouse_button_down;

    assert!(
        !(left_mouse_button_pressed && left_mouse_button_released)
    );

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

    let cursor_zo_xy: zo::XY = draw_to_zo_xy(&state.sizes, cursor_xy);

    let cursor_rel_xy = cursor_zo_xy - zo_xy!{0.5, 0.5};

    // The following is meant to be Symplectic Eulerian / Semi-Implicit Eulerian
    // integration, based on the description at:
    // https://www.gamedev.net/forums/topic/611021-euler-integration-collision-response/

    const GRAVITY: zo::XY = zo_xy!{0., -1. * PLAYER_SCALE};

    let mut player_impulse = GRAVITY;

    if left_mouse_button_pressed /* && !is_colliding */ {
        const JUMP_SCALE: f32 = 1024. * PLAYER_SCALE;
        player_impulse += cursor_rel_xy * JUMP_SCALE;
    }

    let mut arrow_impulse = match input {
        NoChange | Interact => zo_xy!{},
        Dir(Up) => zo_xy!{0., 1.},
        // TODO sqrt(2)?
        Dir(UpRight) => zo_xy!{1., 1.},
        Dir(Right) => zo_xy!{1., 0.},
        Dir(DownRight) => zo_xy!{1., -1.},
        Dir(Down) => zo_xy!{0., -1.},
        Dir(DownLeft) => zo_xy!{-1., -1.},
        Dir(Left) => zo_xy!{-1., 0.},
        Dir(UpLeft) => zo_xy!{-1., 1.},
    };

    const ARROW_SCALE: f32 = 1./128. * PLAYER_SCALE;
    arrow_impulse *= ARROW_SCALE;

    player_impulse += arrow_impulse;

    // apply forces
    state.board.player.velocity += player_impulse * dt;

    let mut would_have_bounced = None;
    let mountain_colour;

    if state.board.player.xy.y.0 < 0.
    || {
        let already_overlapping = bounce_vector_if_overlapping(
            &state.board.player,
            &state.board.triangles
        ).is_some();

        already_overlapping
    } {
        mountain_colour = draw::Colour::Arrow;

        state.board.player.velocity = zo_xy!{};
        state.board.player.xy.y.0 += 4. * PLAYER_SCALE;
    } else {
        let mut new_player = state.board.player.clone();

        // intergrate
        new_player.xy += new_player.velocity * dt;
        if left_mouse_button_down {
            new_player.angle += PI * dt;
        }

        if let Some(bounce_vector) = bounce_vector_if_overlapping(
            &new_player,
            &state.board.triangles
        ) {
            // re-integrate including collision response
            new_player = state.board.player.clone();

            new_player.velocity -= bounce_vector;

            new_player.xy += new_player.velocity * dt;
            if left_mouse_button_down {
                new_player.angle += PI * dt;
            }

            if bounce_vector_if_overlapping(
                &new_player,
                &state.board.triangles
            ).is_some() {
                mountain_colour = draw::Colour::Flag;

                would_have_bounced = Some(new_player);
                // prevent overlapping
                state.board.player.velocity = zo_xy!{};
            } else {
                mountain_colour = draw::Colour::Pole;

                state.board.player = new_player;
            }
        } else {
            mountain_colour = draw::Colour::Stone;

            state.board.player = new_player;
        }
    }

    let player_triangles = state.board.player.get_triangles();

    //
    // Render
    //

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

    macro_rules! convert_strip {
        ($strip: expr) => {{
            let mut strip = $strip;

            let camera_scale: f32 = state.sizes.play_xywh.w / CAMERA_SCALE_FACTOR;

            let camera_translation = zo_xy!{
                -(state.board.player.xy.x.0 * camera_scale) + 0.5,
                -(state.board.player.xy.y.0 * camera_scale) + 0.5,
            };

            let camera_transform = [
                camera_scale, 0., camera_translation.x.0,
                0., camera_scale, camera_translation.y.0,
            ];

            apply_transform(&mut strip, camera_transform);

            strip
                .into_iter()
                .map(|xy| zo_to_draw_xy(&state.sizes, xy))
                .collect()
        }}
    }

    let summit_xy = state.board.summit;

    const POLE_HALF_W: f32 = 1./4096.;
    const POLE_H: f32 = POLE_HALF_W * 8.;

    const POLE_SUNK_IN: f32 = SUMMIT_EXTRA / 2.;

    let pole_min_y = summit_xy.y.0 - POLE_SUNK_IN;
    let pole_max_y = summit_xy.y.0 + POLE_H;
    let pole_min_x = summit_xy.x.0 - POLE_HALF_W;
    let pole_max_x = summit_xy.x.0 + POLE_HALF_W;

    // TODO avoid this per-frame allocation or merge it with others.
    let pole = vec![
        zo_xy!{ pole_min_x, pole_min_y },
        zo_xy!{ pole_max_x, pole_min_y },
        zo_xy!{ pole_min_x, pole_max_y },
        zo_xy!{ pole_max_x, pole_max_y },
    ];

    commands.push(TriangleStrip(convert_strip!(pole), draw::Colour::Pole));

    const FLAG_H: f32 = POLE_H / 4.;
    const FLAG_W: f32 = FLAG_H;

    // TODO Animate the flag blowing in the wind.

    let flag = vec![
        zo_xy!{ pole_max_x, pole_max_y },
        zo_xy!{ pole_max_x, pole_max_y - FLAG_H },
        zo_xy!{ pole_max_x + FLAG_W, pole_max_y - FLAG_H / 2. },
    ];

    commands.push(TriangleStrip(convert_strip!(flag), draw::Colour::Flag));

    let mountain: draw::TriangleStrip =
        convert_strip!(
            state.board.triangles.clone()
        );

    commands.push(TriangleStrip(mountain, mountain_colour));

    commands.push(TriangleStrip(
        convert_strip!(player_triangles),
        draw::Colour::Arrow//draw::Colour::Stone
    ));

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
        const JUMP_ARROW_HALF_W: f32 = PLAYER_SCALE;
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

        let angle = f32::atan2(cursor_rel_xy.y.0, cursor_rel_xy.x.0);

        const ARROW_RADIUS: f32 = JUMP_ARROW_HALF_W * 8.;

        let (sin_of, cos_of) = angle.sin_cos();

        let xy = move_along_angle_with_pre_sin_cos(
            (sin_of, cos_of),
            ARROW_RADIUS,
            state.board.player.xy,
        );

        let x = xy.x.0;
        let y = xy.y.0;

        let transform = [
            cos_of, -sin_of, x,
            sin_of, cos_of, y,
        ];

        apply_transform(
            &mut arrow,
            transform,
        );

        arrow
    };

    commands.push(TriangleStrip(
        convert_strip!(jump_arrow),
        draw::Colour::Arrow
    ));

    //
    // Debugging player(s) {
    //
    if let Some(p) = would_have_bounced {
        commands.push(TriangleStrip(
            convert_strip!(p.get_triangles()),
            draw::Colour::Pole
        ));
    }
    //
    // }
    //


    //
    // Debugging Lines
    //
    #[cfg(any())]
    {
        let mountain_points: Vec<_> = convert_strip!(vec![
            state.board.triangles[2],
            state.board.triangles[0],
        ]);

        let player_triangles = state.board.player.get_triangles();

        let player_points: Vec<_> = convert_strip!(player_triangles.clone());

        let lines = [
            (mountain_points[0], mountain_points[1]),
            (player_points[13], player_points[14]),
            (player_points[15], player_points[13]),
        ];

        for line in lines {
            commands.push(Line(
                line,
                draw::Colour::Flag
            ));
        }
    }

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
                "sizes: {:?}\nanimation_timer: {:?}\ncursor_rel_xy: {}\nplayer.xy: {}\nbounce.xy: {}\n",
                state.sizes,
                state.animation_timer,
                cursor_rel_xy,
                state.board.player.xy,
                would_have_bounced.map(|p| p.xy.to_string()).unwrap_or("".to_string()),
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
