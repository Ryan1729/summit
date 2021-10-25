#![deny(unused)]
#![deny(bindings_with_variant_name)]

// In case we decide that we care about no_std/not allocating
type StrBuf = String;

type PlayX = DrawLength;
type PlayY = DrawLength;
type PlayW = DrawLength;
type PlayH = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct PlayXYWH {
    pub x: PlayX,
    pub y: PlayY,
    pub w: PlayW,
    pub h: PlayH,
}

type BoardX = DrawLength;
type BoardY = DrawLength;
type BoardW = DrawLength;
type BoardH = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct BoardXYWH {
    pub x: BoardX,
    pub y: BoardY,
    pub w: BoardW,
    pub h: BoardH,
}

pub type DrawX = DrawLength;
pub type DrawY = DrawLength;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DrawXY {
    pub x: DrawX,
    pub y: DrawY,
}

impl core::ops::Add for DrawXY {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl core::ops::AddAssign for DrawXY {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

pub type DrawLength = f32;
pub type DrawW = DrawLength;
pub type DrawH = DrawLength;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DrawWH {
    pub w: DrawW,
    pub h: DrawH,
}

pub type TileCount = usize;
pub type TileSideLength = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct Sizes {
    pub draw_wh: DrawWH,
    pub play_xywh: PlayXYWH,
    pub board_xywh: BoardXYWH,
    pub tile_side_length: TileSideLength,
}

const LEFT_UI_WIDTH_TILES: TileCount = 9;
const RIGHT_UI_WIDTH_TILES: TileCount = 9;
const CENTER_UI_WIDTH_TILES: TileCount = if crate::tile::X::COUNT < crate::tile::Y::COUNT {
    crate::tile::X::COUNT as _
} else {
    crate::tile::Y::COUNT as _
};
const DRAW_WIDTH_TILES: TileCount = LEFT_UI_WIDTH_TILES 
    + CENTER_UI_WIDTH_TILES 
    + RIGHT_UI_WIDTH_TILES;

pub fn fresh_sizes(wh: DrawWH) -> Sizes {
    let w_length_bound = wh.w / DRAW_WIDTH_TILES as DrawW;
    let h_length_bound = wh.h / CENTER_UI_WIDTH_TILES as DrawH;

    let (raw_bound, tile_side_length, board_x_offset, board_y_offset) = {
        if (w_length_bound - h_length_bound).abs() < 0.5 {
            (h_length_bound, h_length_bound.trunc(), h_length_bound.fract() / 2., h_length_bound.fract() / 2.)
        } else if w_length_bound > h_length_bound {
            (h_length_bound, h_length_bound.trunc(), 0., h_length_bound.fract() / 2.)
        } else if w_length_bound < h_length_bound {
            (w_length_bound, w_length_bound.trunc(), w_length_bound.fract() / 2., 0.)
        } else {
            // NaN ends up here
            // TODO return a Result? Panic? Take only known non-NaN values?
            (100., 100., 0., 0.)
        }
    };

    let play_area_w = raw_bound * DRAW_WIDTH_TILES as PlayW;
    let play_area_h = raw_bound * CENTER_UI_WIDTH_TILES as PlayH;
    let play_area_x = (wh.w - play_area_w) / 2.;
    let play_area_y = (wh.h - play_area_h) / 2.;

    let board_area_w = tile_side_length * CENTER_UI_WIDTH_TILES as BoardW;
    let board_area_h = tile_side_length * CENTER_UI_WIDTH_TILES as BoardH;
    let board_area_x = play_area_x + board_x_offset + (play_area_w - board_area_w) / 2.;
    let board_area_y = play_area_y + board_y_offset + (play_area_h - board_area_h) / 2.;

    Sizes {
        draw_wh: wh,
        play_xywh: PlayXYWH {
            x: play_area_x,
            y: play_area_y,
            w: play_area_w,
            h: play_area_h,
        },
        board_xywh: BoardXYWH {
            x: board_area_x,
            y: board_area_y,
            w: board_area_w,
            h: board_area_h,
        },
        tile_side_length,
    }
}

use crate::{ArrowKind, Dir};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpriteKind {
    NeutralEye,
    DirEye(Dir),
    Arrow(Dir, ArrowKind),
    SmallPupilEye,
    NarrowLeftEye,
    NarrowCenterEye,
    NarrowRightEye,
    ClosedEye,
    HalfLidEye,
}

impl Default for SpriteKind {
    fn default() -> Self {
        Self::NeutralEye
    }
}

#[derive(Debug)]
pub enum Command {
    Sprite(SpriteSpec),
    Text(TextSpec),
}

#[derive(Debug)]
pub struct SpriteSpec {
    pub sprite: SpriteKind,
    pub xy: DrawXY,
}

/// This is provided to make font selection etc. easier for platform layers.
#[derive(Debug)]
pub enum TextKind {
    UI,
}

#[derive(Debug)]
pub struct TextSpec {
    pub text: StrBuf,
    pub xy: DrawXY,
    /// We'd rather define a rectangle for the text to (hopefully) lie inside than
    /// a font size directly.
    pub wh: DrawWH,
    pub kind: TextKind,
}
