from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import generate_symmetry_many_slit_paths_phasors_interference_teaser as teaser


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "content" / "drafts" / "animations"
PREFIX = "symmetry-many-slit-teaser-palette"


PALETTES = (
    {
        "key": "01-oxblood-aqua",
        "label": "01  OXBLOOD / AQUA",
        "BG_TOP": (49, 8, 18),
        "BG_BOTTOM": (91, 17, 30),
        "INK": (255, 245, 232),
        "MUTED": (232, 196, 204),
        "FAINT": (142, 64, 80),
        "CYAN": (88, 218, 230),
        "GOLD": (255, 197, 83),
        "GREEN": (116, 231, 177),
    },
    {
        "key": "02-brick-sky",
        "label": "02  BRICK / SKY",
        "BG_TOP": (91, 24, 17),
        "BG_BOTTOM": (145, 48, 29),
        "INK": (255, 247, 232),
        "MUTED": (245, 205, 184),
        "FAINT": (188, 91, 61),
        "CYAN": (162, 212, 232),
        "GOLD": (245, 196, 81),
        "GREEN": (142, 221, 176),
    },
    {
        "key": "03-wine-powder-blue",
        "label": "03  WINE / POWDER BLUE",
        "BG_TOP": (55, 8, 36),
        "BG_BOTTOM": (112, 20, 59),
        "INK": (255, 242, 232),
        "MUTED": (226, 191, 213),
        "FAINT": (145, 66, 113),
        "CYAN": (169, 201, 242),
        "GOLD": (244, 162, 97),
        "GREEN": (113, 214, 166),
    },
    {
        "key": "04-black-cherry-turquoise",
        "label": "04  BLACK CHERRY / TURQUOISE",
        "BG_TOP": (23, 4, 12),
        "BG_BOTTOM": (61, 9, 25),
        "INK": (253, 245, 236),
        "MUTED": (210, 188, 198),
        "FAINT": (99, 49, 68),
        "CYAN": (112, 200, 207),
        "GOLD": (232, 176, 108),
        "GREEN": (101, 201, 154),
    },
    {
        "key": "05-bottle-green-cornflower",
        "label": "05  BOTTLE GREEN / CORNFLOWER",
        "BG_TOP": (8, 42, 32),
        "BG_BOTTOM": (18, 69, 54),
        "INK": (247, 241, 223),
        "MUTED": (191, 207, 194),
        "FAINT": (62, 109, 93),
        "CYAN": (164, 196, 224),
        "GOLD": (242, 158, 116),
        "GREEN": (178, 219, 104),
    },
    {
        "key": "06-deep-teal-sand",
        "label": "06  DEEP TEAL / SAND",
        "BG_TOP": (7, 52, 49),
        "BG_BOTTOM": (12, 87, 80),
        "INK": (255, 244, 223),
        "MUTED": (197, 213, 203),
        "FAINT": (58, 117, 110),
        "CYAN": (233, 194, 139),
        "GOLD": (237, 130, 111),
        "GREEN": (184, 227, 125),
    },
    {
        "key": "07-petrol-salmon",
        "label": "07  PETROL / SALMON",
        "BG_TOP": (17, 53, 60),
        "BG_BOTTOM": (28, 83, 91),
        "INK": (249, 241, 228),
        "MUTED": (200, 209, 206),
        "FAINT": (71, 112, 120),
        "CYAN": (238, 173, 138),
        "GOLD": (244, 208, 111),
        "GREEN": (151, 217, 175),
    },
    {
        "key": "08-forest-lavender",
        "label": "08  FOREST / LAVENDER",
        "BG_TOP": (18, 45, 26),
        "BG_BOTTOM": (39, 71, 45),
        "INK": (247, 240, 218),
        "MUTED": (198, 206, 184),
        "FAINT": (88, 112, 90),
        "CYAN": (200, 185, 232),
        "GOLD": (242, 166, 90),
        "GREEN": (196, 220, 101),
    },
    {
        "key": "09-burnt-orange-ice-blue",
        "label": "09  BURNT ORANGE / ICE BLUE",
        "BG_TOP": (106, 33, 15),
        "BG_BOTTOM": (179, 70, 34),
        "INK": (255, 245, 223),
        "MUTED": (240, 196, 168),
        "FAINT": (195, 109, 71),
        "CYAN": (159, 207, 224),
        "GOLD": (247, 215, 119),
        "GREEN": (132, 207, 161),
    },
    {
        "key": "10-ochre-navy",
        "label": "10  OCHRE / NAVY",
        "BG_TOP": (112, 80, 18),
        "BG_BOTTOM": (175, 125, 28),
        "INK": (33, 25, 12),
        "MUTED": (70, 55, 25),
        "FAINT": (140, 106, 40),
        "CYAN": (39, 61, 101),
        "GOLD": (183, 47, 69),
        "GREEN": (23, 78, 66),
    },
    {
        "key": "11-aubergine-peach",
        "label": "11  AUBERGINE / PEACH",
        "BG_TOP": (48, 16, 47),
        "BG_BOTTOM": (92, 35, 88),
        "INK": (252, 242, 226),
        "MUTED": (217, 192, 213),
        "FAINT": (118, 73, 114),
        "CYAN": (240, 184, 143),
        "GOLD": (232, 200, 86),
        "GREEN": (141, 212, 159),
    },
    {
        "key": "12-plum-celadon",
        "label": "12  PLUM / CELADON",
        "BG_TOP": (68, 17, 62),
        "BG_BOTTOM": (125, 42, 106),
        "INK": (255, 243, 229),
        "MUTED": (228, 195, 217),
        "FAINT": (151, 73, 136),
        "CYAN": (173, 217, 204),
        "GOLD": (255, 156, 116),
        "GREEN": (184, 219, 105),
    },
    {
        "key": "13-rust-slate",
        "label": "13  RUST / SLATE",
        "BG_TOP": (90, 29, 19),
        "BG_BOTTOM": (157, 62, 39),
        "INK": (255, 244, 227),
        "MUTED": (235, 198, 181),
        "FAINT": (171, 86, 61),
        "CYAN": (184, 194, 224),
        "GOLD": (242, 190, 88),
        "GREEN": (150, 213, 169),
    },
    {
        "key": "14-umber-cornflower",
        "label": "14  UMBER / CORNFLOWER",
        "BG_TOP": (40, 26, 20),
        "BG_BOTTOM": (84, 49, 38),
        "INK": (249, 239, 224),
        "MUTED": (207, 190, 177),
        "FAINT": (107, 76, 64),
        "CYAN": (157, 184, 227),
        "GOLD": (237, 154, 97),
        "GREEN": (175, 208, 108),
    },
    {
        "key": "15-parchment-indigo",
        "label": "15  PARCHMENT / INDIGO",
        "BG_TOP": (243, 221, 192),
        "BG_BOTTOM": (232, 185, 135),
        "INK": (48, 35, 27),
        "MUTED": (104, 83, 70),
        "FAINT": (180, 150, 117),
        "CYAN": (43, 86, 128),
        "GOLD": (185, 71, 50),
        "GREEN": (36, 99, 79),
    },
    {
        "key": "16-sage-aubergine",
        "label": "16  SAGE / AUBERGINE",
        "BG_TOP": (215, 223, 201),
        "BG_BOTTOM": (183, 201, 176),
        "INK": (35, 38, 31),
        "MUTED": (86, 97, 83),
        "FAINT": (139, 160, 133),
        "CYAN": (104, 79, 134),
        "GOLD": (180, 106, 43),
        "GREEN": (31, 104, 76),
    },
)


def font(size: int, bold: bool = False):
    candidates = (
        "seguisb.ttf" if bold else "segoeui.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def set_palette(palette: dict[str, object]) -> None:
    for name in ("BG_TOP", "BG_BOTTOM", "INK", "MUTED", "FAINT", "CYAN", "GOLD", "GREEN"):
        setattr(teaser, name, palette[name])
    teaser.draw_text.__defaults__ = (palette["INK"], teaser.LABEL, None)
    teaser.BACKGROUND = teaser.make_background()


def render_options() -> list[tuple[dict[str, object], Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_index = round(teaser.DURATION * teaser.FPS) - 1
    outputs: list[tuple[dict[str, object], Path]] = []
    for palette in PALETTES:
        set_palette(palette)
        path = OUTPUT_DIR / f"{PREFIX}-{palette['key']}.png"
        teaser.draw_frame(frame_index).save(path)
        outputs.append((palette, path))
    return outputs


def make_sheet(
    outputs: list[tuple[dict[str, object], Path]],
    *,
    columns: int,
    thumb_width: int,
    label_height: int,
    margin: int,
    filename: str,
) -> Path:
    rows = (len(outputs) + columns - 1) // columns
    thumb_height = round(thumb_width * teaser.HEIGHT / teaser.WIDTH)
    sheet_width = columns * thumb_width + (columns + 1) * margin
    sheet_height = rows * (label_height + thumb_height) + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_width, sheet_height), (18, 14, 15))
    draw = ImageDraw.Draw(sheet)
    for index, (palette, path) in enumerate(outputs):
        column = index % columns
        row = index // columns
        x = margin + column * (thumb_width + margin)
        y = margin + row * (label_height + thumb_height + margin)
        label_size = 18 if columns == 4 else 25
        label_font = font(label_size, True)
        label = str(palette["label"])
        while label_size > 12 and draw.textbbox((0, 0), label, font=label_font)[2] > thumb_width - 8:
            label_size -= 1
            label_font = font(label_size, True)
        draw.text(
            (x + thumb_width / 2, y + label_height / 2),
            label,
            fill=(248, 239, 229),
            font=label_font,
            anchor="mm",
        )
        image = Image.open(path).convert("RGB")
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + label_height))

    output = OUTPUT_DIR / filename
    sheet.save(output)
    return output


def main() -> None:
    outputs = render_options()
    overview = make_sheet(
        outputs,
        columns=4,
        thumb_width=260,
        label_height=52,
        margin=20,
        filename=f"{PREFIX}s-overview-4x4.png",
    )
    detail_sheets = []
    for index, name in enumerate(("reds", "greens-teals", "earths-plums", "light-and-reserved")):
        detail_sheets.append(
            make_sheet(
                outputs[index * 4 : (index + 1) * 4],
                columns=2,
                thumb_width=500,
                label_height=66,
                margin=28,
                filename=f"{PREFIX}s-detail-{name}.png",
            )
        )
    for _, path in outputs:
        print(path)
    print(overview)
    for path in detail_sheets:
        print(path)


if __name__ == "__main__":
    main()
