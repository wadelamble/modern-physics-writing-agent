from __future__ import annotations

from pathlib import Path

import generate_symmetry_many_slit_paths_phasors_interference_teaser as teaser


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "content"
    / "drafts"
    / "animations"
    / "symmetry-many-slit-paths-phasors-interference-teaser-palette-black-high-pop-preview.png"
)

# Near-black foundation with saturated, restrained accents. The three main
# colors remain functionally distinct: blue for contributions, orange for the
# active path, and green for the resultant.
PALETTE = {
    "BG_TOP": (2, 2, 4),
    "BG_BOTTOM": (16, 6, 12),
    "INK": (255, 242, 228),
    "MUTED": (207, 170, 198),
    "FAINT": (113, 39, 76),
    "CYAN": (70, 190, 255),
    "GOLD": (255, 137, 68),
    "GREEN": (54, 224, 145),
}


def main() -> None:
    for name, color in PALETTE.items():
        setattr(teaser, name, color)
    teaser.draw_text.__defaults__ = (PALETTE["INK"], teaser.LABEL, None)
    teaser.BACKGROUND = teaser.make_background()
    frame_index = round(teaser.DURATION * teaser.FPS) - 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    teaser.draw_frame(frame_index).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
