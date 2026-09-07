from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_symmetry_many_slit_paths_phasors_interference as geometry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "content" / "drafts" / "animations"
NAME = "symmetry-many-slit-wave-field-preview"

WIDTH = 1080
HEIGHT = 1080
FPS = 30
DURATION = 7.0
FRAME_COUNT = round(FPS * DURATION)
WAVE_CYCLES = 2.0

# One physically proportioned world-coordinate view.  Keeping equal x/y scale
# makes the outgoing circular fronts remain circular in the rendered panel.
WORLD_X_MIN = -5.4
WORLD_X_MAX = 5.4
WORLD_Y_MIN = -3.5
WORLD_Y_MAX = 3.5
SOURCE = (-geometry.SOURCE_DISTANCE, 0.0)
BARRIER_X = 0.0

PLOT_LEFT = 40
PLOT_TOP = 190
PLOT_WIDTH = 1000
PLOT_HEIGHT = round(PLOT_WIDTH * (WORLD_Y_MAX - WORLD_Y_MIN) / (WORLD_X_MAX - WORLD_X_MIN))
PLOT_RIGHT = PLOT_LEFT + PLOT_WIDTH
PLOT_BOTTOM = PLOT_TOP + PLOT_HEIGHT

# Calculate the field on a moderately oversampled grid and use a high-quality
# resize.  The 0.5-unit wavelength remains more than thirty source pixels wide,
# comfortably above the aliasing limit.
FIELD_WIDTH = 720
FIELD_HEIGHT = round(FIELD_WIDTH * (WORLD_Y_MAX - WORLD_Y_MIN) / (WORLD_X_MAX - WORLD_X_MIN))

BG = (6, 7, 16)
NEUTRAL = np.asarray((9.0, 11.0, 25.0))
POSITIVE = np.asarray((255.0, 55.0, 84.0))
NEGATIVE = np.asarray((34.0, 126.0, 255.0))
INK = (244, 239, 245)
MUTED = (188, 182, 199)
BARRIER = (235, 230, 241)
SOURCE_COLOR = (255, 190, 96)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


TITLE_FONT = font(34, True)
LABEL_FONT = font(24, True)
SMALL_FONT = font(21)


def world_to_pixel(x_value: float, y_value: float) -> tuple[float, float]:
    px = PLOT_LEFT + (x_value - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * PLOT_WIDTH
    py = PLOT_TOP + (WORLD_Y_MAX - y_value) / (WORLD_Y_MAX - WORLD_Y_MIN) * PLOT_HEIGHT
    return px, py


def compute_complex_field() -> np.ndarray:
    x_values = np.linspace(WORLD_X_MIN, WORLD_X_MAX, FIELD_WIDTH)
    y_values = np.linspace(WORLD_Y_MAX, WORLD_Y_MIN, FIELD_HEIGHT)
    xx, yy = np.meshgrid(x_values, y_values)

    source_radius = np.hypot(xx - SOURCE[0], yy - SOURCE[1])
    source_radius_soft = np.sqrt(source_radius**2 + 0.06**2)
    source_spreading = 1.0 / np.sqrt(source_radius_soft)
    source_ramp = 1.0 - np.exp(-((source_radius / 0.14) ** 2))
    incident = source_spreading * source_ramp * np.exp(1j * geometry.WAVE_NUMBER * source_radius)

    transmitted = np.zeros_like(xx, dtype=np.complex128)
    opening_spacing = abs(float(geometry.OPENING_YS[1] - geometry.OPENING_YS[0]))
    for opening_y in geometry.OPENING_YS:
        opening_y = float(opening_y)
        source_to_opening = math.hypot(geometry.SOURCE_DISTANCE, opening_y)
        opening_to_point = np.hypot(xx - BARRIER_X, yy - opening_y)
        opening_to_point_soft = np.sqrt(opening_to_point**2 + 0.07**2)
        spreading = opening_spacing / np.sqrt(source_to_opening * opening_to_point_soft)
        transmitted += spreading * np.exp(
            1j * geometry.WAVE_NUMBER * (source_to_opening + opening_to_point)
        )

    # Use one fixed robust scale on each side for every animation frame.  This
    # keeps the far field legible without changing its phase or interference
    # structure and without frame-to-frame brightness pumping.
    left_mask = (xx < BARRIER_X) & (source_radius > 0.18)
    right_mask = xx > BARRIER_X + 0.08
    left_scale = float(np.percentile(np.abs(incident[left_mask]), 98.5))
    right_scale = float(np.percentile(np.abs(transmitted[right_mask]), 99.0))
    incident /= max(left_scale, 1e-12)
    transmitted /= max(right_scale, 1e-12)

    return np.where(xx < BARRIER_X, incident, transmitted)


COMPLEX_FIELD = compute_complex_field()


def signed_field_image(phase: float) -> Image.Image:
    values = np.real(COMPLEX_FIELD * np.exp(-1j * phase))
    normalized = np.clip(values, -1.0, 1.0)
    strength = np.abs(normalized) ** 0.72
    target = np.where((normalized >= 0.0)[..., None], POSITIVE, NEGATIVE)
    rgb = NEUTRAL + (target - NEUTRAL) * strength[..., None]
    field = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return field.resize((PLOT_WIDTH, PLOT_HEIGHT), Image.Resampling.BICUBIC)


def draw_centered(draw: ImageDraw.ImageDraw, y: float, text: str, *, fill, font_obj) -> None:
    box = draw.textbbox((0, 0), text, font=font_obj)
    draw.text(((WIDTH - (box[2] - box[0])) / 2.0, y), text, fill=fill, font=font_obj)


def draw_barrier(draw: ImageDraw.ImageDraw) -> None:
    barrier_x, _ = world_to_pixel(BARRIER_X, 0.0)
    opening_pixels = sorted(world_to_pixel(BARRIER_X, float(y))[1] for y in geometry.OPENING_YS)
    gap_half_height = 2.6
    cursor = float(PLOT_TOP)
    for opening_y in opening_pixels:
        segment_end = opening_y - gap_half_height
        if segment_end > cursor:
            draw.line((barrier_x, cursor, barrier_x, segment_end), fill=BARRIER, width=9)
        cursor = opening_y + gap_half_height
    if cursor < PLOT_BOTTOM:
        draw.line((barrier_x, cursor, barrier_x, PLOT_BOTTOM), fill=BARRIER, width=9)


def draw_frame_for_phase(phase: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    image.paste(signed_field_image(phase), (PLOT_LEFT, PLOT_TOP))
    draw = ImageDraw.Draw(image, "RGBA")

    draw_centered(draw, 57, "Many-slit propagation as a wave field", fill=INK, font_obj=TITLE_FONT)
    draw_centered(
        draw,
        112,
        "red = positive amplitude    blue = negative amplitude    dark = near zero",
        fill=MUTED,
        font_obj=SMALL_FONT,
    )

    draw_barrier(draw)
    barrier_px, _ = world_to_pixel(BARRIER_X, 0.0)
    draw_centered(draw, 153, "49 sampled openings", fill=MUTED, font_obj=SMALL_FONT)

    source_px = world_to_pixel(*SOURCE)
    radius = 8
    draw.ellipse(
        (source_px[0] - radius, source_px[1] - radius, source_px[0] + radius, source_px[1] + radius),
        fill=SOURCE_COLOR,
        outline=INK,
        width=2,
    )
    draw.text((source_px[0] - 34, source_px[1] + 15), "A", fill=INK, font=LABEL_FONT)

    # Small end marks keep the barrier visually distinct from the changing
    # scalar field without adding diagrammatic machinery.
    draw.line((barrier_px - 11, PLOT_TOP, barrier_px + 11, PLOT_TOP), fill=BARRIER, width=3)
    draw.line((barrier_px - 11, PLOT_BOTTOM, barrier_px + 11, PLOT_BOTTOM), fill=BARRIER, width=3)
    return image


def draw_video_frame(frame_index: int) -> Image.Image:
    phase = 2.0 * math.pi * WAVE_CYCLES * frame_index / FRAME_COUNT
    return draw_frame_for_phase(phase)


def verify(path: Path) -> str:
    result = subprocess.run(
        [
            str(geometry.wave.FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def render() -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / f"{NAME}.mp4"
    still_path = OUTPUT_DIR / f"{NAME}-still.png"
    contact_path = OUTPUT_DIR / f"{NAME}-contact-sheet.png"

    process = subprocess.Popen(
        [
            str(geometry.wave.FFMPEG),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for frame_index in range(FRAME_COUNT):
            frame = draw_video_frame(frame_index)
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0 or not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed for {NAME}")

    # A phase with strong simultaneous structure on both sides makes the still
    # more useful than merely duplicating the video's final frame.
    draw_frame_for_phase(0.34 * 2.0 * math.pi).save(still_path)

    phases = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    thumb_size = 340
    margin = 15
    contact = Image.new("RGB", (1080, 725), BG)
    for index, phase in enumerate(phases):
        thumb = draw_frame_for_phase(float(phase)).resize(
            (thumb_size, thumb_size), Image.Resampling.LANCZOS
        )
        row, column = divmod(index, 3)
        x = margin + column * (thumb_size + margin)
        y = margin + row * (thumb_size + margin)
        contact.paste(thumb, (x, y))
    contact.save(contact_path)
    return video_path, still_path, contact_path


def main() -> None:
    video_path, still_path, contact_path = render()
    print(video_path)
    print(still_path)
    print(contact_path)
    print(verify(video_path))


if __name__ == "__main__":
    main()
