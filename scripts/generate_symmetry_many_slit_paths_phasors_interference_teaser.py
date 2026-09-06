from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_symmetry_many_slit_paths_phasors_interference as model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "content" / "drafts" / "animations"
NAME = "symmetry-many-slit-paths-phasors-interference-teaser"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 15.0

# High-pop black reel palette.
BG_TOP = (2, 2, 4)
BG_BOTTOM = (16, 6, 12)
INK = (255, 242, 228)
MUTED = (207, 170, 198)
FAINT = (113, 39, 76)
CYAN = (70, 190, 255)
GOLD = (255, 137, 68)
GREEN = (54, 224, 145)

# Reel-safe vertical layout. Important information stays above the area usually
# covered by social-app captions and playback controls.
ROUTE_TITLE_Y = 55
ROUTE_PROGRESS_Y = 57
ROUTE_A = (86.0, 467.0)
ROUTE_SCREEN_X = 526.0
ROUTE_DETECTOR_X = 985.0
ROUTE_TOP = 286.0
ROUTE_BOTTOM = 647.0
ROUTE_CENTER_Y = (ROUTE_TOP + ROUTE_BOTTOM) / 2.0
ROUTE_Y_SCALE = (ROUTE_BOTTOM - ROUTE_TOP) / (2.0 * model.DETECTOR_HALF_HEIGHT)

PHASOR_TITLE_Y = 735
PHASOR_BOUNDS = (78.0, 807.0, 1002.0, 1247.0)

DETECTOR_TITLE_Y = 1370
DETECTOR_TOP = 1455.0
DETECTOR_BOTTOM = 1664.0
# Match the desktop figure's horizontal-intensity/vertical-position proportion:
# (1218 - 1065) / (568 - 190) ~= 0.405.
DETECTOR_TRACE_SPAN = 0.405 * (DETECTOR_BOTTOM - DETECTOR_TOP)
# Keep the detector-position axis where the earlier compact plot began, while
# placing the contrast-expanded interference trace in its right-hand band.
DETECTOR_AXIS_X = 497.6775
DETECTOR_TRACE_MIN_X = 675.355
DETECTOR_TRACE_MAX_X = 760.0
# The physical calculation has a shallow but real change in crest heights.
# Contrast-stretch the *single computed intensity curve* over its displayed
# detector interval so that variation remains legible in the compact reel plot.
# A single monotone gamma curve then separates the closely spaced high crests.
# Neither operation changes the computed order or location of any crest, and
# neither makes a per-crest adjustment.
DETECTOR_INTENSITY_FLOOR = float(model.DETECTOR_INTENSITIES.min())
DETECTOR_DISPLAY_GAMMA = 2.0

# A 15-second cut with the explanatory motion retained and idle time removed.
INTRO_END = 0.5
SLOW_BUILD_END = 2.2
BUILD_END = 5.8
CENTER_HOLD_END = 6.9
MOVE_TO_TOP_END = 7.8
SCAN_END = 13.6
SETTLE_END = 14.3


def font(size: int, bold: bool = False):
    candidates = [
        "seguisb.ttf" if bold else "segoeui.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


SECTION = font(28, True)
LABEL = font(25)
LABEL_BOLD = font(26, True)
SMALL = font(21)


def rgba(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], max(0, min(255, round(255 * alpha)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def smootherstep(value: float) -> float:
    value = clamp01(value)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def draw_text(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    text: str,
    fill=INK,
    font_obj=LABEL,
    anchor: str | None = None,
) -> None:
    draw.text(point, text, fill=fill, font=font_obj, anchor=anchor)


def make_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        amount = y / max(1, HEIGHT - 1)
        color = tuple(round(lerp(BG_TOP[i], BG_BOTTOM[i], amount)) for i in range(3))
        draw.line((0, y, WIDTH, y), fill=color)
    return image.convert("RGBA")


BACKGROUND = make_background()


def route_y(value: float) -> float:
    return ROUTE_CENTER_Y - ROUTE_Y_SCALE * value


def detector_y(value: float) -> float:
    amount = (model.DETECTOR_HALF_HEIGHT - value) / (2.0 * model.DETECTOR_HALF_HEIGHT)
    return DETECTOR_TOP + amount * (DETECTOR_BOTTOM - DETECTOR_TOP)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    radius: float,
    fill,
    outline=None,
    width: int = 2,
) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill,
    width: int,
    head: float,
) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * math.cos(angle - math.pi / 6.0),
        end[1] - head * math.sin(angle - math.pi / 6.0),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6.0),
        end[1] - head * math.sin(angle + math.pi / 6.0),
    )
    draw.polygon((end, left, right), fill=fill)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill,
    width: int = 3,
    dash: float = 11.0,
    gap: float = 8.0,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    distance = 0.0
    while distance < length:
        stop = min(length, distance + dash)
        q0 = distance / length
        q1 = stop / length
        draw.line(
            (
                start[0] + q0 * dx,
                start[1] + q0 * dy,
                start[0] + q1 * dx,
                start[1] + q1 * dy,
            ),
            fill=fill,
            width=width,
        )
        distance += dash + gap


def draw_route(
    draw: ImageDraw.ImageDraw,
    opening_y: float,
    b_value: float,
    fill,
    width: int,
    fraction: float = 1.0,
) -> None:
    start = ROUTE_A
    middle = (ROUTE_SCREEN_X, route_y(opening_y))
    finish = (ROUTE_DETECTOR_X, route_y(b_value))
    first_length = math.dist(start, middle)
    second_length = math.dist(middle, finish)
    total_length = first_length + second_length
    travel = clamp01(fraction) * total_length
    if travel <= first_length:
        amount = travel / max(first_length, 1e-9)
        partial = (lerp(start[0], middle[0], amount), lerp(start[1], middle[1], amount))
        draw.line((*start, *partial), fill=fill, width=width)
        return
    draw.line((*start, *middle), fill=fill, width=width)
    amount = (travel - first_length) / max(second_length, 1e-9)
    partial = (lerp(middle[0], finish[0], amount), lerp(middle[1], finish[1], amount))
    draw.line((*middle, *partial), fill=fill, width=width)


def make_phasor_mapper():
    values: list[complex] = [0j]
    for b_value in np.linspace(model.DETECTOR_HALF_HEIGHT, -model.DETECTOR_HALF_HEIGHT, 121):
        values.extend(model.cumulative_values(model.build_contributions(float(b_value))))
    min_x = min(value.real for value in values)
    max_x = max(value.real for value in values)
    min_y = min(value.imag for value in values)
    max_y = max(value.imag for value in values)
    left, top, right, bottom = PHASOR_BOUNDS
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min((right - left - 46.0) / span_x, (bottom - top - 46.0) / span_y)
    offset_x = (left + right) / 2.0 - scale * (min_x + max_x) / 2.0
    offset_y = (top + bottom) / 2.0 + scale * (min_y + max_y) / 2.0

    def mapper(value: complex) -> tuple[float, float]:
        return offset_x + scale * value.real, offset_y - scale * value.imag

    return mapper


MAP_PHASOR = make_phasor_mapper()


def draw_dividers(draw: ImageDraw.ImageDraw) -> None:
    draw.line((54, 690, 1026, 690), fill=rgba(FAINT, 0.74), width=2)
    draw.line((54, 1322, 1026, 1322), fill=rgba(FAINT, 0.74), width=2)


def draw_routes(
    draw: ImageDraw.ImageDraw,
    b_value: float,
    completed_count: int,
    active_index: int | None,
    active_fraction: float,
    complete: bool,
) -> None:
    draw_text(draw, (55, ROUTE_TITLE_Y), "Accumulate phase along each path", font_obj=SECTION)

    contributions = model.build_contributions(b_value)
    count = model.OPENING_COUNT if complete else completed_count
    for index in range(count):
        draw_route(draw, contributions[index].opening_y, b_value, rgba(CYAN, 0.19), 3)

    draw.line(
        (ROUTE_SCREEN_X, ROUTE_TOP, ROUTE_SCREEN_X, ROUTE_BOTTOM),
        fill=rgba(INK, 0.88),
        width=7,
    )
    for opening_y in model.OPENING_YS:
        draw_circle(draw, (ROUTE_SCREEN_X, route_y(float(opening_y))), 3.6, BG_TOP)
    draw_text(
        draw,
        (ROUTE_SCREEN_X, ROUTE_TOP - 44),
        "barrier with many narrow slits",
        fill=MUTED,
        font_obj=SMALL,
        anchor="ma",
    )
    draw.line(
        (ROUTE_DETECTOR_X, ROUTE_TOP, ROUTE_DETECTOR_X, ROUTE_BOTTOM),
        fill=rgba(MUTED, 0.65),
        width=3,
    )
    target = (ROUTE_DETECTOR_X, route_y(b_value))
    draw_circle(draw, target, 9.0, INK)
    draw_text(draw, (target[0] - 17, target[1]), "B", font_obj=LABEL_BOLD, anchor="rm")

    stationary_index = min(
        range(model.OPENING_COUNT),
        key=lambda index: abs(
            contributions[index].opening_y - model.stationary_opening_y(b_value)
        ),
    )
    if complete:
        stationary = contributions[stationary_index]
        draw_route(draw, stationary.opening_y, b_value, rgba(GOLD, 0.96), 7)
        draw_circle(draw, (ROUTE_SCREEN_X, route_y(stationary.opening_y)), 7.0, GOLD)

    if active_index is not None:
        active = contributions[active_index]
        eased_fraction = smootherstep(active_fraction)
        draw_route(draw, active.opening_y, b_value, GOLD, 7, eased_fraction)
        draw_circle(draw, (ROUTE_SCREEN_X, route_y(active.opening_y)), 7.0, GOLD)
        draw_text(
            draw,
            (1018, ROUTE_PROGRESS_Y),
            f"{active_index + 1} / 49",
            fill=GOLD,
            font_obj=LABEL_BOLD,
            anchor="ra",
        )
    elif complete:
        draw_text(draw, (1018, ROUTE_PROGRESS_Y), "all 49", fill=GREEN, font_obj=LABEL_BOLD, anchor="ra")

    draw_circle(draw, ROUTE_A, 9.0, INK)
    draw_text(draw, (ROUTE_A[0] + 18, ROUTE_A[1]), "A", font_obj=LABEL_BOLD, anchor="lm")


def draw_phasors(
    draw: ImageDraw.ImageDraw,
    b_value: float,
    completed_count: int,
    active_index: int | None,
    active_fraction: float,
    complete: bool,
) -> complex:
    draw_text(draw, (55, PHASOR_TITLE_Y), "Sum the amplitudes from each path in the complex plane", font_obj=SECTION)
    contributions = model.build_contributions(b_value)
    cumulative = model.cumulative_values(contributions)
    count = model.OPENING_COUNT if complete else completed_count
    origin = MAP_PHASOR(0j)

    draw.line(
        (PHASOR_BOUNDS[0], origin[1], PHASOR_BOUNDS[2], origin[1]),
        fill=rgba(MUTED, 0.22),
        width=2,
    )
    draw.line(
        (origin[0], PHASOR_BOUNDS[1], origin[0], PHASOR_BOUNDS[3]),
        fill=rgba(MUTED, 0.22),
        width=2,
    )
    for index in range(count):
        draw_arrow(
            draw,
            MAP_PHASOR(cumulative[index]),
            MAP_PHASOR(cumulative[index + 1]),
            rgba(CYAN, 0.86),
            3,
            6.0,
        )

    current = cumulative[count]
    if active_index is not None:
        start_value = cumulative[active_index]
        current = start_value + smootherstep(active_fraction) * contributions[active_index].value
        draw_arrow(
            draw,
            MAP_PHASOR(start_value),
            MAP_PHASOR(current),
            GOLD,
            7,
            11.0,
        )

    if count > 0 or active_index is not None:
        dashed_line(draw, origin, MAP_PHASOR(current), rgba(GREEN, 0.42))

    if complete:
        total = cumulative[-1]
        draw_arrow(draw, origin, MAP_PHASOR(total), GREEN, 10, 17.0)
        draw_text(
            draw,
            ((PHASOR_BOUNDS[0] + PHASOR_BOUNDS[2]) / 2.0, 1277),
            "green arrow = total amplitude at B",
            fill=GREEN,
            font_obj=LABEL_BOLD,
            anchor="ma",
        )
        return total

    if count > 0 or active_index is not None:
        draw_circle(draw, MAP_PHASOR(current), 5.0, GOLD)
    return current


def detector_intensity_x(intensity: float) -> float:
    intensity_span = max(1e-12, model.MAX_INTENSITY - DETECTOR_INTENSITY_FLOOR)
    normalized = clamp01((intensity - DETECTOR_INTENSITY_FLOOR) / intensity_span)
    amount = normalized**DETECTOR_DISPLAY_GAMMA
    return DETECTOR_TRACE_MIN_X + amount * (DETECTOR_TRACE_MAX_X - DETECTOR_TRACE_MIN_X)


def draw_detector(
    draw: ImageDraw.ImageDraw,
    b_value: float,
    current_total: complex,
    reveal_fraction: float,
    scanning: bool,
    trace_complete: bool,
) -> None:
    draw_text(draw, (55, DETECTOR_TITLE_Y), "Square the magnitude to obtain the interference pattern", font_obj=SECTION)
    draw.line(
        (DETECTOR_AXIS_X, DETECTOR_TOP, DETECTOR_AXIS_X, DETECTOR_BOTTOM),
        fill=rgba(MUTED, 0.62),
        width=3,
    )

    if scanning or trace_complete:
        visited = (
            len(model.DETECTOR_B_VALUES)
            if trace_complete
            else max(1, round(clamp01(reveal_fraction) * len(model.DETECTOR_B_VALUES)))
        )
        points: list[tuple[float, float]] = []
        for index, (sample_b, intensity) in enumerate(zip(model.DETECTOR_B_VALUES, model.DETECTOR_INTENSITIES)):
            y = detector_y(float(sample_b))
            if index < visited:
                x = detector_intensity_x(float(intensity))
                points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=rgba(CYAN, 0.94), width=5)

    if scanning:
        intensity = abs(current_total) ** 2
        marker_x = detector_intensity_x(intensity)
        marker = (marker_x, detector_y(b_value))
        draw.line((DETECTOR_AXIS_X, marker[1], marker[0], marker[1]), fill=rgba(GOLD, 0.82), width=5)
        draw_circle(draw, marker, 8.0, GREEN if trace_complete else GOLD)
    draw_text(draw, (DETECTOR_AXIS_X - 18, DETECTOR_BOTTOM + 32), "detector position", fill=MUTED, font_obj=SMALL, anchor="ra")
    draw_text(draw, (DETECTOR_TRACE_MAX_X + 18, DETECTOR_BOTTOM + 32), "brighter →", fill=MUTED, font_obj=SMALL, anchor="la")


def build_progress(seconds: float) -> float:
    if seconds <= INTRO_END:
        return 0.0
    if seconds < SLOW_BUILD_END:
        amount = smootherstep((seconds - INTRO_END) / (SLOW_BUILD_END - INTRO_END))
        return 5.0 * amount
    amount = smootherstep((seconds - SLOW_BUILD_END) / (BUILD_END - SLOW_BUILD_END))
    return 5.0 + (model.OPENING_COUNT - 5.0) * amount


def animation_state(seconds: float):
    if seconds < INTRO_END:
        return 0.0, 0, None, 0.0, False, 0.0, False

    if seconds < BUILD_END:
        raw = min(model.OPENING_COUNT - 1e-8, build_progress(seconds))
        completed = int(math.floor(raw))
        active = min(model.OPENING_COUNT - 1, completed)
        fraction = raw - completed
        return 0.0, completed, active, fraction, False, 0.0, False

    if seconds < CENTER_HOLD_END:
        return 0.0, model.OPENING_COUNT, None, 0.0, True, 0.0, False

    if seconds < MOVE_TO_TOP_END:
        amount = smootherstep((seconds - CENTER_HOLD_END) / (MOVE_TO_TOP_END - CENTER_HOLD_END))
        b_value = lerp(0.0, model.DETECTOR_HALF_HEIGHT, amount)
        return b_value, model.OPENING_COUNT, None, 0.0, True, 0.0, False

    if seconds < SCAN_END:
        amount = smootherstep((seconds - MOVE_TO_TOP_END) / (SCAN_END - MOVE_TO_TOP_END))
        b_value = lerp(model.DETECTOR_HALF_HEIGHT, -model.DETECTOR_HALF_HEIGHT, amount)
        return b_value, model.OPENING_COUNT, None, 0.0, True, amount, True

    if seconds < SETTLE_END:
        amount = smootherstep((seconds - SCAN_END) / (SETTLE_END - SCAN_END))
        b_value = lerp(-model.DETECTOR_HALF_HEIGHT, model.SELECTED_B, amount)
        return b_value, model.OPENING_COUNT, None, 0.0, True, 1.0, True

    return model.SELECTED_B, model.OPENING_COUNT, None, 0.0, True, 1.0, True


def draw_frame(frame: int) -> Image.Image:
    seconds = min(DURATION - 1.0 / FPS, frame / FPS)
    (
        b_value,
        completed_count,
        active_index,
        active_fraction,
        complete,
        reveal_fraction,
        scanning,
    ) = animation_state(seconds)

    image = BACKGROUND.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    draw_dividers(draw)
    draw_routes(draw, b_value, completed_count, active_index, active_fraction, complete)
    current_total = draw_phasors(draw, b_value, completed_count, active_index, active_fraction, complete)
    draw_detector(
        draw,
        b_value,
        current_total,
        reveal_fraction,
        scanning,
        seconds >= SCAN_END,
    )
    return image.convert("RGB")


def verify(path: Path) -> str:
    result = subprocess.run(
        [
            str(model.wave.FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def encode() -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video = OUTPUT_DIR / f"{NAME}.mp4"
    contact = OUTPUT_DIR / f"{NAME}-contact-sheet.png"
    final_still = OUTPUT_DIR / f"{NAME}-final.png"

    process = subprocess.Popen(
        [
            str(model.wave.FFMPEG),
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
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for index in range(round(DURATION * FPS)):
            frame = draw_frame(index)
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0 or not video.exists() or video.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed for {NAME}")

    draw_frame(round(DURATION * FPS) - 1).save(final_still)

    samples = (0.2, 1.5, 5.7, 7.3, 11.0, 14.6)
    thumb_width = 180
    thumb_height = 320
    margin = 18
    sheet = Image.new(
        "RGB",
        (3 * thumb_width + 4 * margin, 2 * thumb_height + 3 * margin),
        BG_TOP,
    )
    for index, seconds in enumerate(samples):
        frame_index = min(round(DURATION * FPS) - 1, round(seconds * FPS))
        thumb = draw_frame(frame_index).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        column = index % 3
        row = index // 3
        x = margin + column * (thumb_width + margin)
        y = margin + row * (thumb_height + margin)
        sheet.paste(thumb, (x, y))
    sheet.save(contact)
    return video, contact, final_still


def main() -> None:
    video, contact, final_still = encode()
    print(video)
    print(contact)
    print(final_still)
    print(verify(video))


if __name__ == "__main__":
    main()
