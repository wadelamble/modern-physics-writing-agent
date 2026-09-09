"""Render the progressive many-slit / many-slice desktop animation.

Run with Python and requirements-progressive-animation.txt installed. The
optional repository-local .tools/animation-python-packages directory is used
when present. Existing desktop assets are left intact.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".tools" / "animation-python-packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from matplotlib import mathtext, rc_context
from matplotlib.font_manager import FontProperties
from progressive_paths_model import FresnelModel, DETECTOR_B_VALUES, run_checks

OUT = ROOT / "content" / "drafts" / "animations"
NAME = "symmetry-progressive-slits-to-path-integral"
WIDTH, HEIGHT, SCALE, FPS = 1280, 720, 2, 24
DURATION = 64.0
BG = (255, 252, 246)
PANEL = (252, 248, 240)
INK = (37, 39, 42)
MUTED = (103, 100, 95)
FAINT = (221, 214, 204)
BLUE = (51, 91, 133)
GOLD = (198, 138, 45)
GREEN = (65, 126, 95)
ROTATE = np.exp(0.65j)
ROUTE = (35, 122, 585, 612)
PHASOR = (602, 122, 1015, 612)
DETECTOR = (1032, 122, 1245, 612)
YMAX = 3.2
X0, X1, YMID, YSCALE = 82.0, 551.0, 377.0, 54.0
PBOX = (627, 212, 987, 547)
DX0, DX1, DY0, DY1 = 1064.0, 1215.0, 219.0, 557.0
MODELS = {n: FresnelModel(n) for n in (385, 769, 1537)}
PROFILE = {(n, ap): m.profile() for n, m in MODELS.items() for ap in (False, True)}
IMAX = max(float(np.max(np.abs(v) ** 2)) for v in PROFILE.values()) * 1.06
SAMPLES = (1.0, 4.0, 8.5, 12.0, 15.5, 18.5, 22.0, 26.0,
           31.0, 36.0, 41.0, 46.0, 50.0, 54.0, 58.0, 62.5)


def s(v):
    return round(float(v) * SCALE)


def ease(v):
    v = np.clip(v, 0.0, 1.0)
    return float(v * v * (3.0 - 2.0 * v))


def mix(a, b, q):
    return a + (b - a) * q


def shade(color, amount):
    return tuple(round(mix(PANEL[i], color[i], amount)) for i in range(3))


@lru_cache(None)
def font(size, bold=False):
    for name in ("seguisb.ttf" if bold else "segoeui.ttf",
                 "arialbd.ttf" if bold else "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, s(size))
        except OSError:
            pass
    return ImageFont.truetype(str(LOCAL_PACKAGES / "matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"), s(size))


def text(draw, xy, label, size=15, color=INK, bold=False, anchor=None):
    draw.text((s(xy[0]), s(xy[1])), label, font=font(size, bold), fill=color, anchor=anchor)


def line(draw, points, color=BLUE, width=1):
    draw.line([(s(x), s(y)) for x, y in points], fill=color, width=max(1, s(width)), joint="curve")


def dot(draw, xy, radius=3, color=INK):
    x, y = xy
    draw.ellipse((s(x-radius), s(y-radius), s(x+radius), s(y+radius)), fill=color)


def arrow(draw, start, finish, color, width=3, head=7):
    line(draw, (start, finish), color, width)
    dx, dy = finish[0]-start[0], finish[1]-start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    head = min(head, 0.6 * length)
    ux, uy = dx/length, dy/length
    pts = [finish, (finish[0]-head*ux+head*.42*uy, finish[1]-head*uy-head*.42*ux),
           (finish[0]-head*ux-head*.42*uy, finish[1]-head*uy+head*.42*ux)]
    draw.polygon([(s(x), s(y)) for x, y in pts], fill=color)


def dashed(draw, x, color, width=1, top=204, bottom=550):
    for y in range(top, bottom, 10):
        line(draw, ((x, y), (x, min(y+5, bottom))), color, width)


@lru_cache(None)
def equation(label, size=22):
    stream = io.BytesIO()
    with rc_context({"mathtext.fontset": "stix", "font.family": "STIXGeneral", "savefig.transparent": True}):
        mathtext.math_to_image(label, stream, dpi=144, format="png", color="#25272a",
                              prop=FontProperties(size=size))
    stream.seek(0)
    return Image.open(stream).convert("RGBA")


def place_equation(image, label, y=648, size=22, maxwidth=1150):
    tile = equation(label, size).copy()
    if tile.width > s(maxwidth):
        tile = tile.resize((s(maxwidth), round(tile.height * s(maxwidth)/tile.width)), Image.Resampling.LANCZOS)
    image.alpha_composite(tile, ((image.width-tile.width)//2, s(y)-tile.height//2))


def state(t):
    b = .94
    n, slices, old, refinement = 385, 1, 1, 1.0
    if 8 <= t < 9:
        b = mix(.94, 3.2, ease(t-8))
    elif 9 <= t < 13:
        b = mix(3.2, -3.2, (t-9)/4)
    elif 13 <= t < 14:
        b = mix(-3.2, .94, ease(t-13))
    elif 52 <= t < 58:
        b = mix(3.2, -3.2, (t-52)/6)
    elif 50 <= t < 52:
        b = mix(.94, 3.2, ease((t-50)/2))
    elif 58 <= t < 60:
        b = mix(-3.2, .94, ease((t-58)/2))
    release = ease((t-14)/4)
    for start, prev, target, mesh in ((20,1,3,385), (25,3,7,385), (30,7,15,769),
                                     (35,15,31,769), (40,31,63,1537)):
        if t >= start:
            n, old, slices, refinement = mesh, prev, target, ease((t-start)/2.5)
    return b, n, slices, old, refinement, release


def caption(t):
    if t < 8:
        return "One screen, 49 openings", "Add the contribution through each opening."
    if t < 14:
        return "One screen, 49 openings", "Move B: the phases change, and so does the intensity."
    if t < 18:
        return "Remove the barrier; retain an imaginary slice", "Every intermediate point can now contribute."
    if t < 20:
        return "One imaginary slice", "The slice organizes the calculation. It introduces no obstacle."
    if t < 30:
        return "More slices, more possible chains", "Choose a crossing on each slice; multiply along the chain, then sum."
    if t < 45:
        return "Refine the slices and their points", "The number of path terms grows. The resulting field stays the same."
    if t < 50:
        return "The many-slice limit is a sum over paths", "The broken-line labels become increasingly fine."
    return "Many path terms, one amplitude at B", "At each detector point, add amplitudes first; square the magnitude afterward."


@lru_cache(maxsize=128)
def displayed_terms(b, n, release):
    m = MODELS[n]
    free = m.groups(b)[::-1] * ROTATE
    if release < 1:
        aperture = m.groups(b, aperture=True)[::-1] * ROTATE
        return mix(aperture, free, release)
    return free


def phasor_mapper():
    points = [0j]
    for n in MODELS:
        for b in np.linspace(-3.2, 3.2, 19):
            for ap in (False, True):
                points.extend(np.cumsum(MODELS[n].groups(float(b), aperture=ap)[::-1] * ROTATE))
    z = np.asarray(points)
    xmin, xmax, ymin, ymax = z.real.min(), z.real.max(), z.imag.min(), z.imag.max()
    scale = min((PBOX[2]-PBOX[0]-30)/(xmax-xmin), (PBOX[3]-PBOX[1]-30)/(ymax-ymin))
    origin = ((PBOX[0]+PBOX[2])/2 - scale*(xmin+xmax)/2,
              (PBOX[1]+PBOX[3])/2 + scale*(ymin+ymax)/2)
    return lambda v: (origin[0]+scale*v.real, origin[1]-scale*v.imag)


MAP = phasor_mapper()


def xy(u, y):
    return mix(X0, X1, u), YMID-YSCALE*y


def route_values(u, b, middle, variant, dy):
    amplitude = middle-b/2
    shape = b*u + amplitude*np.sin(np.pi*u)
    shape += variant*np.sin(2*np.pi*u)*np.sin(np.pi*u)
    if dy:
        shape = np.round(shape/dy)*dy
    shape[0], shape[-1] = 0, b
    return shape


@lru_cache(maxsize=12)
def plane_layer(n, slices, old, new_planes):
    """Stationary transverse grids are drawn once per refinement stage."""
    layer = Image.new("RGBA",(s(WIDTH),s(HEIGHT)),(0,0,0,0))
    draw = ImageDraw.Draw(layer)
    old_us = np.linspace(0,1,old+2)
    values = [float(v) for v in MODELS[n].y if abs(v)<=YMAX]
    radius = max(.48,1.65*(385/n)**.45)
    for u in np.linspace(0,1,slices+2)[1:-1]:
        persistent = bool(np.any(np.isclose(old_us,u)))
        if persistent == new_planes:
            continue
        x = float(xy(u,0)[0])
        dashed(draw,x,shade(BLUE,.28),1)
        for value in values:
            dot(draw,(x,YMID-YSCALE*value),radius,shade(BLUE,.47))
    return layer


def draw_routes(image, draw, t, b, n, slices, old, refine, release, active_y):
    text(draw, (54,143), "paths through the slices" if release else "routes through 49 openings", 18, bold=True)
    if t < 14:
        text(draw, (54,172), "one opening → one contribution", 13, MUTED)
    else:
        text(draw, (54,172), "central window · representative path labels", 13, MUTED)
    active_us = np.linspace(0,1,slices+2)
    old_us = np.linspace(0,1,old+2)
    route_ys = np.linspace(2.8, -2.8, 49)
    shown = min(49, max(0, int((t-1.0)/6*49))) if t < 7 else 49
    # Clip geometry to its plot window, keeping paths out of the labels.
    geometry = Image.new("RGBA",(s(WIDTH),s(HEIGHT)),(0,0,0,0))
    geom = ImageDraw.Draw(geometry)
    for idx, middle in enumerate(route_ys[:shown]):
        variant = .57*math.sin(idx*2.31)
        ynew = route_values(active_us, b, middle, variant, MODELS[n].dy)
        yold = route_values(old_us, b, middle, variant, MODELS[n].dy)
        ys = mix(np.interp(active_us,old_us,yold), ynew, refine)
        color = shade(BLUE, .42 if slices > 1 else .73)
        line(geom, [xy(u,y) for u,y in zip(active_us,ys)], color, 1)
    # Persisting planes stay in place; new planes appear halfway between them.
    if release >= 1:
        geometry.alpha_composite(plane_layer(n,slices,old,False))
        additions = plane_layer(n,slices,old,True)
        if refine < 1:
            additions = additions.copy()
            additions.putalpha(additions.getchannel("A").point([round(i*refine) for i in range(256)]))
        geometry.alpha_composite(additions)
        geom = ImageDraw.Draw(geometry)
    else:
        x = xy(.5,0)[0]
        line(geom,((x,202),(x,552)),shade(INK,1-release),4)
        if release > 0:
            dashed(geom,x,shade(BLUE,.28*release),1)
        for value in MODELS[n].y[np.abs(MODELS[n].y)<=YMAX]:
            if release == 0 and abs(value)>2.800001:
                continue
            dot(geom,(x,xy(.5,value)[1]),1.65,shade(BLUE,.47*release) if release else PANEL)
    # A gold route marks an example; its middle crossing selects a grouped arrow.
    if shown:
        example_new = route_values(active_us,b,active_y,.68,MODELS[n].dy)
        example_old = route_values(old_us,b,active_y,.68,MODELS[n].dy)
        ey = mix(np.interp(active_us,old_us,example_old),example_new,refine)
        if slices > 1:
            for variant in (-1.05, -.5, 1.2):
                group_new = route_values(active_us,b,active_y,variant,MODELS[n].dy)
                group_old = route_values(old_us,b,active_y,variant,MODELS[n].dy)
                gy = mix(np.interp(active_us,old_us,group_old),group_new,refine)
                line(geom,[xy(u,y) for u,y in zip(active_us,gy)],shade(GOLD,.38),1.4)
        line(geom, [xy(u,y) for u,y in zip(active_us,ey)], GOLD, 2.6)
        for u,y in zip(active_us[1:-1],ey[1:-1]):
            if slices < 16:
                dot(geom,xy(u,y),2.6,GOLD)
        dot(geom,xy(.5,active_y),4.3,GOLD)
    crop = (s(65),s(202),s(558),s(552))
    tile = geometry.crop(crop)
    image.paste(tile,(crop[0],crop[1]),tile)
    line(draw, (xy(1,YMAX),xy(1,-YMAX)), shade(INK,.62), 1.2)
    dot(draw,xy(0,0),7.5,INK)
    dot(draw,xy(1,b),6.5,INK)
    text(draw, (X0-14,YMID-3), "A", 15, bold=True, anchor="rm")
    text(draw, (X1+11,xy(1,b)[1]), "B", 15, bold=True, anchor="lm")
    if t < 14:
        footer = "49 openings"
    elif t < 20:
        footer = "1 imaginary slice"
    else:
        footer = f"{slices} slices · {n:,} sampled points per slice"
    text(draw, (310,578), footer, 14, MUTED, anchor="mm")


def draw_phasors(draw, t, b, n, release, active_index):
    text(draw, (621,143), "complex contributions at B",18,bold=True)
    label = ("one arrow per opening" if t < 14 else "one contribution per sampled point"
             if t < 20 else "paths grouped by their middle crossing")
    text(draw, (621,172), label,13,MUTED)
    terms = displayed_terms(b,n,release)
    if t < 7:
        inds = np.flatnonzero(np.abs(terms)>1e-15)
        progress = np.clip((t-1)/6*49,0,49)
        keep = int(progress)
        mask = np.zeros(len(terms))
        mask[inds[:keep]] = 1
        if keep < len(inds):
            mask[inds[keep]] = progress-keep
        terms = terms*mask
    chain = np.r_[0j,np.cumsum(terms)]
    origin = MAP(0j)
    line(draw, ((PBOX[0],origin[1]),(PBOX[2],origin[1])),shade(MUTED,.32),1)
    line(draw, ((origin[0],PBOX[1]),(origin[0],PBOX[3])),shade(MUTED,.32),1)
    text(draw,(PBOX[2],origin[1]+6),"Re",12,MUTED,anchor="ra")
    text(draw,(origin[0]+6,PBOX[1]),"Im",12,MUTED)
    coords = [MAP(v) for v in chain]
    line(draw,coords,BLUE,1.4)
    for i in range(len(terms)):
        a,z = coords[i],coords[i+1]
        if math.dist(a,z)>5:
            arrow(draw,a,z,BLUE,1.2,3.4)
    i = max(0,min(active_index,len(terms)-1))
    if t >= 1:
        arrow(draw,coords[i],coords[i+1],GOLD,3.5,6)
        dot(draw,coords[i+1],3,GOLD)
    current = chain[-1]
    if t >= 7:
        arrow(draw,origin,MAP(current),GREEN,6.2,12)
    else:
        line(draw,(origin,MAP(current)),shade(GREEN,.6),2)
    text(draw,(808,578),"green arrow = total amplitude",14,GREEN,True,anchor="mm")
    return current


def detector_y(b):
    return mix(DY1,DY0,(b+YMAX)/(2*YMAX))


def detector_x(intensity):
    return mix(DX0,DX1,min(1.0,max(0.0,intensity/IMAX)))


def draw_detector(draw,t,b,n,release,total):
    text(draw,(1051,143),"intensity at detector",17,bold=True)
    text(draw,(1051,172),"|total amplitude|²",13,MUTED)
    values = mix(PROFILE[n,True],PROFILE[n,False],release)
    intensity = np.abs(values)**2
    line(draw,((DX0,DY0),(DX0,DY1)),shade(MUTED,.5),1.4)
    reveal = np.clip((t-7)/2,0,1)
    pts = [(detector_x(i),detector_y(y)) for y,i in zip(DETECTOR_B_VALUES,intensity)]
    if reveal > 0:
        line(draw,pts[:max(2,round(len(pts)*reveal))],BLUE,2.4)
    marker = (detector_x(abs(total)**2),detector_y(b))
    line(draw,((DX0,marker[1]),marker),GOLD,2.4)
    dot(draw,marker,5.2,GREEN if t>=7 else GOLD)
    for y,label in ((DY0,"+"),((DY0+DY1)/2,"0"),(DY1,"−")):
        text(draw,(DX0-6,y),label,12,MUTED,anchor="rm")
    text(draw,((DX0+DX1)/2,578),"brighter to the right",12,MUTED,anchor="mm")


def draw_frame(t):
    t = min(t,60.0)  # Four-second final reading hold.
    b,n,slices,old,refine,release = state(t)
    image = Image.new("RGBA",(s(WIDTH),s(HEIGHT)),BG+(255,))
    draw = ImageDraw.Draw(image)
    title,subtitle = caption(t)
    text(draw,(42,28),title,26,bold=True)
    text(draw,(42,69),subtitle,17,MUTED)
    for box in (ROUTE,PHASOR,DETECTOR):
        draw.rounded_rectangle(tuple(s(v) for v in box),radius=s(13),fill=PANEL,outline=FAINT,width=s(2))
    m = MODELS[n]
    if t < 7:
        # Active crossing uses exactly the same top-to-bottom opening order.
        active = min(48,max(0,int((t-1)/6*49)))
        selected = m.y[m.aperture_indices[::-1][active]]
    elif t < 20:
        selected = m.y[np.argmin(abs(m.y-b/2))]
    else:
        selected = m.y[np.argmin(abs(m.y-(b/2+1.4*math.sin((t-20)*.37))))]
    idx = n-1-int(np.argmin(abs(m.y-selected)))
    draw_routes(image,draw,t,b,n,slices,old,refine,release,float(selected))
    total = draw_phasors(draw,t,b,n,release,idx)
    draw_detector(draw,t,b,n,release,total)
    if t < 14:
        eq = r"$\psi(B)=\sum_{j=1}^{49} K(B,C_j)\,\psi(C_j)$"
    elif t < 20:
        eq = r"$\psi(B)=\int dC_1\;K(B,C_1)\,\psi(C_1)$"
    elif t < 30:
        eq = r"$\psi(B)=\sum_{j_1}\cdots\sum_{j_N}K(B,C_{j_N})\cdots K(C_{j_2},C_{j_1})\,\psi(C_{j_1})$"
    elif t < 45:
        eq = r"$\psi(B)=\int dC_N\cdots\int dC_1\;K(B,C_N)\cdots K(C_2,C_1)\,\psi(C_1)$"
    else:
        eq = r"$\psi(B)=\int\mathcal{D}\gamma\;\mathcal{A}[\gamma]\qquad I(B)=|\psi(B)|^2$"
    place_equation(image,eq)
    if t < 20:
        bottom = "multiply along a route · add across alternatives"
    elif t < 45:
        bottom = f"M points on each of N slices → Mᴺ crossing sequences"
    else:
        bottom = "continuum limit of repeated propagation through intermediate points"
    text(draw,(640,692),bottom,13,MUTED,anchor="mm")
    return image.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)


def contact_sheet(frames,times,path,columns=4,thumb=(320,180)):
    rows = math.ceil(len(frames)/columns)
    sheet = Image.new("RGB",(columns*thumb[0],rows*(thumb[1]+25)),BG)
    draw = ImageDraw.Draw(sheet)
    label_font = font(7)
    for i,(im,t) in enumerate(zip(frames,times)):
        x,y = (i%columns)*thumb[0],(i//columns)*(thumb[1]+25)
        sheet.paste(im.resize(thumb,Image.Resampling.LANCZOS),(x,y))
        draw.text((x+6,y+thumb[1]+1),f"{t:.2f} s",fill=INK,font=label_font)
    sheet.save(path)


def preview():
    OUT.mkdir(parents=True,exist_ok=True)
    frames = [draw_frame(t) for t in SAMPLES]
    contact_sheet(frames,SAMPLES,OUT/f"{NAME}-contact-sheet.png")
    for t in (8.5,22.0,41.0,62.5):
        draw_frame(t).save(OUT/f"{NAME}-check-{t:g}.png")
    draw_frame(DURATION-1/FPS).save(OUT/f"{NAME}-final.png")


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/f"{NAME}.mp4"
    writer = imageio_ffmpeg.write_frames(str(path),(WIDTH,HEIGHT),fps=FPS,codec="libx264",
            pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=8,macro_block_size=16,
            ffmpeg_log_level="warning",output_params=["-preset","medium","-movflags","+faststart"])
    writer.send(None)
    try:
        for f in range(round(DURATION*FPS)):
            writer.send(np.asarray(draw_frame(f/FPS)))
            if f % (FPS*4) == 0:
                print(f"rendered {f/FPS:.0f} / {DURATION:.0f} seconds",flush=True)
    finally:
        writer.close()
    print(path,flush=True)


def qa():
    """Inspect media properties and create dense sheets from encoded frames."""
    path = OUT/f"{NAME}.mp4"
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([executable,"-v","error","-i",str(path),"-f","null","-"],check=True)
    qa_dir = ROOT/".tools"/"progressive-animation-qa"
    qa_dir.mkdir(parents=True,exist_ok=True)
    reader = imageio_ffmpeg.read_frames(str(path),pix_fmt="rgb24",output_params=["-vf","fps=2,scale=640:360"])
    metadata = next(reader)
    frames, times, page = [],[],1
    for i,data in enumerate(reader):
        frames.append(Image.frombytes("RGB",(640,360),data))
        times.append(i/2)
        if len(frames) == 16:
            contact_sheet(frames,times,qa_dir/f"motion-{page:02d}.png")
            frames,times,page = [],[],page+1
    if frames:
        contact_sheet(frames,times,qa_dir/f"motion-{page:02d}.png")
    # Dense frame sampling around every scene/mesh boundary.
    transitions = [7,9,14,18,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,50,52,58,60]
    times = [v+d for v in transitions for d in (-.25,0,.25,.5)]
    for page,start in enumerate(range(0,len(times),16),1):
        batch = times[start:start+16]
        frames = []
        for t in batch:
            raw = subprocess.run([executable,"-v","error","-ss",str(t),"-i",str(path),"-frames:v","1",
                    "-vf","scale=640:360","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
            frames.append(Image.frombytes("RGB",(640,360),raw))
        contact_sheet(frames,batch,qa_dir/f"transitions-{page:02d}.png")
    print(json.dumps({k:str(v) for k,v in metadata.items()},indent=2))
    print(qa_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--qa",action="store_true")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(run_checks(),indent=2))
    if args.preview:
        preview()
    if args.render:
        render()
    if args.qa:
        qa()
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
