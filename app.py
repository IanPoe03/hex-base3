import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_shortcuts import add_shortcuts

# =====================
# CONFIG
# =====================

RADIUS = 3
HEX_SIZE = 42
PADDING = 40

BG = (25, 25, 30)
GRID_COLOR = (80, 80, 90)
OUTLINE = (40, 40, 50)
TEXT_COLOR = (20, 20, 25)

TILE_COLORS = [
    (230, 230, 230),
    (200, 220, 255),
    (180, 200, 255),
    (255, 220, 180),
    (255, 200, 160),
    (255, 180, 140),
    (255, 160, 120),
    (255, 140, 100),
]

# Axial directions
DIRECTIONS = [
    (1, 0),    # 0
    (1, -1),   # 1
    (0, -1),   # 2
    (-1, 0),   # 3
    (-1, 1),   # 4
    (0, 1),    # 5
]

# Keyboard mapping (FEEL-CORRECT)
MOVE = {
    "left": 0,        # A
    "right": 3,       # D
    "up_left": 5,     # Q
    "up_right": 4,    # E
    "down_left": 1,   # Z
    "down_right": 2,  # C
}

# =====================
# HEX GEOMETRY
# =====================

def axial_to_pixel(q, r, size=HEX_SIZE):
    x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2) * r)
    y = size * (3.0 / 2) * r
    return x, y

def hex_corners(cx, cy, size=HEX_SIZE):
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return pts

def generate_hex_cells(radius):
    cells = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            cells.append((q, r))
    return cells

# =====================
# GAME LOGIC
# =====================

def build_direction_lines(cells, directions):
    cell_set = set(cells)
    all_lines = []
    for dq, dr in directions:
        lines = []
        for q, r in cells:
            if (q - dq, r - dr) not in cell_set:
                line = []
                cq, cr = q, r
                while (cq, cr) in cell_set:
                    line.append((cq, cr))
                    cq += dq
                    cr += dr
                lines.append(line)
        all_lines.append(lines)
    return all_lines

def merge_triples(values):
    result, score = [], 0
    i = 0
    while i < len(values):
        v = values[i]
        j = i
        while j < len(values) and values[j] == v:
            j += 1
        count = j - i
        for _ in range(count // 3):
            result.append(v * 3)
            score += v * 3
        for _ in range(count % 3):
            result.append(v)
        i = j
    return result, score

def level(v):
    return 0 if v == 0 else int(round(math.log(v, 3)))

@dataclass
class GameState:
    board: Dict
    score: int
    game_over: bool

def new_game(cells):
    board = {c: 0 for c in cells}
    state = GameState(board, 0, False)
    spawn(state); spawn(state)
    return state

def spawn(state):
    empties = [c for c in state.board if state.board[c] == 0]
    if empties:
        state.board[random.choice(empties)] = 1 if random.random() < 0.75 else 3

def has_moves(state, lines):
    if any(v == 0 for v in state.board.values()):
        return True
    for d in range(6):
        for line in lines[d]:
            vals = [state.board[c] for c in line]
            for i in range(len(vals) - 2):
                if vals[i] == vals[i+1] == vals[i+2] != 0:
                    return True
    return False

def move(state, d, lines):
    if state.game_over:
        return
    changed, gained = False, 0
    for line in lines[d]:
        vals = [state.board[c] for c in line if state.board[c] != 0]
        merged, score = merge_triples(vals)
        new_vals = merged + [0] * (len(line) - len(merged))
        if [state.board[c] for c in line] != new_vals:
            changed = True
        gained += score
        for c, v in zip(line, new_vals):
            state.board[c] = v
    if changed:
        state.score += gained
        spawn(state)
        if not has_moves(state, lines):
            state.game_over = True

# =====================
# ARROWS + HIT TEST
# =====================

def point_in_triangle(p, a, b, c):
    px, py = p
    ax, ay = a; bx, by = b; cx, cy = c
    v0 = (cx-ax, cy-ay)
    v1 = (bx-ax, by-ay)
    v2 = (px-ax, py-ay)
    dot = lambda u, v: u[0]*v[0]+u[1]*v[1]
    d00 = dot(v0, v0); d01 = dot(v0, v1); d02 = dot(v0, v2)
    d11 = dot(v1, v1); d12 = dot(v1, v2)
    inv = 1/(d00*d11-d01*d01)
    u = (d11*d02-d01*d12)*inv
    v = (d00*d12-d01*d02)*inv
    return u >= 0 and v >= 0 and u+v <= 1

def build_arrows(cells, centers):
    arrows = []
    for i, (dq, dr) in enumerate(DIRECTIONS):
        mvx, mvy = axial_to_pixel(-dq, -dr)
        L = math.hypot(mvx, mvy)
        ux, uy = mvx/L, mvy/L
        t = max(cx*ux+cy*uy for cx, cy in centers.values())
        cx, cy = (t + HEX_SIZE*1.4)*ux, (t + HEX_SIZE*1.4)*uy
        tip = (cx+ux*26, cy+uy*26)
        base = (cx-ux*16, cy-uy*16)
        px, py = -uy, ux
        arrows.append({"dir": i, "pts": [
            tip,
            (base[0]+px*11, base[1]+py*11),
            (base[0]-px*11, base[1]-py*11)
        ]})
    return arrows

# =====================
# RENDER
# =====================

def render(cells, centers, state, arrows):
    xs, ys = [], []
    for c in cells:
        for x,y in hex_corners(*centers[c]):
            xs.append(x); ys.append(y)
    for a in arrows:
        for x,y in a["pts"]:
            xs.append(x); ys.append(y)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w, h = int(maxx-minx+2*PADDING), int(maxy-miny+2*PADDING)
    sx, sy = -minx+PADDING, -miny+PADDING
    img = Image.new("RGB",(w,h),BG)
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default()
    for c in cells:
        pts = [(x+sx,y+sy) for x,y in hex_corners(*centers[c])]
        v = state.board[c]
        if v:
            d.polygon(pts, fill=TILE_COLORS[min(level(v),len(TILE_COLORS)-1)])
            d.text((centers[c][0]+sx-6, centers[c][1]+sy-6), str(v), fill=TEXT_COLOR, font=f)
        d.polygon(pts, outline=GRID_COLOR)
    arrows_img=[]
    for a in arrows:
        pts=[(x+sx,y+sy) for x,y in a["pts"]]
        d.polygon(pts, fill=(200,200,220))
        arrows_img.append({"dir":a["dir"],"pts":pts})
    return img, arrows_img

# =====================
# STREAMLIT APP
# =====================

st.set_page_config("Hex 2048 Base-3")

cells = generate_hex_cells(RADIUS)
lines = build_direction_lines(cells, DIRECTIONS)
centers0 = {c: axial_to_pixel(*c) for c in cells}
cx = sum(x for x,_ in centers0.values())/len(cells)
cy = sum(y for _,y in centers0.values())/len(cells)
centers = {c:(centers0[c][0]-cx, centers0[c][1]-cy) for c in cells}

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)

state = st.session_state.state

st.title("Hex 2048 (Base-3)")
st.caption("Keys: Q↖ E↗ A← D→ Z↙ C↘ | Or click arrows")

# keyboard shortcuts
with st.sidebar:
    ml = st.button("ml"); mr = st.button("mr")
    mul = st.button("mul"); mur = st.button("mur")
    mdl = st.button("mdl"); mdr = st.button("mdr")
    reset = st.button("Reset")

add_shortcuts(
    ml="a", mr="d",
    mul="q", mur="e",
    mdl="z", mdr="c"
)

if reset:
    st.session_state.state = new_game(cells)
    st.rerun()

if ml: move(state, MOVE["left"], lines); st.rerun()
if mr: move(state, MOVE["right"], lines); st.rerun()
if mul: move(state, MOVE["up_left"], lines); st.rerun()
if mur: move(state, MOVE["up_right"], lines); st.rerun()
if mdl: move(state, MOVE["down_left"], lines); st.rerun()
if mdr: move(state, MOVE["down_right"], lines); st.rerun()

arrows = build_arrows(cells, centers)
img, arrows_img = render(cells, centers, state, arrows)
click = streamlit_image_coordinates(img, key="board")

st.image(img, use_container_width=True)
st.metric("Score", state.score)

if click:
    px, py = click["x"], click["y"]
    for a in arrows_img:
        if point_in_triangle((px,py), *a["pts"]):
            move(state, a["dir"], lines)
            st.rerun()

if state.game_over:
    st.error("Game Over — press Reset")

