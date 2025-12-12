import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

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
    (230, 230, 230),  # 1
    (200, 220, 255),  # 3
    (180, 200, 255),  # 9
    (255, 220, 180),  # 27
    (255, 200, 160),  # 81
    (255, 180, 140),  # 243
    (255, 160, 120),  # 729
    (255, 140, 100),  # 2187
    (255, 120, 80),
    (255, 100, 60),
]

# Axial directions (pointy-top)
DIRECTIONS = [
    (1, 0),    # 0
    (1, -1),   # 1
    (0, -1),   # 2
    (-1, 0),   # 3
    (-1, 1),   # 4
    (0, 1),    # 5
]

# FEEL-CORRECT mapping (because the slide happens along -DIRECTIONS[dir_index])
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

def axial_to_pixel(q: int, r: int, size: int = HEX_SIZE) -> Tuple[float, float]:
    x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2) * r)
    y = size * (3.0 / 2) * r
    return x, y

def hex_corners(cx: float, cy: float, size: int = HEX_SIZE) -> List[Tuple[float, float]]:
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return pts

def generate_hex_cells(radius: int) -> List[Tuple[int, int]]:
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

def build_direction_lines(cells: List[Tuple[int, int]], directions) -> List[List[List[Tuple[int, int]]]]:
    cell_set = set(cells)
    all_dir_lines = []
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
        all_dir_lines.append(lines)
    return all_dir_lines

def merge_triples(values: List[int]) -> Tuple[List[int], int]:
    if not values:
        return [], 0
    result = []
    score_add = 0
    i = 0
    n = len(values)
    while i < n:
        v = values[i]
        j = i + 1
        while j < n and values[j] == v:
            j += 1
        run_len = j - i
        triples = run_len // 3
        rem = run_len % 3

        for _ in range(triples):
            new_val = v * 3
            result.append(new_val)
            score_add += new_val
        for _ in range(rem):
            result.append(v)

        i = j
    return result, score_add

def level_from_value(v: int) -> int:
    if v <= 0:
        return 0
    return max(0, int(round(math.log(v, 3))))

@dataclass
class GameState:
    board: Dict[Tuple[int, int], int]
    score: int
    game_over: bool

def new_game(cells: List[Tuple[int, int]]) -> GameState:
    board = {c: 0 for c in cells}
    state = GameState(board=board, score=0, game_over=False)
    spawn_tile(state)
    spawn_tile(state)
    return state

def spawn_tile(state: GameState) -> bool:
    empties = [c for c, v in state.board.items() if v == 0]
    if not empties:
        return False
    c = random.choice(empties)
    state.board[c] = 1 if random.random() < 0.75 else 3
    return True

def has_moves(state: GameState, dir_lines) -> bool:
    if any(v == 0 for v in state.board.values()):
        return True
    for d_idx in range(6):
        for line in dir_lines[d_idx]:
            vals = [state.board[c] for c in line]
            prev = None
            cnt = 0
            for v in vals:
                if v != 0 and v == prev:
                    cnt += 1
                else:
                    prev = v
                    cnt = 1 if v != 0 else 0
                if v != 0 and cnt >= 3:
                    return True
    return False

def do_move(state: GameState, dir_index: int, dir_lines) -> bool:
    if state.game_over:
        return False

    changed = False
    gained = 0

    for line in dir_lines[dir_index]:
        vals = [state.board[c] for c in line]
        original = list(vals)

        non_zero = [v for v in vals if v != 0]
        if not non_zero:
            continue

        merged, add = merge_triples(non_zero)
        new_vals = merged + [0] * (len(vals) - len(merged))

        if new_vals != original:
            changed = True

        gained += add

        for c, v in zip(line, new_vals):
            state.board[c] = v

    if changed:
        state.score += gained
        spawn_tile(state)
        if not has_moves(state, dir_lines):
            state.game_over = True

    return changed

# =====================
# CLICKABLE ARROWS
# =====================

def point_in_triangle(px: float, py: float, a, b, c) -> bool:
    ax, ay = a
    bx, by = b
    cx, cy = c
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def build_corner_arrows(cells, centers, board_cx=0.0, board_cy=0.0):
    arrows = []
    offset = HEX_SIZE * 1.4
    arrow_len = 26
    arrow_width = 22

    for idx, (dq, dr) in enumerate(DIRECTIONS):
        # movement is along -direction
        mvx, mvy = axial_to_pixel(-dq, -dr, size=HEX_SIZE)
        L = math.hypot(mvx, mvy)
        ux, uy = mvx / L, mvy / L

        # farthest tile center along movement direction
        t_max = -float("inf")
        for c in cells:
            cx, cy = centers[c]
            t = (cx - board_cx) * ux + (cy - board_cy) * uy
            t_max = max(t_max, t)

        cx = board_cx + (t_max + offset) * ux
        cy = board_cy + (t_max + offset) * uy

        tip = (cx + ux * arrow_len, cy + uy * arrow_len)
        back = (cx - ux * arrow_len * 0.6, cy - uy * arrow_len * 0.6)
        px, py = -uy, ux
        left = (back[0] + px * arrow_width / 2, back[1] + py * arrow_width / 2)
        right = (back[0] - px * arrow_width / 2, back[1] - py * arrow_width / 2)

        arrows.append({"dir": idx, "points": [tip, left, right]})

    return arrows

# =====================
# RENDERING (PIL)
# =====================

def render_board_image(cells, centers, state, arrows):
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for c in cells:
        cx, cy = centers[c]
        for x, y in hex_corners(cx, cy):
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)
    for a in arrows:
        for x, y in a["points"]:
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)

    w = int((maxx - minx) + 2 * PADDING)
    h = int((maxy - miny) + 2 * PADDING)
    sx = -minx + PADDING
    sy = -miny + PADDING

    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 26)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # tiles
    for c in cells:
        cx, cy = centers[c]
        pts = [(x + sx, y + sy) for x, y in hex_corners(cx, cy)]
        draw.polygon(pts, fill=GRID_COLOR)

        v = state.board[c]
        if v:
            lv = min(level_from_value(v), len(TILE_COLORS) - 1)
            draw.polygon(pts, fill=TILE_COLORS[lv])
            draw.polygon(pts, outline=OUTLINE, width=2)

            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font_big)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx + sx - tw / 2, cy + sy - th / 2), text, fill=TEXT_COLOR, font=font_big)
        else:
            draw.polygon(pts, outline=(55, 55, 65), width=1)

    # arrows
    arrows_img = []
    for a in arrows:
        pts = [(x + sx, y + sy) for x, y in a["points"]]
        draw.polygon(pts, fill=(200, 200, 220))
        draw.polygon(pts, outline=(50, 50, 60), width=2)
        arrows_img.append({"dir": a["dir"], "points": pts})

    # HUD text
    hud = "Keys: Q↖ E↗  A← D→  Z↙ X↘   |   Click arrows"
    draw.text((12, 10), hud, fill=(235, 235, 235), font=font_small)

    if state.game_over:
        msg = "GAME OVER — Press Reset"
        draw.text((12, 28), msg, fill=(255, 200, 200), font=font_small)

    return img, arrows_img

# =====================
# STREAMLIT APP
# =====================

st.set_page_config(page_title="Hex 2048 Base-3", layout="centered")

cells = generate_hex_cells(RADIUS)
dir_lines = build_direction_lines(cells, DIRECTIONS)

# centers (recenter to 0,0)
centers_local = {c: axial_to_pixel(c[0], c[1], size=HEX_SIZE) for c in cells}
cx0 = sum(x for x, _ in centers_local.values()) / len(centers_local)
cy0 = sum(y for _, y in centers_local.values()) / len(centers_local)
centers = {c: (centers_local[c][0] - cx0, centers_local[c][1] - cy0) for c in cells}

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)
if "last_click_t" not in st.session_state:
    st.session_state.last_click_t = None

state: GameState = st.session_state.state

st.title("Hex 2048 (Base-3)")

# --- Keyboard shortcuts using native Streamlit button shortcuts (works on Cloud) ---
with st.sidebar:
    st.subheader("Controls")
    st.write("Keyboard: Q↖ E↗  A← D→  Z↙ C↘")
    st.write("Mouse: click the arrow triangles")

    reset = st.button("Reset", key="reset_btn")

    # Small “hotkey buttons” (they can be visible; Streamlit will show hints)
    # Using shortcut=... is now built-in. :contentReference[oaicite:2]{index=2}
    mv_ul = st.button("↖ Up-Left", shortcut="Q", disabled=state.game_over, key="mv_ul")
    mv_ur = st.button("↗ Up-Right", shortcut="E", disabled=state.game_over, key="mv_ur")
    mv_l  = st.button("← Left",    shortcut="A", disabled=state.game_over, key="mv_l")
    mv_r  = st.button("→ Right",   shortcut="D", disabled=state.game_over, key="mv_r")
    mv_dl = st.button("↙ Down-Left", shortcut="Z", disabled=state.game_over, key="mv_dl")
    mv_dr = st.button("↘ Down-Right", shortcut="X", disabled=state.game_over, key="mv_dr")

if reset:
    st.session_state.state = new_game(cells)
    st.rerun()

if mv_l:
    do_move(state, MOVE["left"], dir_lines); st.rerun()
if mv_r:
    do_move(state, MOVE["right"], dir_lines); st.rerun()
if mv_ul:
    do_move(state, MOVE["up_left"], dir_lines); st.rerun()
if mv_ur:
    do_move(state, MOVE["up_right"], dir_lines); st.rerun()
if mv_dl:
    do_move(state, MOVE["down_left"], dir_lines); st.rerun()
if mv_dr:
    do_move(state, MOVE["down_right"], dir_lines); st.rerun()

# --- Render + clickable arrow triangles ---
arrows_world = build_corner_arrows(cells, centers, 0.0, 0.0)
img, arrows_img = render_board_image(cells, centers, state, arrows_world)

click = streamlit_image_coordinates(img, key="board_click")

st.image(img, use_container_width=True)
st.metric("Score", state.score)

if click and "x" in click and "y" in click and "time" in click:
    if st.session_state.last_click_t != click["time"]:
        st.session_state.last_click_t = click["time"]

        px = float(click["x"])
        py = float(click["y"])

        clicked_dir: Optional[int] = None
        for a in arrows_img:
            p0, p1, p2 = a["points"]
            if point_in_triangle(px, py, p0, p1, p2):
                clicked_dir = a["dir"]
                break

        if clicked_dir is not None and not state.game_over:
            do_move(state, clicked_dir, dir_lines)
            st.rerun()

if state.game_over:
    st.error("No moves left. Press Reset (R).")

