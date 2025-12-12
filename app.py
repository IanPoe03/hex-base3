import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =====================
# CONFIG
# =====================

RADIUS = 3  # <- requested
HEX_SIZE = 42  # pixel radius of each small hex
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

# Axial (pointy-top) directions
DIRECTIONS = [
    (1, 0),    # 0
    (1, -1),   # 1
    (0, -1),   # 2
    (-1, 0),   # 3
    (-1, 1),   # 4
    (0, 1),    # 5
]

DIR_LABELS = ["→", "↗", "↖", "←", "↙", "↘"]  # for UI buttons


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
        angle_deg = 60 * i - 30  # pointy-top
        a = math.radians(angle_deg)
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
        # start cells: those with no neighbor in the -direction
        for q, r in cells:
            back = (q - dq, r - dr)
            if back not in cell_set:
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
    """
    Compress list (non-zero), then merge triples of identical values from the front.
    """
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
    # value is power of 3 (including 1=3^0)
    # level = log_3(v)
    lv = int(round(math.log(v, 3)))
    return max(0, lv)


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
    # 1 with 0.75, 3 with 0.25
    state.board[c] = 1 if random.random() < 0.75 else 3
    return True

def has_moves(state: GameState, dir_lines) -> bool:
    if any(v == 0 for v in state.board.values()):
        return True
    # any triple in any direction line?
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
# RENDERING (PIL)
# =====================

def render_board_image(
    cells: List[Tuple[int, int]],
    centers: Dict[Tuple[int, int], Tuple[float, float]],
    state: GameState,
    arrows: List[dict],
) -> Image.Image:
    # compute bounds from hex corners
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for c in cells:
        cx, cy = centers[c]
        for x, y in hex_corners(cx, cy):
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    # include arrow bounds
    for a in arrows:
        for x, y in a["points"]:
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    w = int((maxx - minx) + 2 * PADDING)
    h = int((maxy - miny) + 2 * PADDING)

    # shift all drawing coordinates so they fit in image
    sx = -minx + PADDING
    sy = -miny + PADDING

    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    # font
    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # draw hex tiles
    for c in cells:
        cx, cy = centers[c]
        pts = [(x + sx, y + sy) for x, y in hex_corners(cx, cy)]
        draw.polygon(pts, fill=GRID_COLOR)
        v = state.board[c]
        if v != 0:
            lv = level_from_value(v)
            lv = min(lv, len(TILE_COLORS) - 1)
            draw.polygon(pts, fill=TILE_COLORS[lv])
            draw.polygon(pts, outline=OUTLINE, width=2)

            text = str(v)
            tw, th = draw.textbbox((0, 0), text, font=font_big)[2:]
            draw.text((cx + sx - tw / 2, cy + sy - th / 2), text, fill=TEXT_COLOR, font=font_big)
        else:
            draw.polygon(pts, outline=(55, 55, 65), width=1)

    # draw movement arrows (corners, outside board)
    for a in arrows:
        pts = [(x + sx, y + sy) for x, y in a["points"]]
        draw.polygon(pts, fill=(200, 200, 220))
        draw.polygon(pts, outline=(50, 50, 60), width=2)

    # game over overlay text (simple)
    if state.game_over:
        msg = "GAME OVER"
        sub = "Press Reset"
        mw, mh = draw.textbbox((0, 0), msg, font=font_big)[2:]
        sw, sh = draw.textbbox((0, 0), sub, font=font_small)[2:]
        draw.rectangle([0, 0, w, h], fill=None, outline=None)
        draw.text((w / 2 - mw / 2, 18), msg, fill=(240, 240, 240), font=font_big)
        draw.text((w / 2 - sw / 2, 18 + mh + 6), sub, fill=(220, 220, 220), font=font_small)

    return img


def build_arrow_polys_corner_outside(
    cells: List[Tuple[int, int]],
    centers: Dict[Tuple[int, int], Tuple[float, float]],
    center_x: float,
    center_y: float,
) -> List[dict]:
    arrows = []
    offset = HEX_SIZE * 1.4
    arrow_len = 26
    arrow_width = 22

    for idx, (dq, dr) in enumerate(DIRECTIONS):
        # movement is along -direction
        mvx, mvy = axial_to_pixel(-dq, -dr, size=HEX_SIZE)
        L = math.hypot(mvx, mvy)
        if L == 0:
            continue
        ux, uy = mvx / L, mvy / L

        # farthest tile center along this movement direction => corner
        t_max = -float("inf")
        for c in cells:
            cx, cy = centers[c]
            vx, vy = cx - center_x, cy - center_y
            t = vx * ux + vy * uy
            if t > t_max:
                t_max = t

        # arrow center outside board corner
        cx = center_x + (t_max + offset) * ux
        cy = center_y + (t_max + offset) * uy

        # arrow tip outward
        tip = (cx + ux * arrow_len, cy + uy * arrow_len)
        back_center = (cx - ux * arrow_len * 0.6, cy - uy * arrow_len * 0.6)
        px, py = -uy, ux
        left = (back_center[0] + px * arrow_width / 2, back_center[1] + py * arrow_width / 2)
        right = (back_center[0] - px * arrow_width / 2, back_center[1] - py * arrow_width / 2)

        arrows.append({"dir": idx, "points": [tip, left, right]})

    return arrows


# =====================
# STREAMLIT APP
# =====================

st.set_page_config(page_title="Hex 2048 Base-3", layout="centered")

cells = generate_hex_cells(RADIUS)
cell_set = set(cells)
dir_lines = build_direction_lines(cells, DIRECTIONS)

# Precompute centers in local coords
centers_local = {c: axial_to_pixel(c[0], c[1], size=HEX_SIZE) for c in cells}

# Center board at (0,0) for drawing; shift later
# But we need a "board center" in the same coord system:
cx0 = sum(x for x, y in centers_local.values()) / len(centers_local)
cy0 = sum(y for x, y in centers_local.values()) / len(centers_local)

# Shift centers so board is centered at (0,0)
centers = {c: (centers_local[c][0] - cx0, centers_local[c][1] - cy0) for c in cells}

# Board center is now (0,0)
BOARD_CX, BOARD_CY = 0.0, 0.0

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)

state: GameState = st.session_state.state

st.title("Hex 2048 (Base-3, Triple-Merge)")
st.caption("Spawn: 1 (75%), 3 (25%). Merge: 3-in-a-row along move direction → value × 3.")

# Controls
colA, colB, colC = st.columns([1, 1, 1])

with colA:
    if st.button("↖ (up-left)", use_container_width=True, disabled=state.game_over):
        do_move(state, 2, dir_lines)
with colB:
    if st.button("↗ (up-right)", use_container_width=True, disabled=state.game_over):
        do_move(state, 1, dir_lines)
with colC:
    if st.button("Reset", use_container_width=True):
        st.session_state.state = new_game(cells)
        state = st.session_state.state

colD, colE = st.columns([1, 1])
with colD:
    if st.button("← (left)", use_container_width=True, disabled=state.game_over):
        do_move(state, 0, dir_lines)  # <-- FIXED FOR "feel": left is dir 0 (since movement is -dir)
with colE:
    if st.button("→ (right)", use_container_width=True, disabled=state.game_over):
        do_move(state, 3, dir_lines)  # <-- FIXED FOR "feel": right is dir 3

colF, colG, colH = st.columns([1, 1, 1])
with colF:
    if st.button("↙ (down-left)", use_container_width=True, disabled=state.game_over):
        do_move(state, 4, dir_lines)
with colG:
    if st.button("↘ (down-right)", use_container_width=True, disabled=state.game_over):
        do_move(state, 5, dir_lines)
with colH:
    st.metric("Score", state.score)

# Build arrows (for visualization only, like your pygame version)
arrows = build_arrow_polys_corner_outside(cells, centers, BOARD_CX, BOARD_CY)

img = render_board_image(cells, centers, state, arrows)
st.image(img, use_container_width=True)

if state.game_over:
    st.error("No moves left. Hit Reset to play again.")

st.markdown(
    """
**Tip:** On Streamlit Cloud this is “button-driven” (not real-time keyboard), but it’s perfect for public playtesting.
"""
)

