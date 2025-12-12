import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# =====================
# DESKTOP CONFIG
# =====================
RADIUS = 3
HEX_SIZE = 48
PADDING = 18

BG = (18, 18, 22)
GRID_COLOR = (70, 70, 82)
OUTLINE = (35, 35, 45)
TEXT_COLOR = (15, 15, 18)

TILE_COLORS = [
    (235, 235, 235),  # 1
    (200, 220, 255),  # 3
    (180, 205, 255),  # 9
    (255, 225, 190),  # 27
    (255, 205, 170),  # 81
    (255, 185, 150),  # 243
    (255, 165, 130),  # 729
    (255, 145, 110),  # 2187
    (255, 125, 95),   # 6561
    (255, 105, 80),   # 19683
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

def axial_to_pixel(q: int, r: int, size: int = HEX_SIZE) -> Tuple[float, float]:
    x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2) * r)
    y = size * (3.0 / 2) * r
    return x, y

def hex_corners(cx: float, cy: float, size: int = HEX_SIZE) -> List[Tuple[float, float]]:
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)  # pointy-top
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
    out = []
    score = 0
    i = 0
    n = len(values)
    while i < n:
        v = values[i]
        j = i + 1
        while j < n and values[j] == v:
            j += 1
        run = j - i
        triples = run // 3
        rem = run % 3
        for _ in range(triples):
            nv = v * 3
            out.append(nv)
            score += nv
        for _ in range(rem):
            out.append(v)
        i = j
    return out, score

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
    s = GameState(board=board, score=0, game_over=False)
    spawn_tile(s); spawn_tile(s)
    return s

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
            for i in range(len(vals) - 2):
                if vals[i] != 0 and vals[i] == vals[i + 1] == vals[i + 2]:
                    return True
    return False

def do_move(state: GameState, dir_index: int, dir_lines) -> bool:
    if state.game_over:
        return False

    changed = False
    gained = 0

    for line in dir_lines[dir_index]:
        original = [state.board[c] for c in line]
        non_zero = [v for v in original if v != 0]
        if not non_zero:
            continue

        merged, add = merge_triples(non_zero)
        new_vals = merged + [0] * (len(line) - len(merged))

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

def render_board(cells, centers, state: GameState) -> Image.Image:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for c in cells:
        cx, cy = centers[c]
        for x, y in hex_corners(cx, cy):
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)

    w = int((maxx - minx) + 2 * PADDING)
    h = int((maxy - miny) + 2 * PADDING)
    sx = -minx + PADDING
    sy = -miny + PADDING

    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for c in cells:
        cx, cy = centers[c]
        pts = [(x + sx, y + sy) for x, y in hex_corners(cx, cy)]
        draw.polygon(pts, fill=GRID_COLOR)

        v = state.board[c]
        if v:
            lv = min(level_from_value(v), len(TILE_COLORS) - 1)
            draw.polygon(pts, fill=TILE_COLORS[lv])
            draw.polygon(pts, outline=OUTLINE, width=3)
            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font_big)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx + sx - tw / 2, cy + sy - th / 2), text, fill=TEXT_COLOR, font=font_big)
        else:
            draw.polygon(pts, outline=(55, 55, 66), width=2)

    draw.text((10, 10), f"Score: {state.score}", fill=(235, 235, 235), font=font_small)
    if state.game_over:
        draw.text((10, 30), "GAME OVER (press R)", fill=(255, 180, 180), font=font_small)

    return img

# =====================
# KEY MAPPING (Q E A D Z X)
# =====================
KEY_TO_MOVE = {
    "a": 0,  # left
    "d": 3,  # right
    "q": 5,  # up-left
    "e": 4,  # up-right
    "z": 1,  # down-left
    "x": 2,  # down-right
}

# =====================
# STREAMLIT APP
# =====================
st.set_page_config(page_title="Hex 2048 Base-3 (Desktop)", layout="centered")

# Keep keyboard focus on the hidden input
components.html(
    """
    <script>
      function focusKeyInput() {
        const el = window.parent.document.querySelector('input[data-testid="stTextInput"]');
        if (el) el.focus();
      }
      window.addEventListener('load', () => setTimeout(focusKeyInput, 200));
      window.addEventListener('click', () => setTimeout(focusKeyInput, 50));
    </script>
    """,
    height=0,
)

cells = generate_hex_cells(RADIUS)
dir_lines = build_direction_lines(cells, DIRECTIONS)

centers_local = {c: axial_to_pixel(c[0], c[1], size=HEX_SIZE) for c in cells}
cx0 = sum(x for x, _ in centers_local.values()) / len(centers_local)
cy0 = sum(y for _, y in centers_local.values()) / len(centers_local)
centers = {c: (centers_local[c][0] - cx0, centers_local[c][1] - cy0) for c in cells}

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)

st.title("Hex 2048 (Base-3) — Desktop")
st.caption("Controls: Q=↖  E=↗  A=←  D=→  Z=↙  X=↘   |   R=Reset")

# The "hidden-ish" key input (must exist to capture keys)
key = st.text_input("Key capture (click once if needed)", value="", max_chars=1, key="keycap")
k = (key or "").strip().lower()

state: GameState = st.session_state.state

if k == "r":
    st.session_state.state = new_game(cells)
    st.session_state.keycap = ""
    st.rerun()

if k in KEY_TO_MOVE and not state.game_over:
    moved = do_move(state, KEY_TO_MOVE[k], dir_lines)
    st.session_state.keycap = ""  # clear so same key can be pressed repeatedly
    st.rerun()

# render board
img = render_board(cells, centers, state)
st.image(img, use_container_width=True)

