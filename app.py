import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# =====================
# MOBILE CONFIG
# =====================
RADIUS = 2
HEX_SIZE = 58
PADDING = 22

BG = (18, 18, 22)
GRID_COLOR = (70, 70, 82)
OUTLINE = (35, 35, 45)
TEXT_COLOR = (15, 15, 18)

ARROW_FILL = (220, 220, 235)
ARROW_OUTLINE = (55, 55, 70)

TILE_COLORS = [
    (235, 235, 235),  # 1
    (200, 220, 255),  # 3
    (180, 205, 255),  # 9
    (255, 225, 190),  # 27
    (255, 205, 170),  # 81
    (255, 185, 150),  # 243
    (255, 165, 130),  # 729
    (255, 145, 110),  # 2187
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

# Sliding happens along -DIRECTIONS[dir_index]
MOVE_LABELS = {
    0: "←",
    3: "→",
    5: "↖",
    4: "↗",
    1: "↙",
    2: "↘",
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

# =====================
# CLICKABLE TRIANGLES
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

def build_arrows_world(cells, centers, board_cx=0.0, board_cy=0.0):
    arrows = []
    offset = HEX_SIZE * 1.55  # outside board
    arrow_len = 30
    arrow_width = 34

    for idx, (dq, dr) in enumerate(DIRECTIONS):
        # movement vector is along -direction
        mvx, mvy = axial_to_pixel(-dq, -dr, size=HEX_SIZE)
        L = math.hypot(mvx, mvy)
        ux, uy = mvx / L, mvy / L

        # find farthest tile in that direction
        t_max = -1e18
        for c in cells:
            cx, cy = centers[c]
            t = (cx - board_cx) * ux + (cy - board_cy) * uy
            t_max = max(t_max, t)

        cx = board_cx + (t_max + offset) * ux
        cy = board_cy + (t_max + offset) * uy

        tip = (cx + ux * arrow_len, cy + uy * arrow_len)
        back = (cx - ux * arrow_len * 0.70, cy - uy * arrow_len * 0.70)

        px, py = -uy, ux
        left = (back[0] + px * arrow_width / 2, back[1] + py * arrow_width / 2)
        right = (back[0] - px * arrow_width / 2, back[1] - py * arrow_width / 2)

        arrows.append({"dir": idx, "tri": [tip, left, right]})

    return arrows

# =====================
# RENDER
# =====================
def render_scene(cells, centers, state: GameState):
    arrows_world = build_arrows_world(cells, centers, 0.0, 0.0)

    # bounds (tiles + arrows)
    minx = miny = 1e18
    maxx = maxy = -1e18

    for c in cells:
        cx, cy = centers[c]
        for x, y in hex_corners(cx, cy):
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)

    for a in arrows_world:
        for x, y in a["tri"]:
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)

    w = int((maxx - minx) + 2 * PADDING)
    h = int((maxy - miny) + 2 * PADDING)
    sx = -minx + PADDING
    sy = -miny + PADDING

    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 34)
        font_mid = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()
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
            draw.polygon(pts, outline=OUTLINE, width=3)

            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font_big)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx + sx - tw / 2, cy + sy - th / 2), text, fill=TEXT_COLOR, font=font_big)
        else:
            draw.polygon(pts, outline=(55, 55, 66), width=2)

    # arrows (convert to image coords + store for hit testing)
    arrows_img = []
    for a in arrows_world:
        tri = [(x + sx, y + sy) for x, y in a["tri"]]
        draw.polygon(tri, fill=ARROW_FILL)
        draw.polygon(tri, outline=ARROW_OUTLINE, width=3)

        # label centered
        cx = sum(p[0] for p in tri) / 3
        cy = sum(p[1] for p in tri) / 3
        label = MOVE_LABELS.get(a["dir"], "")
        bbox = draw.textbbox((0, 0), label, font=font_mid)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2), label, fill=(25, 25, 30), font=font_mid)

        arrows_img.append({"dir": a["dir"], "tri": tri})

    # HUD
    draw.text((10, 10), f"Score: {state.score}", fill=(235, 235, 235), font=font_small)
    if state.game_over:
        draw.text((10, 30), "GAME OVER", fill=(255, 180, 180), font=font_small)

    return img, arrows_img

# =====================
# STREAMLIT APP (MOBILE FIRST)
# =====================
st.set_page_config(page_title="Hex 2048 Base-3 (Mobile)", layout="wide")

st.markdown(
    """
    <style>
      html, body { height: 100%; overflow: hidden !important; }
      #root, .stApp { height: 100vh; overflow: hidden !important; }
      header, footer { display: none !important; }
      [data-testid="stSidebar"] { display: none !important; }
      .block-container { padding: 10px !important; height: 100vh; overflow: hidden !important; }
      button[kind="primary"] { width: 100% !important; height: 52px !important; font-size: 18px !important; border-radius: 16px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

cells = generate_hex_cells(RADIUS)
dir_lines = build_direction_lines(cells, DIRECTIONS)

centers_local = {c: axial_to_pixel(c[0], c[1], size=HEX_SIZE) for c in cells}
cx0 = sum(x for x, _ in centers_local.values()) / len(centers_local)
cy0 = sum(y for _, y in centers_local.values()) / len(centers_local)
centers = {c: (centers_local[c][0] - cx0, centers_local[c][1] - cy0) for c in cells}

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)
if "last_click_time" not in st.session_state:
    st.session_state.last_click_time = None

state: GameState = st.session_state.state

top_left, top_right = st.columns([2, 1])
with top_left:
    st.markdown("### Hex 2048 Base-3")
with top_right:
    if st.button("Reset", type="primary"):
        st.session_state.state = new_game(cells)
        st.rerun()

img, arrows_img = render_scene(cells, centers, state)

click = streamlit_image_coordinates(img, key="board", use_column_width=True)

# Tap arrows
if click and "x" in click and "y" in click and "time" in click:
    if st.session_state.last_click_time != click["time"]:
        st.session_state.last_click_time = click["time"]

        px = float(click["x"])
        py = float(click["y"])

        clicked_dir: Optional[int] = None
        for a in arrows_img:
            p0, p1, p2 = a["tri"]
            if point_in_triangle(px, py, p0, p1, p2):
                clicked_dir = a["dir"]
                break

        if clicked_dir is not None and not state.game_over:
            do_move(state, clicked_dir, dir_lines)
            st.rerun()

if state.game_over:
    st.error("No moves left. Tap Reset.")

