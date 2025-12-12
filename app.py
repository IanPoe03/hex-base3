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

# "Feel-correct" mapping (same as the fixed pygame build you liked):
# because the move implementation slides along -DIRECTIONS[dir_index]
MOVE = {
    "left": 0,
    "right": 3,
    "up_left": 5,
    "up_right": 4,
    "down_left": 1,
    "down_right": 2,
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
    Merge triples of identical values from the front (after compression).
    Example: [1,1,1,1] -> [3,1]
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
# CLICKABLE ARROWS (HIT TEST)
# =====================

def point_in_triangle(px: float, py: float, a, b, c) -> bool:
    # Barycentric technique
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


def build_corner_arrows(
    cells: List[Tuple[int, int]],
    centers: Dict[Tuple[int, int], Tuple[float, float]],
    board_cx: float,
    board_cy: float,
) -> List[dict]:
    """
    Build outward-pointing arrow triangles located just outside the board corners.
    Each arrow corresponds directly to a dir_index used by do_move().
    """
    arrows = []
    offset = HEX_SIZE * 1.4
    arrow_len = 26
    arrow_width = 22

    for idx, (dq, dr) in enumerate(DIRECTIONS):
        # move happens along -direction
        mvx, mvy = axial_to_pixel(-dq, -dr, size=HEX_SIZE)
        L = math.hypot(mvx, mvy)
        if L == 0:
            continue
        ux, uy = mvx / L, mvy / L

        # farthest tile center along this movement direction => corner
        t_max = -float("inf")
        for c in cells:
            cx, cy = centers[c]
            vx, vy = cx - board_cx, cy - board_cy
            t = vx * ux + vy * uy
            if t > t_max:
                t_max = t

        # arrow center outside the corner
        cx = board_cx + (t_max + offset) * ux
        cy = board_cy + (t_max + offset) * uy

        tip = (cx + ux * arrow_len, cy + uy * arrow_len)
        back_center = (cx - ux * arrow_len * 0.6, cy - uy * arrow_len * 0.6)

        px, py = -uy, ux
        left = (back_center[0] + px * arrow_width / 2, back_center[1] + py * arrow_width / 2)
        right = (back_center[0] - px * arrow_width / 2, back_center[1] - py * arrow_width / 2)

        arrows.append({"dir": idx, "points": [tip, left, right]})

    return arrows


# =====================
# RENDERING (PIL) + COORD TRANSFORM
# =====================

def render_board_image(
    cells: List[Tuple[int, int]],
    centers: Dict[Tuple[int, int], Tuple[float, float]],
    state: GameState,
    arrows: List[dict],
) -> Tuple[Image.Image, List[dict]]:
    """
    Returns:
      - PIL image
      - arrows with points transformed into image pixel coords for hit testing
    """
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for c in cells:
        cx, cy = centers[c]
        for x, y in hex_corners(cx, cy):
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    for a in arrows:
        for x, y in a["points"]:
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

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

    # tiles
    for c in cells:
        cx, cy = centers[c]
        pts = [(x + sx, y + sy) for x, y in hex_corners(cx, cy)]
        draw.polygon(pts, fill=GRID_COLOR)

        v = state.board[c]
        if v != 0:
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

    # simple overlay text
    if state.game_over:
        msg = "GAME OVER"
        sub = "Press Reset"
        mb = draw.textbbox((0, 0), msg, font=font_big)
        sb = draw.textbbox((0, 0), sub, font=font_small)
        mw, mh = mb[2] - mb[0], mb[3] - mb[1]
        sw, sh = sb[2] - sb[0], sb[3] - sb[1]
        draw.text((w / 2 - mw / 2, 12), msg, fill=(240, 240, 240), font=font_big)
        draw.text((w / 2 - sw / 2, 12 + mh + 4), sub, fill=(220, 220, 220), font=font_small)

    return img, arrows_img


# =====================
# STREAMLIT APP
# =====================

st.set_page_config(page_title="Hex 2048 Base-3", layout="centered")

cells = generate_hex_cells(RADIUS)
dir_lines = build_direction_lines(cells, DIRECTIONS)

# centers in local coords (then recenter to (0,0))
centers_local = {c: axial_to_pixel(c[0], c[1], size=HEX_SIZE) for c in cells}
cx0 = sum(x for x, _ in centers_local.values()) / len(centers_local)
cy0 = sum(y for _, y in centers_local.values()) / len(centers_local)
centers = {c: (centers_local[c][0] - cx0, centers_local[c][1] - cy0) for c in cells}

BOARD_CX, BOARD_CY = 0.0, 0.0

if "state" not in st.session_state:
    st.session_state.state = new_game(cells)
if "last_click_t" not in st.session_state:
    st.session_state.last_click_t = None

state: GameState = st.session_state.state

st.title("Hex 2048 (Base-3, Triple Merge)")
st.caption("Click the arrows around the board to move. Spawn: 1 (75%), 3 (25%). Merge: 3-in-a-row → value × 3.")

# ---- Keyboard shortcuts (no visible move buttons) ----
# We create hidden-ish buttons in the sidebar so add_shortcuts has widget keys to target.
with st.sidebar:
    st.subheader("Controls")
    st.write("Keyboard:")
    st.write("← left | → right | ↑ up-left | Shift+↑ up-right | ↓ down-right | Shift+↓ down-left")
    reset_clicked = st.button("Reset", key="reset_btn")

    # Hidden move buttons (tiny / tucked away in sidebar)
    # These exist ONLY so keyboard shortcuts can trigger them.
    st.button("Move Left", key="mv_left", disabled=state.game_over)
    st.button("Move Right", key="mv_right", disabled=state.game_over)
    st.button("Move Up-Left", key="mv_ul", disabled=state.game_over)
    st.button("Move Up-Right", key="mv_ur", disabled=state.game_over)
    st.button("Move Down-Left", key="mv_dl", disabled=state.game_over)
    st.button("Move Down-Right", key="mv_dr", disabled=state.game_over)

# Map keys to those widget keys
add_shortcuts(
    mv_left="arrowleft",
    mv_right="arrowright",
    mv_ul="arrowup",
    mv_ur="shift+arrowup",
    mv_dr="arrowdown",
    mv_dl="shift+arrowdown",
)

# When those sidebar buttons are triggered (by click OR shortcut), apply moves
# Note: st.button returns True only on the rerun it was activated.
# We re-check them via session_state on each rerun by reading widget values.
# Streamlit sets the button value in session_state for that run.
def _consume_button(key: str) -> bool:
    v = bool(st.session_state.get(key, False))
    if v:
        # reset so it doesn't repeatedly fire
        st.session_state[key] = False
    return v

if reset_clicked or _consume_button("reset_btn"):
    st.session_state.state = new_game(cells)
    state = st.session_state.state
    st.rerun()

# Handle keyboard-triggered move buttons
if _consume_button("mv_left"):
    do_move(state, MOVE["left"], dir_lines)
    st.rerun()
if _consume_button("mv_right"):
    do_move(state, MOVE["right"], dir_lines)
    st.rerun()
if _consume_button("mv_ul"):
    do_move(state, MOVE["up_left"], dir_lines)
    st.rerun()
if _consume_button("mv_ur"):
    do_move(state, MOVE["up_right"], dir_lines)
    st.rerun()
if _consume_button("mv_dl"):
    do_move(state, MOVE["down_left"], dir_lines)
    st.rerun()
if _consume_button("mv_dr"):
    do_move(state, MOVE["down_right"], dir_lines)
    st.rerun()

# ---- Render board + clickable arrows ----
arrows_world = build_corner_arrows(cells, centers, BOARD_CX, BOARD_CY)
img, arrows_img = render_board_image(cells, centers, state, arrows_world)

# This component displays the image and returns click coords {x,y} (and time)
click = streamlit_image_coordinates(img, key="board_click")

# Score display
st.metric("Score", state.score)

# If user clicked the image, check if click was inside any arrow triangle.
if click and "x" in click and "y" in click and "time" in click:
    # de-bounce: only act on new clicks
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
    st.error("No moves left. Use Reset to play again.")

