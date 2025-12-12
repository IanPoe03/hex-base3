import base64
import io
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# =====================
# MOBILE-ONLY CONFIG
# =====================
RADIUS = 2
HEX_SIZE = 58
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
]

# Axial (pointy-top)
DIRECTIONS = [
    (1, 0),    # 0
    (1, -1),   # 1
    (0, -1),   # 2
    (-1, 0),   # 3
    (-1, 1),   # 4
    (0, 1),    # 5
]

# Sliding happens along -DIRECTIONS[dir_index]
MOVE = {"l": 0, "r": 3, "ul": 5, "ur": 4, "dl": 1, "dr": 2}

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
    out = []
    score = 0
    i = 0
    while i < len(values):
        v = values[i]
        j = i + 1
        while j < len(values) and values[j] == v:
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
                if vals[i] != 0 and vals[i] == vals[i+1] == vals[i+2]:
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
# RENDER -> PNG DATA URI
# =====================
def render_board_png(cells, centers, state: GameState) -> bytes:
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
        font_big = ImageFont.truetype("DejaVuSans.ttf", 34)
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
        draw.text((10, 30), "GAME OVER", fill=(255, 180, 180), font=font_small)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return "data:image/png;base64," + b64


# =====================
# APP (MOBILE ONLY)
# =====================
st.set_page_config(page_title="Hex 2048 Base-3 (Mobile)", layout="wide")

# No-scroll + hide chrome
st.markdown(
    """
    <style>
      html, body { height: 100%; overflow: hidden !important; }
      #root, .stApp { height: 100vh; overflow: hidden !important; }
      header, footer { display: none !important; }
      .block-container { padding: 0 !important; height: 100vh; overflow: hidden !important; }
      [data-testid="stSidebar"] { display: none !important; }
      body { overscroll-behavior: none; }
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
if "last_action_id" not in st.session_state:
    st.session_state.last_action_id = None

state: GameState = st.session_state.state

data_uri = png_to_data_uri(render_board_png(cells, centers, state))

# IMPORTANT: use Streamlit.setComponentValue(...) instead of window.location
# and return {"mv": "...", "id": "..."} so Python can de-dupe.
html = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
<style>
  html, body { margin:0; padding:0; width:100%; height:100%; overflow:hidden; background: rgb(18,18,22); }
  #wrap { position:fixed; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; touch-action:none; }
  #top { width:100%; display:flex; justify-content:space-between; align-items:center;
         padding: 12px 14px; box-sizing:border-box; color: rgba(255,255,255,0.92);
         font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  #board { max-width:96vw; max-height:82vh; border-radius:18px; border:1px solid rgba(255,255,255,0.10);
           box-shadow: 0 10px 30px rgba(0,0,0,0.35);
           user-select:none; -webkit-user-drag:none; touch-action:none; }
  #hint { margin-top:10px; font-size:13px; color: rgba(255,255,255,0.70);
          font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  button { font-size:14px; padding:8px 12px; border-radius:14px;
           border: 1px solid rgba(255,255,255,0.16);
           background: rgba(255,255,255,0.08);
           color: rgba(255,255,255,0.92); }
</style>
</head>
<body>
  <div id="wrap">
    <div id="top">
      <div style="font-size:16px;">Hex 2048 Base-3</div>
      <button id="resetBtn">Reset</button>
    </div>
    <img id="board" />
    <div id="hint">Swipe on the board (6 directions)</div>
  </div>

<script>
  // Streamlit component bridge
  function send(val) {
    // include a unique id so python can ignore repeats
    const payload = { ...val, id: String(Date.now()) + "-" + String(Math.random()) };
    if (window.Streamlit && window.Streamlit.setComponentValue) {
      window.Streamlit.setComponentValue(payload);
    }
  }

  // Prevent scrolling (must be passive:false)
  document.addEventListener('touchmove', function(e) { e.preventDefault(); }, { passive: false });

  const img = document.getElementById("board");
  const resetBtn = document.getElementById("resetBtn");

  let sx=0, sy=0;

  function classify(dx, dy) {
    const dist = Math.hypot(dx, dy);
    if (dist < 22) return null;

    // 0 right, 60 up-right, 120 up-left, 180 left, -120 down-left, -60 down-right
    const ang = Math.atan2(-dy, dx) * 180 / Math.PI;
    const k = Math.round(ang / 60);
    const bucket = ((k % 6) + 6) % 6;

    // 0: r, 1: ur, 2: ul, 3: l, 4: dl, 5: dr
    if (bucket === 0) return "r";
    if (bucket === 1) return "ur";
    if (bucket === 2) return "ul";
    if (bucket === 3) return "l";
    if (bucket === 4) return "dl";
    if (bucket === 5) return "dr";
    return null;
  }

  img.addEventListener("touchstart", (e) => {
    const t = e.touches[0];
    sx = t.clientX; sy = t.clientY;
  }, { passive: true });

  img.addEventListener("touchend", (e) => {
    const t = e.changedTouches[0];
    const dx = t.clientX - sx;
    const dy = t.clientY - sy;
    const mv = classify(dx, dy);
    if (mv) send({ mv: mv });
  }, { passive: true });

  resetBtn.addEventListener("click", () => send({ reset: 1 }));

  // Let Streamlit size the iframe nicely
  if (window.Streamlit && window.Streamlit.setFrameHeight) {
    window.Streamlit.setFrameHeight(900);
  }
</script>
</body>
</html>
"""

# Render component, then set image src safely AFTER render (simple string replace)
# This avoids any weird HTML parsing breakage around big base64 strings.
html_with_img = html.replace('<img id="board" />', f'<img id="board" src="{data_uri}" />')

event = components.html(html_with_img, height=900)

# event comes back as dict from send(...)
mv_action: Optional[str] = None
do_reset = False

if isinstance(event, dict):
    if event.get("id") != st.session_state.last_action_id:
        st.session_state.last_action_id = event.get("id")
        if event.get("reset") == 1:
            do_reset = True
        if event.get("mv") in MOVE:
            mv_action = event.get("mv")

if do_reset:
    st.session_state.state = new_game(cells)
    st.rerun()

if mv_action and not state.game_over:
    do_move(state, MOVE[mv_action], dir_lines)
    st.rerun()

