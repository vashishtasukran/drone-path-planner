import math
import time
import pybullet as p
import pybullet_data
from planner import astar, SIZE

CELL = 0.5               # size of one grid cell in world units
FRAMES_PER_STEP = 24     # how long each grid move lasts (higher = slower)

START1, GOAL1 = (0, 5, 5), (9, 5, 5)
START2, GOAL2 = (9, 4, 4), (0, 4, 4)
PROTECTED = {START1, GOAL1, START2, GOAL2}   # blocks can't be placed here


def to_world(cell):
    return [c * CELL for c in cell]


# ---------- scene ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.loadURDF("plane.urdf", [0, 0, -0.3])
p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=50, cameraPitch=-30,
                             cameraTargetPosition=[2.25, 2.25, 2.25])

half = [CELL / 2] * 3
wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.8, 0.2, 0.2, 0.6])
cursor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CELL / 2.2] * 3, rgbaColor=[0, 1, 0, 0.5])
d1_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 0.5, 1, 1])
d2_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0.6, 0, 1])

drone1 = p.createMultiBody(0, -1, d1_vis, to_world(START1))
drone2 = p.createMultiBody(0, -1, d2_vis, to_world(START2))
OFFSCREEN = [0, 0, -5]
cursor = p.createMultiBody(0, -1, cursor_vis, OFFSCREEN)

# ---------- walls ----------
blocked = set()
wall_bodies = {}


def add_block(cell):
    if cell in blocked or cell in PROTECTED:
        return False
    blocked.add(cell)
    wall_bodies[cell] = p.createMultiBody(0, -1, wall_vis, to_world(cell))
    return True


def remove_block(cell):
    if cell not in blocked:
        return False
    blocked.remove(cell)
    p.removeBody(wall_bodies.pop(cell))
    return True


for y in range(SIZE):
    for z in range(SIZE):
        if (y, z) not in {(0, 0), (9, 9)}:      # two gaps, as in v1
            add_block((5, y, z))

# ---------- controls (buttons and a slider in the side panel) ----------
btn_add_mode = p.addUserDebugParameter("MODE: ADD blocks", 1, 0, 1)
btn_remove_mode = p.addUserDebugParameter("MODE: REMOVE blocks", 1, 0, 1)
btn_replan = p.addUserDebugParameter("REPLAN + RESTART", 1, 0, 1)
layer_slider = p.addUserDebugParameter("edit layer (x)", 0, SIZE - 1, 5)

last_click = {}


def clicked(button_id):
    value = p.readUserDebugParameter(button_id)
    before = last_click.get(button_id, value)
    last_click[button_id] = value
    return value != before          # a button's value changes each time it is clicked


# ---------- layer grid (shows which plane you are editing) ----------
layer_lines = []


def draw_layer(layer):
    for line_id in layer_lines:
        p.removeUserDebugItem(line_id)
    layer_lines.clear()
    x = layer * CELL
    lo, hi = -0.5 * CELL, (SIZE - 0.5) * CELL
    for i in range(SIZE + 1):
        v = (i - 0.5) * CELL
        layer_lines.append(p.addUserDebugLine([x, v, lo], [x, v, hi], [0.3, 1, 0.3], 1))
        layer_lines.append(p.addUserDebugLine([x, lo, v], [x, hi, v], [0.3, 1, 0.3], 1))


# ---------- turn a mouse position into a grid cell on the chosen layer ----------
def mouse_to_cell(mouse_x, mouse_y, layer):
    cam = p.getDebugVisualizerCamera()
    width, height = cam[0], cam[1]
    cam_forward, horizontal, vertical = cam[5], cam[6], cam[7]
    dist, target = cam[10], cam[11]

    cam_pos = [target[i] - dist * cam_forward[i] for i in range(3)]
    ray_forward = [target[i] - cam_pos[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in ray_forward))
    ray_forward = [v * 10000.0 / length for v in ray_forward]

    ray_to = [cam_pos[i] + ray_forward[i]
              - 0.5 * horizontal[i] + 0.5 * vertical[i]
              + float(mouse_x) / width * horizontal[i]
              - float(mouse_y) / height * vertical[i]
              for i in range(3)]
    direction = [ray_to[i] - cam_pos[i] for i in range(3)]

    # where does this line hit the plane x = layer?
    if abs(direction[0]) < 1e-9:
        return None
    t = (layer * CELL - cam_pos[0]) / direction[0]
    if t < 0:
        return None
    hit_y = cam_pos[1] + t * direction[1]
    hit_z = cam_pos[2] + t * direction[2]

    cell = (layer, round(hit_y / CELL), round(hit_z / CELL))
    if 0 <= cell[1] < SIZE and 0 <= cell[2] < SIZE:
        return cell
    return None


# ---------- planning and metrics ----------
line_ids = []
text_ids = [-1] * 4
mode_text_id = -1


def count_collisions(path1, path2):
    hits = 0
    for i in range(max(len(path1), len(path2))):
        a = path1[min(i, len(path1) - 1)]
        b = path2[min(i, len(path2) - 1)]
        if a == b:
            hits += 1
    return hits


def plan_and_show():
    t0 = time.perf_counter()
    s1, s2 = {}, {}

    path1 = astar(blocked, START1, GOAL1, stats=s1)
    found1 = path1 is not None
    if not found1:
        path1 = [START1]

    # drone 2 treats drone 1's whole route as walls (same idea as v1)
    blocked2 = (blocked | set(path1)) - {START2, GOAL2}
    path2 = astar(blocked2, START2, GOAL2, stats=s2)
    found2 = path2 is not None
    if not found2:
        path2 = [START2]

    ms = (time.perf_counter() - t0) * 1000

    # draw the planned routes
    for line_id in line_ids:
        p.removeUserDebugItem(line_id)
    line_ids.clear()
    for path, color in ((path1, [0, 0.7, 1]), (path2, [1, 0.5, 0])):
        for a, b in zip(path, path[1:]):
            line_ids.append(p.addUserDebugLine(to_world(a), to_world(b), color, 3))

    # show metrics as text in the scene (and in the console)
    lines = [
        f"Drone 1: {len(path1) - 1} moves, explored {s1['explored']}" if found1 else "Drone 1: NO PATH",
        f"Drone 2: {len(path2) - 1} moves, explored {s2['explored']}" if found2 else "Drone 2: NO PATH",
        f"Blocks: {len(blocked)} | replan: {ms:.1f} ms",
        f"Collisions: {count_collisions(path1, path2)}",
    ]
    for i, text in enumerate(lines):
        text_ids[i] = p.addUserDebugText(text, [-1, 2, 5.5 - 0.45 * i],
                                         textColorRGB=[1, 1, 1], textSize=1.2,
                                         replaceItemUniqueId=text_ids[i])
        print(text)
    print("-----")

    return path1, path2


def show_mode(mode, layer):
    global mode_text_id
    mode_text_id = p.addUserDebugText(f"MODE: {mode.upper()} | editing layer x = {layer}",
                                      [-1, 2, 3.4], textColorRGB=[0.4, 1, 0.4], textSize=1.3,
                                      replaceItemUniqueId=mode_text_id)


mode = "add"
layer = 5
draw_layer(layer)
show_mode(mode, layer)
path1, path2 = plan_and_show()
step, frame = 0, 0

# ---------- main loop ----------
while p.isConnected():
    # buttons and slider
    if clicked(btn_add_mode):
        mode = "add"
        show_mode(mode, layer)
    if clicked(btn_remove_mode):
        mode = "remove"
        show_mode(mode, layer)
    new_layer = round(p.readUserDebugParameter(layer_slider))
    if new_layer != layer:
        layer = new_layer
        draw_layer(layer)
        show_mode(mode, layer)

    changed = clicked(btn_replan)

    # mouse: moving updates the green cube, left click edits
    for e in p.getMouseEvents():
        if len(e) < 5:
            continue
        event_type, mx, my, button, state = e
        cell = mouse_to_cell(mx, my, layer)
        if event_type == 1:                                   # mouse moved
            p.resetBasePositionAndOrientation(
                cursor, to_world(cell) if cell else OFFSCREEN, [0, 0, 0, 1])
        elif event_type == 2 and button == 0 and (state & p.KEY_WAS_TRIGGERED):
            if cell is None:
                continue
            if mode == "add":
                changed = add_block(cell) or changed
            else:
                changed = remove_block(cell) or changed

    if changed:
        path1, path2 = plan_and_show()
        step, frame = 0, 0

    # move each drone smoothly from one cell to the next
    t = frame / FRAMES_PER_STEP
    for body, path in ((drone1, path1), (drone2, path2)):
        a = to_world(path[min(step, len(path) - 1)])
        b = to_world(path[min(step + 1, len(path) - 1)])
        pos = [a[i] + (b[i] - a[i]) * t for i in range(3)]
        p.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1])

    frame += 1
    if frame >= FRAMES_PER_STEP:
        frame = 0
        if step < max(len(path1), len(path2)) - 1:
            step += 1

    p.stepSimulation()
    time.sleep(1 / 240)