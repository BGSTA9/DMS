# =============================================================================
# simulation/carla_simulation.py
#
# CARLA semantic LiDAR point cloud visualization for the DMS simulation panel.
#
# Renders a live semantic LiDAR point cloud from a CARLA server into a
# pygame.Surface (WINDOW_WIDTH × WINDOW_HEIGHT//2) using a bird's-eye /
# slight perspective transform.
#
# Architecture:
#   CARLASimulationManager — public API (drop-in replacement for old sim)
#     └── CarlaClient      — background connection + actor management
#
# Sensor:
#   sensor.lidar.ray_cast_semantic attached to ego vehicle roof.
#   Points are coloured by their CARLA semantic tag using the palette
#   defined in SEMANTIC_COLORS.
#
# DMS State → CARLA Behaviour:
#   ALERT    → autopilot on, normal speed
#   DROWSY   → autopilot on, reduced speed, right blinker
#   SLEEPING → disengage autopilot, steer right + brake to stop, hazards
#
# Graceful fallback:
#   If `carla` is not installed or the server is down, a styled placeholder
#   surface is returned. A background thread keeps retrying the connection.
# =============================================================================

import math
import threading
import time
import random
import numpy as np
import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    CARLA_HOST, CARLA_PORT, CARLA_MAP, CARLA_TIMEOUT,
    CARLA_OFFLINE_RETRY_S, CARLA_SYNC_DELTA,
    CARLA_VEHICLE_BP, CARLA_TM_PORT,
    CARLA_LIDAR_RANGE, CARLA_LIDAR_CHANNELS, CARLA_LIDAR_PPS,
    CARLA_OFFLINE_BG,
    CARLA_NORMAL_SPEED, CARLA_DROWSY_SPEED,
)
from simulation.carla_hud import CarlaHUD

# ── Try importing carla ──────────────────────────────────────────────────────
try:
    import carla as _carla_module
    _CARLA_AVAILABLE = True
except ImportError:
    _carla_module = None
    _CARLA_AVAILABLE = False

_PANEL_W = WINDOW_WIDTH
_PANEL_H = WINDOW_HEIGHT // 2

# ── Semantic LiDAR colour palette (tag index → RGB) ─────────────────────────
SEMANTIC_COLORS = {
    0:  (100, 100, 100),   # Unlabeled
    1:  (220, 100,  40),   # Building       orange
    2:  (190, 153, 153),   # Fence
    3:  (180, 130,  70),   # Other
    4:  ( 60, 100, 200),   # Pedestrian     blue
    5:  (153, 153, 153),   # Pole
    6:  (220, 220,   0),   # RoadLine       yellow
    7:  (200,  30,  30),   # Road           red
    8:  (220,  20, 220),   # SideWalk       magenta
    9:  ( 40, 200,  60),   # Vegetation     green
    10: ( 60, 150, 220),   # Vehicles       blue
    11: (150, 100,  90),   # Wall
    12: (220, 200,   0),   # TrafficSign
    13: ( 60, 200, 220),   # Sky
    14: (130, 130, 130),   # Ground
    15: (150, 100, 100),   # Bridge
    16: (230, 150, 140),   # RailTrack
    17: (180, 165, 180),   # GuardRail
    18: (250, 170,  30),   # TrafficLight
    19: (110, 190, 160),   # Static
    20: (170, 120,  50),   # Dynamic
    21: ( 45,  60, 150),   # Water
    22: (145, 170, 100),   # Terrain
}

# Point size by semantic tag (in pixels)
_POINT_SIZE = {
    7: 2, 8: 2, 22: 2, 14: 2,          # Road, SideWalk, Terrain, Ground
    6: 2,                                # RoadLine
    1: 3, 11: 3, 15: 3, 2: 3, 17: 3,   # Buildings, Wall, Bridge, Fence, GuardRail
    9: 3,                                # Vegetation
    10: 4, 4: 4, 20: 4,                 # Vehicles, Pedestrians, Dynamic
}
_DEFAULT_POINT_SIZE = 2


# =============================================================================
# CarlaClient — manages CARLA connection, ego vehicle, and LiDAR sensor
# =============================================================================

class _CarlaClient:
    """
    Background CARLA client.  Connects in a daemon thread, spawns ego
    vehicle + semantic LiDAR, stores point cloud in a thread-safe buffer.
    """

    def __init__(self):
        self._connected = False
        self._shutting_down = False
        self._lock = threading.Lock()
        self._setup_done = threading.Event()

        # CARLA handles
        self._client = None
        self._world = None
        self._vehicle = None
        self._lidar = None
        self._tm = None

        # Latest point cloud: Nx4 float32 (x, y, z, tag)
        self._points: np.ndarray | None = None

        # Vehicle telemetry
        self._speed_kmh: float = 0.0

        # Pull-over state
        self._is_pulling_over = False
        self._is_stopped = False

        # Start background connector
        if _CARLA_AVAILABLE:
            t = threading.Thread(target=self._connect_loop, daemon=True)
            t.start()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def speed_kmh(self) -> float:
        return self._speed_kmh

    @property
    def is_pulling_over(self) -> bool:
        return self._is_pulling_over

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped

    # ── Public API ────────────────────────────────────────────────────────────

    def get_points(self) -> np.ndarray | None:
        """Return latest point cloud (Nx4) or None."""
        with self._lock:
            return self._points.copy() if self._points is not None else None

    def tick(self) -> None:
        """Advance CARLA world by one step (synchronous mode)."""
        if self._connected and self._world is not None:
            try:
                self._world.tick()
                # Read speed
                if self._vehicle is not None:
                    v = self._vehicle.get_velocity()
                    self._speed_kmh = 3.6 * math.sqrt(
                        v.x ** 2 + v.y ** 2 + v.z ** 2
                    )
            except Exception:
                self._connected = False

    def set_driver_state(self, state: str, drowsiness_score: float) -> None:
        """Map DMS state to CARLA vehicle behaviour."""
        if not self._connected or self._vehicle is None:
            return
        try:
            if state == "ALERT":
                self._handle_alert(drowsiness_score)
            elif state == "DROWSY":
                self._handle_drowsy(drowsiness_score)
            elif state == "SLEEPING":
                self._handle_sleeping()
        except Exception:
            pass

    def reset(self) -> None:
        """Teleport vehicle, re-enable autopilot."""
        if not self._connected or self._vehicle is None:
            return
        carla = _carla_module
        try:
            self._is_pulling_over = False
            self._is_stopped = False

            spawns = self._world.get_map().get_spawn_points()
            if spawns:
                self._vehicle.set_transform(random.choice(spawns))

            self._vehicle.apply_control(
                carla.VehicleControl(throttle=0, brake=1.0, steer=0)
            )
            # Let physics settle
            for _ in range(5):
                self._world.tick()

            self._vehicle.set_autopilot(True, self._tm.get_port())
            self._tm.vehicle_percentage_speed_difference(
                self._vehicle, CARLA_NORMAL_SPEED
            )
            self._vehicle.set_light_state(carla.VehicleLightState.NONE)
        except Exception:
            pass

    def destroy(self) -> None:
        """Clean up actors and restore async mode."""
        self._shutting_down = True
        self._connected = False
        try:
            if self._lidar is not None:
                self._lidar.stop()
                self._lidar.destroy()
            if self._vehicle is not None:
                self._vehicle.destroy()
            if self._world is not None:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                self._world.apply_settings(settings)
        except Exception:
            pass
        self._lidar = None
        self._vehicle = None

    # ── Background Connection ────────────────────────────────────────────────

    def _connect_loop(self) -> None:
        while not self._shutting_down:
            if not self._connected:
                try:
                    self._setup_world()
                    self._connected = True
                    self._setup_done.set()
                except Exception:
                    self._connected = False
                    time.sleep(CARLA_OFFLINE_RETRY_S)
            else:
                time.sleep(0.5)

    def _setup_world(self) -> None:
        carla = _carla_module

        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)

        world = client.get_world()
        if CARLA_MAP not in world.get_map().name:
            world = client.load_world(CARLA_MAP)

        # Synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CARLA_SYNC_DELTA
        world.apply_settings(settings)

        # Traffic Manager
        tm = client.get_trafficmanager(CARLA_TM_PORT)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        # Ego vehicle
        bp_lib = world.get_blueprint_library()
        veh_bp = bp_lib.find(CARLA_VEHICLE_BP)
        spawns = world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points")

        vehicle = None
        for sp in random.sample(spawns, min(10, len(spawns))):
            vehicle = world.try_spawn_actor(veh_bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Could not spawn vehicle")

        vehicle.set_autopilot(True, tm.get_port())

        # ── Semantic LiDAR ───────────────────────────────────────────────────
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("upper_fov",           "15.0")
        lidar_bp.set_attribute("lower_fov",          "-25.0")
        lidar_bp.set_attribute("channels",            str(CARLA_LIDAR_CHANNELS))
        lidar_bp.set_attribute("range",               str(CARLA_LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second",   str(CARLA_LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency",  "20.0")

        mount = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        lidar = world.spawn_actor(lidar_bp, mount, attach_to=vehicle)
        lidar.listen(self._on_lidar)

        self._client = client
        self._world = world
        self._vehicle = vehicle
        self._lidar = lidar
        self._tm = tm

        # Initial tick to let sensor warm up
        world.tick()

    def _on_lidar(self, data) -> None:
        """Callback: parse SemanticLidarMeasurement into Nx4 array."""
        # Each detection: [x, y, z, cos_inc_angle, object_idx, object_tag]
        # We need x, y, z, object_tag
        raw = np.frombuffer(data.raw_data, dtype=np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("cos_inc", np.float32),
            ("obj_idx", np.uint32),
            ("obj_tag", np.uint32),
        ]))
        pts = np.column_stack([
            raw["x"], raw["y"], raw["z"],
            raw["obj_tag"].astype(np.float32),
        ])
        with self._lock:
            self._points = pts

    # ── Driver State Handlers ────────────────────────────────────────────────

    def _handle_alert(self, drowsiness_score: float) -> None:
        carla = _carla_module
        if self._is_pulling_over or self._is_stopped:
            self._is_pulling_over = False
            self._is_stopped = False
            self._vehicle.set_autopilot(True, self._tm.get_port())

        # Speed linearly modulated: 0→0%, 1→~83% slower
        # drowsiness_score 0 → 120 km/h, 1 → 20 km/h
        # vehicle_percentage_speed_difference: 0 means road limit,
        # positive means slower.  We want 0–83% reduction.
        pct = drowsiness_score * 83.0
        self._tm.vehicle_percentage_speed_difference(self._vehicle, pct)
        self._vehicle.set_light_state(carla.VehicleLightState.NONE)

    def _handle_drowsy(self, drowsiness_score: float) -> None:
        carla = _carla_module
        if self._is_pulling_over:
            return

        if self._is_stopped:
            self._is_stopped = False
            self._vehicle.set_autopilot(True, self._tm.get_port())

        self._tm.vehicle_percentage_speed_difference(
            self._vehicle, CARLA_DROWSY_SPEED
        )
        self._vehicle.set_light_state(carla.VehicleLightState.RightBlinker)

    def _handle_sleeping(self) -> None:
        carla = _carla_module
        if self._is_stopped:
            return

        if not self._is_pulling_over:
            self._vehicle.set_autopilot(False)
            self._is_pulling_over = True

        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        if speed < 0.3:
            self._vehicle.apply_control(
                carla.VehicleControl(throttle=0, brake=1.0, steer=0)
            )
            self._vehicle.set_light_state(
                carla.VehicleLightState(
                    carla.VehicleLightState.LeftBlinker
                    | carla.VehicleLightState.RightBlinker
                )
            )
            self._is_stopped = True
            self._is_pulling_over = False
        else:
            self._vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.3, brake=0.8)
            )
            self._vehicle.set_light_state(carla.VehicleLightState.RightBlinker)


# =============================================================================
# Point Cloud → Pygame Surface rendering
# =============================================================================

def _render_points(
    surface: pygame.Surface,
    points:  np.ndarray,
    lidar_range: float,
) -> None:
    """
    Project Nx4 point cloud (x, y, z, tag) onto the pygame surface using
    a bird's-eye / slight perspective transform and draw semantic-coloured
    filled circles.
    """
    w = surface.get_width()
    h = surface.get_height()
    cx = w // 2
    cy = h // 2

    scale = (h * 0.45) / lidar_range
    tilt = 0.75  # perspective foreshortening on forward axis

    # Vectorised projection
    fx =  points[:, 0]            # forward
    fy = -points[:, 1]            # lateral (negated so left=screen left)
    z  =  points[:, 2]
    tags = points[:, 3].astype(np.int32)

    px = (cx + fy * scale).astype(np.int32)
    py = (cy - fx * scale * tilt).astype(np.int32)

    # Clipping mask
    mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[mask]
    py = py[mask]
    z  = z[mask]
    tags = tags[mask]

    # Z-based brightness modulation (higher = brighter)
    z_min = z.min() if len(z) > 0 else 0
    z_max = z.max() if len(z) > 0 else 1
    z_range = max(z_max - z_min, 1.0)
    brightness = 0.6 + 0.4 * (z - z_min) / z_range  # 0.6–1.0

    # Draw each point
    draw_circle = pygame.draw.circle
    color_cache = {}

    for i in range(len(px)):
        tag = int(tags[i])
        base_col = SEMANTIC_COLORS.get(tag, (100, 100, 100))
        b = float(brightness[i])

        # Cache bright-adjusted colours
        key = (tag, int(b * 10))
        if key not in color_cache:
            color_cache[key] = (
                min(255, int(base_col[0] * b)),
                min(255, int(base_col[1] * b)),
                min(255, int(base_col[2] * b)),
            )
        col = color_cache[key]
        size = _POINT_SIZE.get(tag, _DEFAULT_POINT_SIZE)
        draw_circle(surface, col, (int(px[i]), int(py[i])), size)


# =============================================================================
# CARLASimulationManager — public API
# =============================================================================

class CARLASimulationManager:
    """
    Top-level simulation controller for semantic LiDAR point cloud.

    Drop-in replacement API:
        sim = CARLASimulationManager()
        surf = sim.update(driver_state, drowsiness_score, distraction_score)
        sim.reset()
        sim.handle_click(x, y)
    """

    def __init__(self, width: int = _PANEL_W, height: int = _PANEL_H):
        self.w = width
        self.h = height

        self._surface = pygame.Surface((width, height))
        self._hud = CarlaHUD(width, height)
        self._client = _CarlaClient()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        driver_state:      str   = "ALERT",
        drowsiness_score:  float = 0.0,
        distraction_score: float = 0.0,
        **kwargs,
    ) -> pygame.Surface:
        """
        Advance simulation one frame.

        Returns:
            pygame.Surface (width × height) with the LiDAR point cloud + HUD.
        """
        client = self._client

        # Push DMS state
        client.set_driver_state(driver_state, drowsiness_score)

        # Tick the world (main thread only)
        client.tick()

        # Get latest point cloud
        points = client.get_points()

        if points is not None and client.connected:
            # Clear surface
            self._surface.fill(CARLA_OFFLINE_BG)

            # Render point cloud
            _render_points(self._surface, points, CARLA_LIDAR_RANGE)

            # HUD overlay
            self._hud.render(
                self._surface,
                driver_state=driver_state,
                drowsiness_score=drowsiness_score,
                speed_kmh=client.speed_kmh,
                connected=True,
                is_pulling_over=client.is_pulling_over,
                is_stopped=client.is_stopped,
            )
        else:
            # Offline placeholder
            self._hud.render_placeholder(self._surface)

        return self._surface

    def reset(self) -> None:
        """Reset simulation to initial driving state."""
        self._client.reset()

    def handle_click(self, x: int, y: int) -> bool:
        """Handle a mouse click. Returns True if restart was clicked."""
        if self._hud.handle_click(x, y):
            self.reset()
            return True
        return False

    def __del__(self):
        try:
            self._client.destroy()
        except Exception:
            pass


# Alias so __init__.py and direct imports both work
SimulationManager = CARLASimulationManager
