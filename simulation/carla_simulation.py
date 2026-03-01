# =============================================================================
# simulation/carla_simulation.py
#
# CARLA integration for the DMS simulation panel.
#
# Renders a live semantic segmentation camera feed from a CARLA server
# into a pygame.Surface (WINDOW_WIDTH × WINDOW_HEIGHT//2).
#
# Architecture:
#   CarlaClient     — background connection + actor management
#   SimulationManager — public API (drop-in replacement for old sim)
#
# DMS State → CARLA Behaviour:
#   ALERT    → autopilot on, normal speed
#   DROWSY   → autopilot on, reduced speed, right blinker
#   SLEEPING → disengage autopilot, steer right + brake, then hazards
#
# Graceful fallback:
#   If the `carla` package is not installed or the server is unreachable,
#   a styled placeholder surface is returned every frame while a background
#   thread keeps retrying the connection.
# =============================================================================

import threading
import time
import numpy as np
import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    CARLA_HOST, CARLA_PORT, CARLA_MAP, CARLA_TIMEOUT,
    CARLA_RECONNECT_INTERVAL,
    CARLA_CAMERA_WIDTH, CARLA_CAMERA_HEIGHT, CARLA_CAMERA_FOV,
    CARLA_TARGET_SPEED_KMH, CARLA_DROWSY_SPEED_FACTOR,
)
from simulation.carla_hud import CarlaHUD

# ── Try importing carla ──────────────────────────────────────────────────────
try:
    import carla
    _CARLA_AVAILABLE = True
except ImportError:
    _CARLA_AVAILABLE = False

_PANEL_W = WINDOW_WIDTH
_PANEL_H = WINDOW_HEIGHT // 2


# =============================================================================
# CarlaClient — background CARLA connection and actor management
# =============================================================================

class CarlaClient:
    """
    Manages a background connection to a CARLA server, spawns an ego vehicle
    with a semantic segmentation camera, and converts incoming frames to
    numpy arrays for pygame rendering.

    All public methods are thread-safe.
    """

    def __init__(self):
        # Connection state
        self._connected = False
        self._shutting_down = False
        self._lock = threading.Lock()

        # CARLA handles
        self._client = None
        self._world = None
        self._vehicle = None
        self._camera = None
        self._traffic_manager = None

        # Latest frame from the semantic segmentation camera
        self._frame: np.ndarray | None = None

        # Vehicle speed in km/h (read from CARLA)
        self._speed_kmh = 0.0

        # Current DMS state tracking for pull-over logic
        self._driver_state = "ALERT"
        self._is_pulling_over = False
        self._is_stopped = False

        # Start background connection thread
        if _CARLA_AVAILABLE:
            t = threading.Thread(target=self._connect_loop, daemon=True)
            t.start()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def speed_kmh(self) -> float:
        return self._speed_kmh

    def get_frame(self) -> np.ndarray | None:
        """Return the latest semantic segmentation frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def set_driver_state(
        self, state: str, drowsiness_score: float
    ) -> None:
        """Map DMS driver state to CARLA vehicle behaviour."""
        if not self._connected or self._vehicle is None:
            return

        try:
            self._driver_state = state

            if state == "ALERT":
                self._handle_alert(drowsiness_score)
            elif state == "DROWSY":
                self._handle_drowsy(drowsiness_score)
            elif state == "SLEEPING":
                self._handle_sleeping()
        except Exception:
            pass  # never crash the DMS

    def reset(self) -> None:
        """Teleport vehicle to a spawn point and re-enable autopilot."""
        if not self._connected or self._vehicle is None:
            return
        try:
            self._is_pulling_over = False
            self._is_stopped = False

            # Teleport to a random spawn point
            spawn_points = self._world.get_map().get_spawn_points()
            if spawn_points:
                import random
                transform = random.choice(spawn_points)
                self._vehicle.set_transform(transform)

            # Reset vehicle physics
            self._vehicle.apply_control(
                carla.VehicleControl(throttle=0, brake=1.0, steer=0)
            )
            time.sleep(0.1)

            # Re-enable autopilot
            self._vehicle.set_autopilot(True, self._traffic_manager.get_port())
            self._traffic_manager.vehicle_percentage_speed_difference(
                self._vehicle, 0.0
            )

            # Clear lights
            self._vehicle.set_light_state(carla.VehicleLightState.NONE)
        except Exception:
            pass

    def destroy(self) -> None:
        """Clean up all CARLA actors."""
        self._shutting_down = True
        self._connected = False
        try:
            if self._camera is not None:
                self._camera.stop()
                self._camera.destroy()
            if self._vehicle is not None:
                self._vehicle.destroy()
        except Exception:
            pass
        self._camera = None
        self._vehicle = None
        self._client = None
        self._world = None

    # ── Background Connection ────────────────────────────────────────────────

    def _connect_loop(self) -> None:
        """Background thread: keep trying to connect to CARLA."""
        while not self._shutting_down:
            if not self._connected:
                try:
                    self._setup_world()
                    self._connected = True
                except Exception:
                    self._connected = False
                    time.sleep(CARLA_RECONNECT_INTERVAL)
            else:
                # Periodically read vehicle speed
                try:
                    if self._vehicle is not None:
                        vel = self._vehicle.get_velocity()
                        self._speed_kmh = 3.6 * (
                            vel.x ** 2 + vel.y ** 2 + vel.z ** 2
                        ) ** 0.5
                except Exception:
                    self._connected = False
                time.sleep(0.05)  # ~20 Hz speed polling

    def _setup_world(self) -> None:
        """Connect, load map, spawn ego + camera."""
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)

        # Load the target map if not already loaded
        world = client.get_world()
        current_map = world.get_map().name
        if CARLA_MAP not in current_map:
            world = client.load_world(CARLA_MAP)

        # Settings: synchronous mode off (we stream asynchronously)
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # Traffic Manager
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(False)
        tm.set_global_distance_to_leading_vehicle(2.5)

        # Find ego vehicle blueprint
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        # Spawn at a random spawn point
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")

        import random
        vehicle = None
        for sp in random.sample(spawn_points, min(10, len(spawn_points))):
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Could not spawn ego vehicle")

        # Enable autopilot
        vehicle.set_autopilot(True, tm.get_port())

        # ── Semantic Segmentation Camera ─────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(CARLA_CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CARLA_CAMERA_HEIGHT))
        cam_bp.set_attribute("fov",          str(CARLA_CAMERA_FOV))

        # Mount on the windshield area
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5.0),
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        camera.listen(self._on_image)

        # Store handles
        self._client = client
        self._world = world
        self._vehicle = vehicle
        self._camera = camera
        self._traffic_manager = tm

    def _on_image(self, image) -> None:
        """Callback: convert CARLA semantic seg image to numpy BGRA array."""
        # CityScapes palette conversion
        image.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))  # BGRA
        # Convert BGRA → RGB
        rgb = arr[:, :, :3][:, :, ::-1].copy()
        with self._lock:
            self._frame = rgb

    # ── Driver State Handlers ────────────────────────────────────────────────

    def _handle_alert(self, drowsiness_score: float) -> None:
        """ALERT: autopilot on, normal speed, clear signals."""
        if self._is_pulling_over or self._is_stopped:
            # Resume after pull-over
            self._is_pulling_over = False
            self._is_stopped = False
            self._vehicle.set_autopilot(
                True, self._traffic_manager.get_port()
            )

        # Speed: linear reduction based on drowsiness_score
        pct_diff = drowsiness_score * 30.0  # 0–30% slower
        self._traffic_manager.vehicle_percentage_speed_difference(
            self._vehicle, pct_diff
        )

        # Clear lights
        self._vehicle.set_light_state(carla.VehicleLightState.NONE)

    def _handle_drowsy(self, drowsiness_score: float) -> None:
        """DROWSY: autopilot on, reduced speed, right blinker."""
        if self._is_pulling_over:
            return  # don't interrupt an active pull-over

        if self._is_stopped:
            self._is_stopped = False
            self._vehicle.set_autopilot(
                True, self._traffic_manager.get_port()
            )

        # Reduce speed
        base_factor = (1.0 - CARLA_DROWSY_SPEED_FACTOR) * 100.0
        extra = drowsiness_score * 20.0
        self._traffic_manager.vehicle_percentage_speed_difference(
            self._vehicle, base_factor + extra
        )

        # Right blinker
        self._vehicle.set_light_state(
            carla.VehicleLightState.RightBlinker
        )

    def _handle_sleeping(self) -> None:
        """SLEEPING: disengage autopilot, emergency pull-over."""
        if self._is_stopped:
            return  # already stopped

        if not self._is_pulling_over:
            # Disengage autopilot
            self._vehicle.set_autopilot(False)
            self._is_pulling_over = True

        # Check current speed
        vel = self._vehicle.get_velocity()
        speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

        if speed < 0.3:
            # Vehicle has stopped — activate hazards
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
            # Steer right + brake
            self._vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.3,
                    brake=0.8,
                    hand_brake=False,
                )
            )
            self._vehicle.set_light_state(
                carla.VehicleLightState.RightBlinker
            )


# =============================================================================
# SimulationManager — public API (drop-in replacement)
# =============================================================================

class SimulationManager:
    """
    Top-level simulation controller wrapping the CARLA client.

    Drop-in replacement for the old Pygame-based simulation.
    API contract:
        sim = SimulationManager()
        surf = sim.update(driver_state, drowsiness_score, distraction_score)
        sim.reset()
        sim.handle_click(x, y)
    """

    def __init__(self, width: int = _PANEL_W, height: int = _PANEL_H):
        self.w = width
        self.h = height

        self._surface = pygame.Surface((width, height))
        self._hud = CarlaHUD(width, height)
        self._client = CarlaClient()

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

        Args:
            driver_state:      "ALERT" | "DROWSY" | "SLEEPING"
            drowsiness_score:  0–1
            distraction_score: 0–1

        Returns:
            pygame.Surface (width × height) ready to blit
        """
        # Push DMS state to CARLA
        self._client.set_driver_state(driver_state, drowsiness_score)

        # Grab latest frame
        frame = self._client.get_frame()

        if frame is not None and self._client.connected:
            # Convert numpy RGB array → pygame Surface
            # frame shape: (H, W, 3) — RGB
            surf = pygame.surfarray.make_surface(
                frame.swapaxes(0, 1)
            )
            # Scale if camera resolution differs from panel size
            if surf.get_size() != (self.w, self.h):
                surf = pygame.transform.scale(surf, (self.w, self.h))
            self._surface.blit(surf, (0, 0))

            # Draw HUD overlay
            self._hud.render(
                self._surface,
                driver_state=driver_state,
                drowsiness_score=drowsiness_score,
                speed_kmh=self._client.speed_kmh,
                connected=True,
            )
        else:
            # Placeholder mode
            self._hud.render_placeholder(self._surface)

        return self._surface

    def reset(self) -> None:
        """Reset simulation to initial driving state."""
        self._client.reset()

    def handle_click(self, x: int, y: int) -> bool:
        """
        Handle a mouse click within the sim panel.

        Args:
            x, y: click position relative to the sim panel's top-left corner

        Returns:
            True if the restart button was clicked.
        """
        if self._hud.handle_click(x, y):
            self.reset()
            return True
        return False

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self._client.destroy()
        except Exception:
            pass
