"""
alerts/alert_system.py — DMS Audio & Visual Alert Manager
Implements a 3-level alert state machine with debounce logic and pygame audio.

Level 0: Normal — silent monitoring
Level 1: Warning (1–2 s drowsy) — soft beep + HUD yellow overlay
Level 2: Critical (3+ s drowsy)  — loud repeating alarm + HUD red overlay
"""

import os
import sys
import time
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Pygame is optional — we degrade gracefully if unavailable
try:
    import pygame
    import pygame.sndarray
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    print("[AlertSystem] pygame not installed — audio alerts disabled.")


class AlertSystem:
    """
    Manages escalating drowsiness alerts with debounce and cooldown.

    Usage:
        alert = AlertSystem()
        alert.start()               # initialises pygame mixer
        level = alert.update(drowsy_score=45, ear=0.22, alert_level=1)
        alert.stop()
    """

    def __init__(self):
        self._level: int = 0           # current alert level (0, 1, 2)
        self._last_alert_time: float = 0.0
        self._alert_active: bool = False
        self._audio_thread: threading.Thread | None = None
        self._stop_audio: threading.Event = threading.Event()
        self._mixer_ready: bool = False
        self._sounds: dict[int, pygame.mixer.Sound | None] = {1: None, 2: None}

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise pygame mixer and pre-load / generate alert sounds."""
        if not _PYGAME_AVAILABLE:
            return

        try:
            pygame.mixer.pre_init(
                frequency=44100, size=-16, channels=1, buffer=512
            )
            pygame.mixer.init()
            self._mixer_ready = True
            self._sounds[1] = self._load_or_generate_sound(
                config.ALERT_SOUND_L1,
                freq=config.ALERT_L1_FREQ,
                duration=config.ALERT_L1_DURATION,
            )
            self._sounds[2] = self._load_or_generate_sound(
                config.ALERT_SOUND_L2,
                freq=config.ALERT_L2_FREQ,
                duration=config.ALERT_L2_DURATION,
            )
            print("[AlertSystem] pygame mixer ready.")
        except Exception as exc:
            print(f"[AlertSystem] pygame init failed: {exc}. Audio disabled.")
            self._mixer_ready = False

    def stop(self) -> None:
        """Stop any playing audio and tear down pygame mixer."""
        self._stop_audio.set()
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)
        if _PYGAME_AVAILABLE and self._mixer_ready:
            pygame.mixer.stop()
            pygame.mixer.quit()

    # ──────────────────────────────────────────────────────────────────────────
    # Core state machine
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        drowsy_score: float,
        ear: float,
        alert_level: int,
    ) -> int:
        """
        Update alert state based on incoming metrics.
        Should be called once per frame.

        Args:
            drowsy_score:  Drowsiness score 0–100.
            ear:           Current Eye Aspect Ratio.
            alert_level:   Alert level computed by the main pipeline (0/1/2).

        Returns:
            Current alert level after update.
        """
        new_level = alert_level

        # Enforce minimum transition: can't jump straight to 0 without cooldown
        if new_level < self._level:
            # Only reset if things have been normal for at least 1.5 s
            if time.time() - self._last_alert_time > 1.5:
                self._level = new_level
                self._stop_audio.set()
        elif new_level > self._level:
            self._level = new_level
            self._last_alert_time = time.time()
            self._trigger_audio(new_level)

        return self._level

    # ──────────────────────────────────────────────────────────────────────────
    # Audio helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _trigger_audio(self, level: int) -> None:
        """Start an audio thread for the given alert level."""
        if not self._mixer_ready:
            return
        # Stop any prior audio thread
        self._stop_audio.set()
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=0.2)
        self._stop_audio.clear()

        interval = (
            config.ALERT_REPEAT_INTERVAL_L1 if level == 1
            else config.ALERT_REPEAT_INTERVAL_L2
        )
        sound = self._sounds.get(level)
        if sound is None:
            return

        def _play_loop():
            while not self._stop_audio.is_set():
                sound.play()
                # Wait for the interval or until stop is signalled
                self._stop_audio.wait(timeout=interval)

        self._audio_thread = threading.Thread(
            target=_play_loop, daemon=True, name=f"alert-audio-L{level}"
        )
        self._audio_thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Sound generation / loading
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_tone(
        freq: float = 880.0,
        duration: float = 0.4,
        sample_rate: int = 44100,
        volume: float = 0.6,
    ) -> np.ndarray:
        """
        Generate a simple sine-wave beep tone as a numpy array.

        Args:
            freq:        Frequency in Hz.
            duration:    Duration in seconds.
            sample_rate: Audio sample rate.
            volume:      Peak amplitude 0–1.

        Returns:
            Int16 mono numpy array suitable for pygame.sndarray.make_sound.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)

        # Apply fade-in and fade-out to avoid clicks
        fade_samples = int(sample_rate * 0.02)  # 20 ms
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

        return (wave * volume * 32767).astype(np.int16)

    def _load_or_generate_sound(
        self,
        path: str,
        freq: float,
        duration: float,
    ) -> "pygame.mixer.Sound | None":
        """
        Load a .wav file from disk, or synthesize a tone if not found.

        Args:
            path:     Path to the .wav file.
            freq:     Fallback tone frequency in Hz.
            duration: Fallback tone duration in seconds.

        Returns:
            pygame.mixer.Sound object or None if pygame is unavailable.
        """
        if not self._mixer_ready:
            return None

        if os.path.exists(path):
            try:
                snd = pygame.mixer.Sound(path)
                print(f"[AlertSystem] Loaded sound: {path}")
                return snd
            except Exception as exc:
                print(f"[AlertSystem] Could not load {path}: {exc}. Generating tone.")

        # Generate numpy tone and wrap in pygame Sound
        tone = self._generate_tone(freq=freq, duration=duration)
        # pygame.sndarray.make_sound expects (samples,) int16 for mono
        snd = pygame.sndarray.make_sound(tone)
        print(f"[AlertSystem] Generated {freq:.0f} Hz tone ({duration:.1f} s).")
        return snd

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def level(self) -> int:
        """Current alert level (0 / 1 / 2)."""
        return self._level

    # ──────────────────────────────────────────────────────────────────────────
    # Context manager
    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing AlertSystem …")
    with AlertSystem() as alert:
        print("Level 0 (silent):")
        lvl = alert.update(drowsy_score=5, ear=0.32, alert_level=0)
        print(f"  → level = {lvl}")
        time.sleep(0.3)

        print("Level 1 (warning beep):")
        lvl = alert.update(drowsy_score=35, ear=0.22, alert_level=1)
        print(f"  → level = {lvl}")
        time.sleep(2.5)

        print("Level 2 (critical alarm):")
        lvl = alert.update(drowsy_score=70, ear=0.12, alert_level=2)
        print(f"  → level = {lvl}")
        time.sleep(2.0)

        print("Reset to level 0:")
        lvl = alert.update(drowsy_score=5, ear=0.32, alert_level=0)
        time.sleep(2.0)
        print(f"  → level = {alert.level}")
    print("AlertSystem OK.")
