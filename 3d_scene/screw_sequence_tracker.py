"""
Screw Sequence Tracker - Tracks the order of screws being tightened on the case.

This module monitors the distance from the e-screwdriver tool tip to each screw
position on the case and tracks:
1. Which screw is currently being worked on (based on proximity)
2. When a screw is tightened (tool stays near for a duration)
3. The sequence of screws tightened
4. Whether the sequence matches the expected order

Expected screw order: BL (bottom-left) → TR (top-right) → BR (bottom-right) → TL (top-left)

The tracking uses a state machine:
- IDLE: Tool is not near any screw
- APPROACHING: Tool is moving toward a screw (within approach distance)
- SCREWING: Tool is close to a screw (within screwing distance)
- COMPLETED: All 4 screws have been tightened

Usage:
    from screw_sequence_tracker import ScrewSequenceTracker

    tracker = ScrewSequenceTracker()

    # Update with distance data (called from backend when frontend sends data)
    tracker.update(distance_cm=2.5, nearest_screw="bottom_left")

    # Get current status
    status = tracker.get_status()
"""

import time
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class TrackerState(Enum):
    """State of the screw tracker."""

    IDLE = "idle"
    APPROACHING = "approaching"
    SCREWING = "screwing"
    COMPLETED = "completed"


@dataclass
class ScrewState:
    """State of a single screw position."""

    position: str  # top_left, top_right, bottom_left, bottom_right
    is_tightened: bool = False
    tighten_order: int = 0  # 1-4, 0 if not tightened
    tighten_time: float = 0.0  # timestamp when tightened
    total_time_near: float = 0.0  # total time tool was near this screw


@dataclass
class ScrewSequenceState:
    """Complete state of the screw sequence tracking."""

    # Screw states
    screws: Dict[str, ScrewState] = field(default_factory=dict)

    # Sequence tracking
    expected_sequence: List[str] = field(
        default_factory=lambda: ["bottom_left", "top_right", "bottom_right", "top_left"]
    )
    actual_sequence: List[str] = field(default_factory=list)

    # Current state
    current_state: TrackerState = TrackerState.IDLE
    active_screw: Optional[str] = None  # Currently being worked on
    current_step: int = 0  # 0-3 for which screw should be next

    # Timing
    time_near_current_screw: float = 0.0  # Time near active screw
    last_update_time: float = 0.0

    # For 3D tracking smoothness
    frames_near_screw: int = 0  # How many consecutive frames near the screw

    # Errors
    errors: List[str] = field(default_factory=list)
    is_correct: bool = True  # True if sequence is correct so far
    completed: bool = False


class ScrewSequenceTracker:
    """
    Tracks the sequence of screws being tightened based on tool proximity.

    Uses distance measurements from the 3D scene to detect when the tool
    is near a screw position and when it has been there long enough to
    consider the screw tightened.
    """

    # Expected order of screws (diagonal pattern for stress distribution)
    EXPECTED_ORDER = ["bottom_left", "top_right", "bottom_right", "top_left"]

    # Position short names for display
    SHORT_NAMES = {
        "top_left": "TL",
        "top_right": "TR",
        "bottom_left": "BL",
        "bottom_right": "BR",
    }

    # Distance thresholds (in centimeters)
    APPROACH_DISTANCE = 10.0  # cm - tool is approaching a screw
    SCREWING_DISTANCE = 7.0  # cm - tool is actively screwing

    # Time thresholds
    TIME_TO_TIGHTEN = 1.5  # seconds - time tool must be near to count as tightened

    # Frame-based thresholds (for 3D tracking at ~30fps)
    FRAMES_TO_COMPLETE = 40  # ~1.3 seconds at 30fps

    def __init__(self):
        """Initialize the screw sequence tracker."""
        self.state = ScrewSequenceState()
        self.lock = threading.Lock()
        self.enabled = True
        self.tracking_mode = "3d"  # "3d" (uses frames) or "time" (uses duration)

        # Initialize screw states
        for pos in self.EXPECTED_ORDER:
            self.state.screws[pos] = ScrewState(position=pos)

        self.state.last_update_time = time.time()

        print(
            "[ScrewTracker] Initialized with expected order:",
            " → ".join([self.SHORT_NAMES[s] for s in self.EXPECTED_ORDER]),
        )

    def reset(self):
        """Reset the tracker to initial state."""
        with self.lock:
            self.state = ScrewSequenceState()
            for pos in self.EXPECTED_ORDER:
                self.state.screws[pos] = ScrewState(position=pos)
            self.state.last_update_time = time.time()

        print("[ScrewTracker] Reset - ready for new sequence")

    def set_enabled(self, enabled: bool):
        """Enable or disable tracking."""
        self.enabled = enabled
        if not enabled:
            with self.lock:
                self.state.current_state = TrackerState.IDLE
                self.state.active_screw = None
        print(f"[ScrewTracker] {'Enabled' if enabled else 'Disabled'}")

    def set_mode(self, mode: str):
        """Set tracking mode: '3d' (frame-based) or 'time' (duration-based)."""
        if mode in ["3d", "time"]:
            self.tracking_mode = mode
            print(f"[ScrewTracker] Mode set to: {mode}")

    def update(
        self, distance_cm: float, nearest_screw: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update tracker with new distance measurement.

        Args:
            distance_cm: Distance from tool tip to nearest screw (in cm)
            nearest_screw: Name of nearest screw position (e.g., "bottom_left")

        Returns:
            Current status dictionary
        """
        if not self.enabled:
            return self.get_status()

        with self.lock:
            now = time.time()
            dt = now - self.state.last_update_time
            self.state.last_update_time = now

            # Don't process if already completed
            if self.state.completed:
                return self._get_status_unlocked()

            # Determine current state based on distance
            if distance_cm <= self.SCREWING_DISTANCE:
                new_state = TrackerState.SCREWING
            elif distance_cm <= self.APPROACH_DISTANCE:
                new_state = TrackerState.APPROACHING
            else:
                new_state = TrackerState.IDLE

            # Handle state transitions
            if new_state == TrackerState.SCREWING and nearest_screw:
                self._handle_screwing(nearest_screw, distance_cm, dt)
            elif new_state == TrackerState.APPROACHING and nearest_screw:
                self._handle_approaching(nearest_screw)
            else:
                self._handle_idle()

            self.state.current_state = new_state

            return self._get_status_unlocked()

    def _handle_screwing(self, screw: str, distance_cm: float, dt: float):
        """Handle state when tool is in screwing position."""
        screw_state = self.state.screws.get(screw)
        if not screw_state or screw_state.is_tightened:
            # Screw already tightened or invalid, just update active screw
            if self.state.active_screw != screw:
                self.state.active_screw = screw
                self.state.time_near_current_screw = 0.0
                self.state.frames_near_screw = 0
            return

        # Check if this is the same screw we were near before
        if self.state.active_screw == screw:
            # Accumulate time/frames near this screw
            self.state.time_near_current_screw += dt
            self.state.frames_near_screw += 1
            screw_state.total_time_near += dt

            # Check if screw should be considered tightened
            should_tighten = False
            if self.tracking_mode == "3d":
                should_tighten = self.state.frames_near_screw >= self.FRAMES_TO_COMPLETE
            else:
                should_tighten = (
                    self.state.time_near_current_screw >= self.TIME_TO_TIGHTEN
                )

            if should_tighten:
                self._tighten_screw(screw)
        else:
            # New screw - reset counters
            self.state.active_screw = screw
            self.state.time_near_current_screw = dt
            self.state.frames_near_screw = 1
            screw_state.total_time_near = dt

    def _handle_approaching(self, screw: str):
        """Handle state when tool is approaching a screw."""
        if self.state.active_screw != screw:
            # Reset counters when approaching a different screw
            self.state.time_near_current_screw = 0.0
            self.state.frames_near_screw = 0
        self.state.active_screw = screw

    def _handle_idle(self):
        """Handle idle state when tool is not near any screw."""
        # Keep active_screw for context but reset counters
        self.state.time_near_current_screw = 0.0
        self.state.frames_near_screw = 0

    def _tighten_screw(self, screw: str):
        """Mark a screw as tightened and validate sequence."""
        screw_state = self.state.screws[screw]
        if screw_state.is_tightened:
            return  # Already tightened

        # Mark as tightened
        screw_state.is_tightened = True
        screw_state.tighten_order = len(self.state.actual_sequence) + 1
        screw_state.tighten_time = time.time()

        # Add to actual sequence
        self.state.actual_sequence.append(screw)

        # Check if this was the expected screw
        expected = self.EXPECTED_ORDER[self.state.current_step]
        if screw != expected:
            self.state.is_correct = False
            error_msg = (
                f"Expected {self.SHORT_NAMES[expected]}, got {self.SHORT_NAMES[screw]}"
            )
            self.state.errors.append(error_msg)
            print(f"[ScrewTracker] ❌ Wrong order! {error_msg}")
        else:
            print(
                f"[ScrewTracker] ✓ Screw {self.SHORT_NAMES[screw]} tightened (step {self.state.current_step + 1}/4)"
            )

        # Move to next step
        self.state.current_step += 1

        # Reset tracking for next screw
        self.state.time_near_current_screw = 0.0
        self.state.frames_near_screw = 0

        # Check if sequence is complete
        if self.state.current_step >= len(self.EXPECTED_ORDER):
            self.state.completed = True
            self.state.current_state = TrackerState.COMPLETED

            if self.state.is_correct:
                print("[ScrewTracker] ✓ SEQUENCE COMPLETE - Correct order!")
            else:
                print(
                    f"[ScrewTracker] ✗ SEQUENCE COMPLETE - Wrong order! Errors: {self.state.errors}"
                )

    def get_status(self) -> Dict[str, Any]:
        """Get current tracking status as a dictionary."""
        with self.lock:
            return self._get_status_unlocked()

    def _get_status_unlocked(self) -> Dict[str, Any]:
        """Get status without acquiring lock (must be called with lock held)."""
        # Get next expected screw
        next_expected = None
        if self.state.current_step < len(self.EXPECTED_ORDER):
            next_expected = self.EXPECTED_ORDER[self.state.current_step]

        return {
            "enabled": self.enabled,
            "mode": self.tracking_mode,
            "current_state": self.state.current_state.value,
            "active_screw": self.state.active_screw,
            "current_step": self.state.current_step,
            "next_expected": next_expected,
            "expected_sequence": self.EXPECTED_ORDER.copy(),
            "actual_sequence": self.state.actual_sequence.copy(),
            "is_correct": self.state.is_correct,
            "completed": self.state.completed,
            "errors": self.state.errors.copy(),
            "time_near_current": self.state.time_near_current_screw,
            "frames_near_3d": self.state.frames_near_screw,
            "frames_to_complete_3d": self.FRAMES_TO_COMPLETE,
            "screws": {
                pos: {
                    "position": state.position,
                    "is_tightened": state.is_tightened,
                    "tighten_order": state.tighten_order,
                    "total_time_near": state.total_time_near,
                }
                for pos, state in self.state.screws.items()
            },
        }


# Global tracker instance
_tracker_instance: Optional[ScrewSequenceTracker] = None


def get_tracker() -> ScrewSequenceTracker:
    """Get the global tracker instance (creates one if needed)."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ScrewSequenceTracker()
    return _tracker_instance


def reset_tracker():
    """Reset the global tracker."""
    tracker = get_tracker()
    tracker.reset()


# Standalone testing
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("Screw Sequence Tracker - Simulation Test")
    print("=" * 60)

    tracker = ScrewSequenceTracker()

    # Simulate correct sequence
    print("\n--- Simulating CORRECT sequence (BL → TR → BR → TL) ---\n")

    screws = ["bottom_left", "top_right", "bottom_right", "top_left"]

    for screw in screws:
        print(f"\nApproaching {tracker.SHORT_NAMES[screw]}...")

        # Approach phase
        for _ in range(10):
            tracker.update(distance_cm=6.0, nearest_screw=screw)
            time.sleep(0.03)

        # Screwing phase (stay near until tightened)
        print(f"Screwing {tracker.SHORT_NAMES[screw]}...")
        for frame in range(50):  # More than FRAMES_TO_COMPLETE
            status = tracker.update(distance_cm=2.5, nearest_screw=screw)
            if status["screws"][screw]["is_tightened"]:
                print(f"  → Tightened at frame {frame}")
                break
            time.sleep(0.03)

        # Move away
        for _ in range(5):
            tracker.update(distance_cm=15.0, nearest_screw=None)
            time.sleep(0.03)

    # Final status
    print("\n" + "=" * 60)
    status = tracker.get_status()
    print(f"Completed: {status['completed']}")
    print(f"Correct: {status['is_correct']}")
    print(
        f"Actual sequence: {' → '.join([tracker.SHORT_NAMES[s] for s in status['actual_sequence']])}"
    )
    print(f"Errors: {status['errors']}")
    print("=" * 60)

    # Test incorrect sequence
    print("\n\n--- Simulating INCORRECT sequence (BL → BR → TR → TL) ---\n")
    tracker.reset()

    wrong_screws = ["bottom_left", "bottom_right", "top_right", "top_left"]

    for screw in wrong_screws:
        print(f"\nScrewing {tracker.SHORT_NAMES[screw]}...")
        for frame in range(50):
            status = tracker.update(distance_cm=2.5, nearest_screw=screw)
            if status["screws"][screw]["is_tightened"]:
                break
            time.sleep(0.03)

        tracker.update(distance_cm=15.0, nearest_screw=None)

    # Final status
    print("\n" + "=" * 60)
    status = tracker.get_status()
    print(f"Completed: {status['completed']}")
    print(f"Correct: {status['is_correct']}")
    print(
        f"Actual sequence: {' → '.join([tracker.SHORT_NAMES[s] for s in status['actual_sequence']])}"
    )
    print(f"Errors: {status['errors']}")
    print("=" * 60)
