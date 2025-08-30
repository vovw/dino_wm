import numpy as np
import cv2
from typing import Dict, Tuple


class PointMazeEnv:
    """
    Simple 2D point maze environment for minimal setup.
    Compatible with existing PointMaze data format and action ranges.
    Point agent in [-1,1]^2, continuous actions a ∈ [-1,1]^2.
    One vertical wall at x=0 with a central gap; agent collides with wall.
    """

    def __init__(self, render_size: int = 64, use_4d_state: bool = False):
        self.render_size = render_size
        self.use_4d_state = use_4d_state  # Match existing env's 4D state (pos + vel)

        if use_4d_state:
            self.state = np.array([0.0, 0.0, 0.0, 0.0])  # (x, y, vx, vy)
        else:
            self.state = np.array([0.0, 0.0])  # (x, y)

        self.action_space = np.array([-1.0, 1.0])  # action bounds (same as MuJoCo)
        self.observation_space = np.array([-1.0, 1.0])  # position bounds
        self.max_speed = 0.1  # maximum movement per step

        # Wall definition: vertical wall at x=0 with gap from -0.3 to 0.3
        self.wall_x = 0.0
        self.wall_gap_min = -0.3
        self.wall_gap_max = 0.3

    def reset(self, seed: int = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset environment to random valid state."""
        if seed is not None:
            np.random.seed(seed)

        # Sample random valid position (avoiding wall)
        valid = False
        while not valid:
            x = np.random.uniform(-0.9, 0.9)
            y = np.random.uniform(-0.9, 0.9)
            valid = self._is_valid_position(x, y)

        self.state = np.array([x, y])
        obs = self._get_obs()
        return obs, self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Take a step in the environment."""
        # Clip action to bounds
        action = np.clip(action, -1.0, 1.0)

        # Scale action to movement
        dx = action[0] * self.max_speed
        dy = action[1] * self.max_speed

        # Update position
        new_x = np.clip(self.state[0] + dx, -1.0, 1.0)
        new_y = np.clip(self.state[1] + dy, -1.0, 1.0)

        # Check wall collision
        if not self._is_valid_position(new_x, new_y):
            # If collision, stay in place
            new_x, new_y = self.state[0], self.state[1]

        if self.use_4d_state:
            # Update velocity (simple physics)
            vx = (new_x - self.state[0]) / self.max_speed if dx != 0 else 0
            vy = (new_y - self.state[1]) / self.max_speed if dy != 0 else 0
            self.state = np.array([new_x, new_y, vx, vy])
        else:
            self.state = np.array([new_x, new_y])

        # Dummy reward and done
        reward = 0.0
        done = False

        obs = self._get_obs()
        info = {"state": self.state.copy()}
        return obs, reward, done, info

    def _is_valid_position(self, x: float, y: float) -> bool:
        """Check if position is valid (not inside wall)."""
        # Wall at x=0 with gap from -0.3 to 0.3
        if abs(x - self.wall_x) < 0.05:  # wall thickness
            if not (self.wall_gap_min <= y <= self.wall_gap_max):
                return False
        return True

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        image = self.render_rgb()
        return {
            "image": image,
            "state": self.state.copy()
        }

    def render_rgb(self) -> np.ndarray:
        """Render 64x64 RGB top-down image with green dot for agent."""
        # Create blank image
        img = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        # Draw wall (vertical line at center with gap)
        wall_width = 3
        gap_start = int((self.wall_gap_min + 1.0) / 2.0 * self.render_size)
        gap_end = int((self.wall_gap_max + 1.0) / 2.0 * self.render_size)

        # Draw wall segments above and below gap
        cv2.rectangle(img, (self.render_size//2 - wall_width//2, 0),
                      (self.render_size//2 + wall_width//2, gap_start), (128, 128, 128), -1)
        cv2.rectangle(img, (self.render_size//2 - wall_width//2, gap_end),
                      (self.render_size//2 + wall_width//2, self.render_size), (128, 128, 128), -1)

        # Draw agent as green circle
        center_x = int((self.state[0] + 1.0) / 2.0 * self.render_size)
        center_y = int((self.state[1] + 1.0) / 2.0 * self.render_size)
        cv2.circle(img, (center_x, center_y), 4, (0, 255, 0), -1)

        return img

    def set_state(self, state: np.ndarray):
        """Set environment state directly."""
        self.state = np.array(state).copy()

    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
