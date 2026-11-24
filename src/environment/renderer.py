"""
Renderer for Snake Environment.
Provides headless and visual rendering using Pillow and Matplotlib.
Supports exporting frames (PNG) and episode gameplay animations (GIF).
"""

from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.environment.game import Direction, Point, SnakeGame


class SnakeRenderer:
    """
    Renders SnakeGame state onto high-quality RGB images.
    Designed with a sleek modern dark theme.
    """

    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        cell_size: int = 24,
        margin_top: int = 48,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.margin_top = margin_top

        self.width = grid_width * cell_size
        self.height = grid_height * cell_size + margin_top

        # Color Palette (Modern Dark Theme)
        self.COLOR_BG = (18, 20, 29)          # Deep dark blue/charcoal
        self.COLOR_GRID_BG = (24, 27, 38)     # Game board background
        self.COLOR_GRID_LINE = (35, 40, 56)   # Subtle grid lines
        self.COLOR_BORDER = (99, 102, 241)    # Indigo border
        self.COLOR_HEAD = (16, 185, 129)      # Emerald green
        self.COLOR_BODY = (5, 150, 105)       # Darker emerald green
        self.COLOR_BODY_ALT = (4, 120, 87)    # Gradient alternating green
        self.COLOR_EYE_WHITE = (255, 255, 255)
        self.COLOR_EYE_PUPIL = (15, 23, 42)
        self.COLOR_FOOD = (239, 68, 68)       # Bright coral/crimson red
        self.COLOR_FOOD_GLOW = (248, 113, 113, 100)
        self.COLOR_TEXT = (241, 245, 249)     # Bright text
        self.COLOR_TEXT_MUTED = (148, 163, 184)

    def render_frame(
        self,
        game: SnakeGame,
        episode: Optional[int] = None,
        q_values: Optional[np.ndarray] = None,
        action_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Renders the current game state into a PIL Image.
        """
        img = Image.new("RGB", (self.width, self.height), self.COLOR_BG)
        draw = ImageDraw.Draw(img, "RGBA")

        # Top Bar: Info and Score
        score_text = f"SCORE: {game.score}"
        steps_text = f"STEPS: {game.steps}"
        ep_text = f"EP: {episode}" if episode is not None else "LIVE"

        # Draw header text
        draw.text((12, 12), ep_text, fill=self.COLOR_TEXT, font=None)
        draw.text((self.width // 2 - 30, 12), score_text, fill=(52, 211, 153), font=None)
        draw.text((self.width - 90, 12), steps_text, fill=self.COLOR_TEXT_MUTED, font=None)

        # Draw board background
        board_rect = [0, self.margin_top, self.width, self.height]
        draw.rectangle(board_rect, fill=self.COLOR_GRID_BG)

        # Draw grid lines
        for x in range(self.grid_width + 1):
            px = x * self.cell_size
            draw.line([(px, self.margin_top), (px, self.height)], fill=self.COLOR_GRID_LINE, width=1)
        for y in range(self.grid_height + 1):
            py = self.margin_top + y * self.cell_size
            draw.line([(0, py), (self.width, py)], fill=self.COLOR_GRID_LINE, width=1)

        # Draw outer boundary
        draw.rectangle(board_rect, outline=self.COLOR_BORDER, width=2)

        # Draw Food
        if game.food.x >= 0 and game.food.y >= 0:
            fx = game.food.x * self.cell_size
            fy = self.margin_top + game.food.y * self.cell_size
            padding = 3
            # Outer glow
            draw.ellipse(
                [fx + padding - 2, fy + padding - 2, fx + self.cell_size - padding + 2, fy + self.cell_size - padding + 2],
                fill=self.COLOR_FOOD_GLOW,
            )
            # Food body
            draw.ellipse(
                [fx + padding, fy + padding, fx + self.cell_size - padding, fy + self.cell_size - padding],
                fill=self.COLOR_FOOD,
            )
            # Food highlight
            draw.ellipse(
                [fx + padding + 3, fy + padding + 2, fx + padding + 6, fy + padding + 5],
                fill=(255, 255, 255, 200),
            )

        # Draw Snake Body (from tail to neck)
        for i, pt in enumerate(reversed(game.snake[1:])):
            bx = pt.x * self.cell_size
            by = self.margin_top + pt.y * self.cell_size
            pad = 2
            color = self.COLOR_BODY if i % 2 == 0 else self.COLOR_BODY_ALT
            draw.rounded_rectangle(
                [bx + pad, by + pad, bx + self.cell_size - pad, by + self.cell_size - pad],
                radius=4,
                fill=color,
            )

        # Draw Snake Head
        if game.snake:
            head = game.snake[0]
            hx = head.x * self.cell_size
            hy = self.margin_top + head.y * self.cell_size
            pad = 1
            draw.rounded_rectangle(
                [hx + pad, hy + pad, hx + self.cell_size - pad, hy + self.cell_size - pad],
                radius=6,
                fill=self.COLOR_HEAD,
            )

            # Draw Eyes based on direction
            eye_radius = 2.5
            pupil_radius = 1.2
            if game.direction == Direction.RIGHT:
                e1 = (hx + self.cell_size - 6, hy + 6)
                e2 = (hx + self.cell_size - 6, hy + self.cell_size - 6)
                p1 = (e1[0] + 1, e1[1])
                p2 = (e2[0] + 1, e2[1])
            elif game.direction == Direction.LEFT:
                e1 = (hx + 6, hy + 6)
                e2 = (hx + 6, hy + self.cell_size - 6)
                p1 = (e1[0] - 1, e1[1])
                p2 = (e2[0] - 1, e2[1])
            elif game.direction == Direction.UP:
                e1 = (hx + 6, hy + 6)
                e2 = (hx + self.cell_size - 6, hy + 6)
                p1 = (e1[0], e1[1] - 1)
                p2 = (e2[0], e2[1] - 1)
            else:  # DOWN
                e1 = (hx + 6, hy + self.cell_size - 6)
                e2 = (hx + self.cell_size - 6, hy + self.cell_size - 6)
                p1 = (e1[0], e1[1] + 1)
                p2 = (e2[0], e2[1] + 1)

            for eye in [e1, e2]:
                draw.ellipse(
                    [eye[0] - eye_radius, eye[1] - eye_radius, eye[0] + eye_radius, eye[1] + eye_radius],
                    fill=self.COLOR_EYE_WHITE,
                )
            for pupil in [p1, p2]:
                draw.ellipse(
                    [pupil[0] - pupil_radius, pupil[1] - pupil_radius, pupil[0] + pupil_radius, pupil[1] + pupil_radius],
                    fill=self.COLOR_EYE_PUPIL,
                )

        # Optional Q-values badge
        if q_values is not None:
            q_str = f"Q:[S:{q_values[0]:.2f} R:{q_values[1]:.2f} L:{q_values[2]:.2f}]"
            draw.text((12, self.margin_top + 6), q_str, fill=(244, 114, 182, 220), font=None)

        if action_name is not None:
            draw.text((self.width - 120, self.margin_top + 6), f"Act: {action_name}", fill=(56, 189, 248), font=None)

        return img

    def save_frame(self, game: SnakeGame, filepath: str, **kwargs) -> None:
        """Saves a single frame to an image file."""
        frame = self.render_frame(game, **kwargs)
        frame.save(filepath)

    def save_gif(
        self,
        frames: List[Image.Image],
        filepath: str,
        duration_ms: int = 70,
        loop: int = 0,
    ) -> None:
        """Saves a sequence of frames as an animated GIF."""
        if not frames:
            return
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=loop,
            optimize=True,
        )
