import gymnasium as gym
from typing import Optional
import numpy as np
import pygame
from gymnasium import spaces

from envs.base_env import BaseParkingEnv
from utils.utils import (
    STATE_WIDTH, STATE_HEIGHT, NUMBER_OF_ACTIONS, NO_OF_OBSTACLES, grid_to_pixels, map_orientation_to_numeric,
    CAR_WIDTH, CAR_HEIGHT, WHITE, CAR_SPEED
)

class ParkingFeature(gym.Env, BaseParkingEnv):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode: Optional[str] = None):
        gym.Env.__init__(self)
        BaseParkingEnv.__init__(self)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.car_images = None
        self.obstacle_image = None

        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        low_values = np.array([0, 0, -STATE_WIDTH, -STATE_HEIGHT, 0] + [0] * 8)
        high_values = np.array([STATE_WIDTH, STATE_HEIGHT, STATE_WIDTH, STATE_HEIGHT, 3] + [STATE_WIDTH] * 8)
        self.observation_space = spaces.Box(low=low_values, high=high_values, dtype=np.int32)

        # Environment state
        self.car_rect = None
        self.car_orientation = ""
        self.parking_rect = None
        self.obstacle_rects = []
        self.obstacle_positions = np.array([])
        self.is_visited = set()
        self.reward = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.car_rect = self.spawn_car()
        self.car_orientation = self.get_orientation()
        self.parking_rect = self.spawn_parking(self.car_rect)
        self.obstacle_rects = self.spawn_obstacles(self.car_rect, self.parking_rect, n=NO_OF_OBSTACLES)
        self.current = (self.car_rect.x, self.car_rect.y)
        self.lot = (self.parking_rect.x, self.parking_rect.y)
        self.orientation = map_orientation_to_numeric(self.car_orientation)
        self.delta = (self.car_rect.x - self.parking_rect.x, self.car_rect.y - self.parking_rect.y)
        self.obstacle_positions = np.array([rect.x for rect in self.obstacle_rects] + [rect.y for rect in self.obstacle_rects])
        if self.render_mode == "human":
            self.render()
        return np.array([self.current[0], self.current[1], self.delta[0], self.delta[1], self.orientation] + self.obstacle_positions.tolist()), {}

    def step(self, action):
        if action == 3:  # straight
            if self.car_orientation == "up":
                self.car_rect.y -= CAR_SPEED
            elif self.car_orientation == "down":
                self.car_rect.y += CAR_SPEED
            elif self.car_orientation == "left":
                self.car_rect.x -= CAR_SPEED
            elif self.car_orientation == "right":
                self.car_rect.x += CAR_SPEED
        elif action == 4:  # backwards
            if self.car_orientation == "up":
                self.car_rect.y += CAR_SPEED
            elif self.car_orientation == "down":
                self.car_rect.y -= CAR_SPEED
            elif self.car_orientation == "left":
                self.car_rect.x += CAR_SPEED
            elif self.car_orientation == "right":
                self.car_rect.x -= CAR_SPEED
        elif action == 1:  # turn left
            if self.car_orientation == "up":
                self.car_orientation = "left"
            elif self.car_orientation == "down":
                self.car_orientation = "right"
            elif self.car_orientation == "left":
                self.car_orientation = "down"
            elif self.car_orientation == "right":
                self.car_orientation = "up"
        elif action == 2:  # turn right
            if self.car_orientation == "up":
                self.car_orientation = "right"
            elif self.car_orientation == "down":
                self.car_orientation = "left"
            elif self.car_orientation == "left":
                self.car_orientation = "up"
            elif self.car_orientation == "right":
                self.car_orientation = "down"

        # Clip to board
        self.car_rect.left = max(0, self.car_rect.left)
        self.car_rect.right = min(STATE_WIDTH, self.car_rect.right)
        self.car_rect.top = max(0, self.car_rect.top)
        self.car_rect.bottom = min(STATE_HEIGHT, self.car_rect.bottom)

        self.current = (self.car_rect.x, self.car_rect.y)
        terminated = False
        if self.parking_rect.colliderect(self.car_rect):
            self.reward = 2000
            terminated = True
        else:
            for obstacle_rect in self.obstacle_rects:
                if self.car_rect.colliderect(obstacle_rect):
                    self.reward = -1000
                    terminated = True
                    break
            else:
                self.reward = -1

        if self.render_mode == "human":
            self.render()

        self.orientation = map_orientation_to_numeric(self.car_orientation)
        self.delta = (self.car_rect.x - self.parking_rect.x, self.car_rect.y - self.parking_rect.y)
        self.obstacle_positions = np.array([rect.x for rect in self.obstacle_rects] + [rect.y for rect in self.obstacle_rects])
        obs = np.array([self.current[0], self.current[1], self.delta[0], self.delta[1], self.orientation] + self.obstacle_positions.tolist())
        return obs, self.reward, terminated, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Car Parking Game")
            self.window = pygame.display.set_mode((STATE_WIDTH, STATE_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.car_images is None:
            self.car_images = [
                pygame.image.load("assets/car-up.png"),
                pygame.image.load("assets/car-down.png"),
                pygame.image.load("assets/car-left.png"),
                pygame.image.load("assets/car-right.png")
            ]
        if self.obstacle_image is None:
            self.obstacle_image = pygame.image.load("assets/obstacle.png")
        self.window.fill(WHITE)
        car_sprite = None
        if self.car_orientation == "up":
            car_sprite = self.car_images[0]
        elif self.car_orientation == "down":
            car_sprite = self.car_images[1]
        elif self.car_orientation == "left":
            car_sprite = self.car_images[2]
        elif self.car_orientation == "right":
            car_sprite = self.car_images[3]
        self.window.blit(car_sprite, self.car_rect)
        for obstacle_rect in self.obstacle_rects:
            self.window.blit(self.obstacle_image, obstacle_rect)
        pygame.draw.rect(self.window, (0, 255, 0), self.parking_rect)
        pygame.event.get()
        pygame.display.flip()
        self.clock.tick(200)