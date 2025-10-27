import copy
import random
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from envs.base_env import BaseParkingEnv
from utils.utils import (
    STATE_WIDTH, STATE_HEIGHT, TILE_SIZE,
    CAR_WIDTH, CAR_HEIGHT, NUMBER_OF_ACTIONS, NO_OF_OBSTACLES,
    grid_to_pixels, GRAY, YELLOW, CAR_SPEED
)

NO_OF_AGENTS = 4
NO_OF_LOTS = 4
MAX_EPISODE_LENGTH = 100

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    environment = raw_env(render_mode=internal_render_mode)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment

def raw_env(render_mode=None):
    environment = ParkingMultiEnv(render_mode=render_mode)
    environment = parallel_to_aec(environment)
    return environment

class ParkingMultiEnv(ParallelEnv, BaseParkingEnv):
    metadata = {"render_modes": ["human"], "name": "parking_multi_v0"}

    def __init__(self, render_mode=None):
        ParallelEnv.__init__(self)
        BaseParkingEnv.__init__(self)
        self.agents = [f"player_{r}" for r in range(NO_OF_AGENTS)]
        self.agent_rects = {}
        self.agent_orientations = {}
        self.parking_rects = []
        self.obstacle_rects = []
        self.possible_agents = self.agents[:]
        self.successfully_parked = []
        self.render_mode = render_mode
        self.window = None
        self.off_screen_surface = None
        self.image = None
        self.clock = None
        self.isopen = True
        self.time_step = 0
        self.prng = None
        self.car_images = None
        self.obstacle_image = None

    def get_random_orientation(self, no_of_agents):
        orientations = []
        while len(orientations) < no_of_agents:
            orientations.append(self.prng.choice(["up", "down", "left", "right"]))
        return orientations

    def get_random_positions(self, num_rectangles, occupied_rects):
        generated_rects = []
        while len(generated_rects) < num_rectangles:
            new_rect = self.get_random_position(occupied_rects + generated_rects)
            generated_rects.append(new_rect)
        return generated_rects

    def get_random_position(self, occupied_rects):
        while True:
            tile = (
                self.prng.randint(0, STATE_WIDTH // TILE_SIZE - 1),
                self.prng.randint(0, STATE_HEIGHT // TILE_SIZE - 1)
            )
            rect = pygame.Rect(grid_to_pixels(*tile), (TILE_SIZE, TILE_SIZE))
            overlap = any(rect.colliderect(existing) for existing in occupied_rects)
            if not overlap:
                return rect

    def action_space(self, agent):
        return spaces.Discrete(NUMBER_OF_ACTIONS)

    def observation_space(self, agent):
        return spaces.Box(low=0, high=255, shape=(STATE_WIDTH, STATE_HEIGHT, 3), dtype=np.uint8)

    def state(self):
        return self.image

    def fill_surface(self):
        if self.car_images is None:
            self.car_images = [
                pygame.image.load("assets/car-up.png"),
                pygame.image.load("assets/car-down.png"),
                pygame.image.load("assets/car-left.png"),
                pygame.image.load("assets/car-right.png")
            ]
        if self.obstacle_image is None:
            self.obstacle_image = pygame.image.load("assets/obstacle.png")
        pygame.init()
        pygame.display.init()
        self.off_screen_surface = pygame.Surface((STATE_WIDTH, STATE_HEIGHT))
        self.off_screen_surface.fill(GRAY)
        for rect in self.parking_rects:
            pygame.draw.rect(self.off_screen_surface, YELLOW, rect)
        for rect in self.obstacle_rects:
            self.off_screen_surface.blit(self.obstacle_image, rect)
        # render agents
        for agent in self.agents:
            orientation = self.agent_orientations[agent]
            idx = ["up", "down", "left", "right"].index(orientation)
            car_sprite = self.car_images[idx]
            self.off_screen_surface.blit(car_sprite, self.agent_rects[agent])
        for parked_rect, orientation in self.successfully_parked:
            idx = ["up", "down", "left", "right"].index(orientation)
            car_sprite = self.car_images[idx]
            self.off_screen_surface.blit(car_sprite, parked_rect)
        self.image = pygame.surfarray.array3d(self.off_screen_surface)
        pygame.event.get()

    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode((STATE_WIDTH, STATE_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.window.blit(self.off_screen_surface, (0, 0))
        pygame.time.delay(150)
        pygame.display.flip()
        self.clock.tick(200)

    def reset(self, seed=None, options=None):
        self.prng = random.Random()
        if seed is not None:
            self.prng.seed(seed)
        else:
            self.prng.seed()
        self.successfully_parked = []
        self.time_step = 0
        self.agents = copy.copy(self.possible_agents)
        self.agent_orientations = dict(zip(self.agents, self.get_random_orientation(NO_OF_AGENTS)))
        self.parking_rects = self.get_random_positions(NO_OF_LOTS, [])
        self.obstacle_rects = self.get_random_positions(NO_OF_OBSTACLES, self.parking_rects)
        self.agent_rects = dict(zip(self.agents, self.get_random_positions(NO_OF_AGENTS, self.parking_rects + self.obstacle_rects)))
        infos = {i: {} for i in self.agents}
        self.fill_surface()
        observations = {i: self.image for i in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions):
        rewards = {i: 0 for i in self.agents}
        terminated = {i: False for i in self.agents}
        truncated = {i: False for i in self.agents}
        infos = {i: {} for i in self.agents}
        self.time_step += 1
        if self.time_step >= MAX_EPISODE_LENGTH:
            self.agents = []
        agents_to_remove = set()
        for agent in actions.keys():
            action = actions[agent]
            # Move logic as above (see feature env for reference)
            if action == 3:  # straight
                orientation = self.agent_orientations[agent]
                if orientation == "up":
                    self.agent_rects[agent].y -= CAR_SPEED
                elif orientation == "down":
                    self.agent_rects[agent].y += CAR_SPEED
                elif orientation == "left":
                    self.agent_rects[agent].x -= CAR_SPEED
                elif orientation == "right":
                    self.agent_rects[agent].x += CAR_SPEED
            elif action == 4:  # backwards
                orientation = self.agent_orientations[agent]
                if orientation == "up":
                    self.agent_rects[agent].y += CAR_SPEED
                elif orientation == "down":
                    self.agent_rects[agent].y -= CAR_SPEED
                elif orientation == "left":
                    self.agent_rects[agent].x += CAR_SPEED
                elif orientation == "right":
                    self.agent_rects[agent].x -= CAR_SPEED
            elif action == 1:  # turn left
                cur = self.agent_orientations[agent]
                turns = {"up": "left", "down": "right", "left": "down", "right": "up"}
                self.agent_orientations[agent] = turns[cur]
            elif action == 2:  # turn right
                cur = self.agent_orientations[agent]
                turns = {"up": "right", "down": "left", "left": "up", "right": "down"}
                self.agent_orientations[agent] = turns[cur]
            # Clip to board
            self.agent_rects[agent].left = max(0, self.agent_rects[agent].left)
            self.agent_rects[agent].right = min(STATE_WIDTH, self.agent_rects[agent].right)
            self.agent_rects[agent].top = max(0, self.agent_rects[agent].top)
            self.agent_rects[agent].bottom = min(STATE_HEIGHT, self.agent_rects[agent].bottom)
            # Check collisions
            for other_agent in self.agents:
                if agent != other_agent and self.agent_rects[agent].colliderect(self.agent_rects[other_agent]):
                    rewards[agent] -= 500
                    rewards[other_agent] -= 500
                    terminated[agent] = True
                    terminated[other_agent] = True
                    agents_to_remove.add(agent)
                    agents_to_remove.add(other_agent)
            for obstacle_rect in self.obstacle_rects:
                if self.agent_rects[agent].colliderect(obstacle_rect):
                    rewards[agent] -= 500
                    terminated[agent] = True
                    agents_to_remove.add(agent)
            for parking_rect in self.parking_rects:
                if self.agent_rects[agent].colliderect(parking_rect):
                    rewards[agent] += 2000
                    terminated[agent] = True
                    agents_to_remove.add(agent)
                    self.successfully_parked.append([parking_rect, self.agent_orientations[agent]])
            rewards[agent] -= 1
        self.fill_surface()
        observations = {i: self.image for i in self.agents}
        for agent in agents_to_remove:
            del self.agent_rects[agent]
            del self.agent_orientations[agent]
            self.agents.remove(agent)
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminated, truncated, infos