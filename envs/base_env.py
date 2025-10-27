import numpy as np
import pygame
from gymnasium import spaces

from utils.utils import (
    STATE_WIDTH, STATE_HEIGHT, TILE_SIZE, CAR_HEIGHT, CAR_WIDTH,
    NUMBER_OF_ACTIONS, grid_to_pixels, map_orientation_to_numeric,
    get_random_orientation, get_random_rect, load_car_images, load_obstacle_image
)

class BaseParkingEnv:
    """
    Common functionality for all Parking environments.
    Inherit from this class in each feature/image env to reduce duplication.
    """

    def __init__(self):
        self.screen_width = STATE_WIDTH
        self.screen_height = STATE_HEIGHT
        self.car_images = None
        self.obstacle_image = None

    def spawn_car(self, occupied=[]):
        return get_random_rect(occupied_rects=occupied)

    def spawn_parking(self, car_rect, occupied=[]):
        while True:
            rect = get_random_rect(occupied_rects=occupied)
            if not rect.colliderect(car_rect):
                return rect

    def spawn_obstacles(self, car_rect, parking_rect, n=4, occupied=[]):
        obstacle_rects = []
        while len(obstacle_rects) < n:
            ob_rect = get_random_rect(occupied_rects=[car_rect, parking_rect] + obstacle_rects + occupied)
            obstacle_rects.append(ob_rect)
        return obstacle_rects

    def get_orientation(self, prng=None):
        return get_random_orientation(prng=prng)
