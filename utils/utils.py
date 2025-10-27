import random

STATE_WIDTH = 720
STATE_HEIGHT = 720
TILE_SIZE = 60
GRID_WIDTH = STATE_WIDTH // TILE_SIZE
GRID_HEIGHT = STATE_HEIGHT // TILE_SIZE
CAR_HEIGHT = 60
CAR_WIDTH = 40
FPS = 200
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
YELLOW = (255, 255, 0)
CAR_SPEED = 60
NUMBER_OF_ACTIONS = 5
NO_OF_OBSTACLES = 4
ORIENTATIONS = ["up", "down", "left", "right"]

def grid_to_pixels(x, y):
    return x * TILE_SIZE, y * TILE_SIZE

def map_orientation_to_numeric(orientation):
    return ORIENTATIONS.index(orientation)

def get_random_orientation(prng=None):
    return (prng or random).choice(ORIENTATIONS)

def get_random_grid_tile(prng=None):
    prng = prng or random
    return prng.randint(0, GRID_WIDTH - 1), prng.randint(0, GRID_HEIGHT - 1)

def get_random_rect(prng=None, occupied_rects=None):
    import pygame
    occupied_rects = occupied_rects or []
    while True:
        x, y = get_random_grid_tile(prng)
        rect = pygame.Rect(grid_to_pixels(x, y), (TILE_SIZE, TILE_SIZE))
        rect.x += (TILE_SIZE - CAR_WIDTH) // 2
        rect.y += (TILE_SIZE - CAR_HEIGHT) // 2
        if not any(rect.colliderect(obj) for obj in occupied_rects):
            return rect

def load_car_images(asset_folder="../assets"):
    import pygame
    return [
        pygame.image.load(f"{asset_folder}/car-up.png"),
        pygame.image.load(f"{asset_folder}/car-down.png"),
        pygame.image.load(f"{asset_folder}/car-left.png"),
        pygame.image.load(f"{asset_folder}/car-right.png")
    ]

def load_obstacle_image(asset_folder="../assets"):
    import pygame
    return pygame.image.load(f"{asset_folder}/obstacle.png")
