import pygame
import sys
import random
from utils.utils import (
    STATE_WIDTH, STATE_HEIGHT, TILE_SIZE, CAR_WIDTH, CAR_HEIGHT,
    BACKGROUND_COLOR, CAR_SPEED
)

pygame.init()
screen = pygame.display.set_mode((STATE_WIDTH, STATE_HEIGHT))
pygame.display.set_caption("Car Parking Game")
clock = pygame.time.Clock()

# Load images
car_images = [
    pygame.image.load("assets/car-up.png"),
    pygame.image.load("assets/car-down.png"),
    pygame.image.load("assets/car-left.png"),
    pygame.image.load("assets/car-right.png")
]
obstacle_image = pygame.image.load("assets/obstacle.png")

def grid_to_pixels(x, y):
    return x * TILE_SIZE, y * TILE_SIZE

def get_random_car_position():
    car_tile = (random.randint(0, STATE_WIDTH // TILE_SIZE - 1), random.randint(0, STATE_HEIGHT // TILE_SIZE - 1))
    car_rect = pygame.Rect(grid_to_pixels(*car_tile), (TILE_SIZE, TILE_SIZE))
    car_rect.x += (TILE_SIZE - CAR_WIDTH) // 2
    car_rect.y += (TILE_SIZE - CAR_HEIGHT) // 2
    return car_rect

def get_random_orientation():
    return random.choice(["up", "down", "left", "right"])

def get_random_obstacle_positions():
    obstacle_rects = []
    while len(obstacle_rects) < 4:
        tile = (random.randint(0, STATE_WIDTH // TILE_SIZE - 1), random.randint(0, STATE_HEIGHT // TILE_SIZE - 1))
        rect = pygame.Rect(grid_to_pixels(*tile), (TILE_SIZE, TILE_SIZE))
        overlap = any(rect.colliderect(obs) for obs in obstacle_rects)
        if not overlap:
            obstacle_rects.append(rect)
    return obstacle_rects

def get_random_parking_position():
    tile = (random.randint(0, STATE_WIDTH // TILE_SIZE - 1), random.randint(0, STATE_HEIGHT // TILE_SIZE - 1))
    return pygame.Rect(grid_to_pixels(*tile), (TILE_SIZE, TILE_SIZE))

car_rect = get_random_car_position()
car_orientation = get_random_orientation()
parking_rect = get_random_parking_position()
obstacles_rect = get_random_obstacle_positions()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_3:
                if car_orientation == "up":
                    car_rect.y -= CAR_SPEED
                elif car_orientation == "down":
                    car_rect.y += CAR_SPEED
                elif car_orientation == "left":
                    car_rect.x -= CAR_SPEED
                elif car_orientation == "right":
                    car_rect.x += CAR_SPEED
            elif event.key == pygame.K_4:
                if car_orientation == "up":
                    car_rect.y += CAR_SPEED
                elif car_orientation == "down":
                    car_rect.y -= CAR_SPEED
                elif car_orientation == "left":
                    car_rect.x += CAR_SPEED
                elif car_orientation == "right":
                    car_rect.x -= CAR_SPEED
            elif event.key == pygame.K_1:
                mapping = {"up": "left", "down": "right", "left": "down", "right": "up"}
                car_orientation = mapping[car_orientation]
            elif event.key == pygame.K_2:
                mapping = {"up": "right", "down": "left", "left": "up", "right": "down"}
                car_orientation = mapping[car_orientation]

            if parking_rect.colliderect(car_rect):
                print("Game Over - You parked the car!")
                pygame.quit()
                sys.exit()
            for obstacle_rect in obstacles_rect:
                if car_rect.colliderect(obstacle_rect):
                    print("Game Over - You collided with an obstacle!")
                    pygame.quit()
                    sys.exit()

            car_rect.left = max(0, car_rect.left)
            car_rect.right = min(STATE_WIDTH, car_rect.right)
            car_rect.top = max(0, car_rect.top)
            car_rect.bottom = min(STATE_HEIGHT, car_rect.bottom)

    screen.fill(BACKGROUND_COLOR)
    # Draw car
    idx = ["up", "down", "left", "right"].index(car_orientation)
    car_sprite = car_images[idx]
    screen.blit(car_sprite, car_rect)
    # Draw obstacles
    for obstacle_rect in obstacles_rect:
        screen.blit(obstacle_image, obstacle_rect)
    # Draw parking
    pygame.draw.rect(screen, (255, 255, 0), parking_rect)
    pygame.display.flip()
    clock.tick(60)