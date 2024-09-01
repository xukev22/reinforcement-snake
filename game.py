import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)
# font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 500


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def four_consecutive_turns(self):
        consecutive_turns = 0
        last_direction = None

        for i in range(len(self.snake) - 1):
            current_point = self.snake[i]
            next_point = self.snake[i + 1]

            # Determine the direction between the current point and the next point
            if next_point.x > current_point.x:
                direction = Direction.RIGHT
            elif next_point.x < current_point.x:
                direction = Direction.LEFT
            elif next_point.y > current_point.y:
                direction = Direction.DOWN
            elif next_point.y < current_point.y:
                direction = Direction.UP

            # Check if there is a change in direction
            if last_direction is not None and direction != last_direction:
                consecutive_turns += 1
            else:
                consecutive_turns = (
                    0  # Reset the counter if the direction remains the same
                )

            if consecutive_turns >= 4:
                return True

            last_direction = direction

        return False

    def snake_in_direction(self, direction):
        point_to_add = None
        cur_point = self.head

        if direction == Direction.RIGHT:
            point_to_add = Point(self.head.x + BLOCK_SIZE, self.head.y)
        elif direction == Direction.LEFT:
            point_to_add = Point(self.head.x - BLOCK_SIZE, self.head.y)
        elif direction == Direction.DOWN:
            point_to_add = Point(self.head.x, self.head.y + BLOCK_SIZE)
        else:  # up
            point_to_add = Point(self.head.x, self.head.y - BLOCK_SIZE)

        while (
            cur_point.x <= 640
            and cur_point.x >= 0
            and cur_point.y >= 0
            and cur_point.y <= 480
        ):
            if cur_point in self.snake:
                return True
            cur_point += point_to_add

        return False

    def path_to_tail(self, start):
        queue = [start]
        visited = [start]

        while queue:
            point = queue.pop()
            # print(point)
            # print(self.snake[-1])

            if point == self.snake[-1]:
                return True

            # left, right, up, down
            potential_neigbors = [
                Point(point.x - BLOCK_SIZE, point.y),
                Point(point.x + BLOCK_SIZE, point.y),
                Point(point.x, point.y - BLOCK_SIZE),
                Point(point.x, point.y + BLOCK_SIZE),
            ]

            for point in potential_neigbors:
                if (
                    point not in visited
                    and point.x >= 0
                    and point.x <= 640
                    and point.y >= 0
                    and point.y <= 480
                ):
                    visited.append(point)
                    queue.append(point)

        return False

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            # print(pygame.event.event_name(event.type))
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # print("2")
        # 2. move
        straight = self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # print("3")
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # print("4")
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward=10
            self._place_food()
        else:
            self.snake.pop()

        # print("5")
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        straight = False

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
            straight = True
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        return straight
