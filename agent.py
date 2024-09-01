import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.90  # discount rate, MUST be smaller than 1
        self.memory = deque(
            maxlen=MAX_MEMORY
        )  # if exceed memory, automatically removes elements from left : popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # move direction (only 1 is true)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            # EXP:
            # game.four_consecutive_turns(),
            # game.path_to_tail(point_l),
            # game.path_to_tail(point_r),
            # game.path_to_tail(point_u),
            # game.path_to_tail(point_d),
            # game.snake_in_direction(Direction.LEFT),
            # game.snake_in_direction(Direction.RIGHT),
            # game.snake_in_direction(Direction.UP),
            # game.snake_in_direction(Direction.DOWN),
            # Add new states
            # game.snake[-1].x < game.snake[0].x,
            # game.snake[-1].x > game.snake[0].x,
            # game.snake[-1].y < game.snake[0].y,
            # game.snake[-1].y > game.snake[0].y,
        ]

        # print(state[10:14])

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):  # done means game_over
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation

        # the more games, the smaller epsilon gets, which means less frequent rand(0, 200) is less than epsilon, explore less often
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # print("1")
        # get old state
        state_old = agent.get_state(game)

        # print("1.1")
        # get move
        final_move = agent.get_action(state_old)

        # print("1.2")
        # perform move and get new state
        # print(final_move)
        reward, done, score = game.play_step(final_move)
        # print("1.3")
        state_new = agent.get_state(game)

        # print("2")
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # print("3")
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        # print("end")


if __name__ == "__main__":
    train()
