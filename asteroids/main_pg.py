"""
Trains a Neural Network AI Input Method using policy gradient
"""

import argparse

import numpy as np
import torch
from torch import nn

from game import Action, Game
from nn_ai_input import NNAIInputMethod, NNAIParameters


def discounted_rewards(rewards, gamma=0.99, normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret - np.mean(ret)) / (np.std(ret) + eps)
    return ret


def train_on_game_results(model, optimizer, x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss


# copies to much from game_runner.py refactor later after it works
def run_game(width, height, params):
    """
    Run a game with the given parameters.
    Return lists of game_state, action_taken, probability, reward
    One element per frame
    """
    game = Game(width, height)
    input_method = NNAIInputMethod(game=game, parameters=params, keep_data=True)
    dt = 1 / 60
    while game.player_alive:
        # Clear turn and acceleration every frame
        game.clear_turn()
        game.clear_acc()

        # Get and execute action
        action = input_method.get_move()
        execute_action(game, action)

        # Update game state
        game.update(dt)

    return [], [], [], []


# need to refactor and use game_runner.py. But being lazy to get it to work now
def execute_action(game, action):
    """Execute action on game"""
    if action == Action.TURN_LEFT:
        game.turning_left()
    elif action == Action.TURN_RIGHT:
        game.turning_right()
    elif action == Action.ACCELERATE:
        game.accelerate()
    elif action == Action.DECELERATE:
        game.decelerate()
    elif action == Action.SHOOT:
        game.shoot()
    elif action == Action.NO_ACTION:
        game.no_action()


def train_model(width, height):
    params = NNAIParameters()
    model = params.model
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    alpha = 1e-4
    for epoch in range(600):
        states, actions, probs, rewards = run_game(width, height, params)
        # recall that an action is 0 or 1 based on the index of the model output selected by probability sample
        # we had an array of actions but we ran np.vstack(action) which makde it into a list of lists where each internal list had one element
        # T transposes the array making it into a list with one list inside and all the elements in that
        # taking out the 0 element grabs that internal list
        # np.eye(2) converts a numeric value with 2 possible values into a 2x2 matrix one hot encoded
        # value of 0 to [1, 0] and value of 1 to [0, 1]
        one_hot_actions = np.eye(params.num_actions)[actions.T][0]
        # a probability row is a list of two probabilities, after 1 hot encoding above we end up
        # with the action taken as a 1. so a single subtraction is like [1, 0] - [0.7, 0.3] = [0.3, -0.3]
        gradients = one_hot_actions - probs
        dr = discounted_rewards(rewards)
        # weight the gradient by dicsounted rewards
        gradients *= dr
        # target here is not a labeled correct value, just a nudge to the model
        # because alpha is small it will take some big rewards to make target much different than the initial probs
        # so when the rewards are small we don't change thigns much
        target = alpha * np.vstack([gradients]) + probs
        train_on_game_results(model, opt, states, target)
        if epoch % 100 == 0:
            print(f"{epoch} -> {np.sum(rewards)}")


def main():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for optimizing AI parameters"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Game width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Game height (default: 720)",
    )
    args = parser.parse_args()
    train_model(args.width, args.height)


if __name__ == "__main__":
    main()
