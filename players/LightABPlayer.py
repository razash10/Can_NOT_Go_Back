"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta
import numpy as np
import time
import random


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.penalty_score = penalty_score
        self.board = None
        self.pos = None
        self.rival_pos = None
        self.fruits_dict = {}
        self.fruits_ate = {}
        self.rival_fruits_ate = {}
        self.first_turn = True

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        pos = np.where(board == 1)
        rival_pos = np.where(board == 2)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        self.rival_pos = tuple(ax[0] for ax in rival_pos)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        start_time = time.time()
        depth = 1
        best_direction = None
        best_score = float('-inf')
        limit = 3

        # At least "buffer" in ms left to run
        while depth <= limit:
            minimax_algo = AlphaBeta(self.utility, self.succ, self.perform_move, None)
            state = [self.pos, self.rival_pos, start_time, time_limit]
            score, direction = minimax_algo.search(state, depth, True)
            depth += 1
            if best_direction is None or score > best_score:
                best_score = score
                best_direction = direction

        if best_direction is None:
            available_moves = self.succ(self.pos)
            random.shuffle(available_moves)
            best_direction = available_moves[0]

        i = self.pos[0] + best_direction[0]
        j = self.pos[1] + best_direction[1]

        self.perform_move(self.pos, (i, j))

        self.pos = (i, j)

        return best_direction

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        if self.first_turn:
            self.first_turn = False
            if self.board[pos] > 0:
                self.rival_fruits_ate[pos] = self.board[pos]
        self.perform_move(self.rival_pos, pos)

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_dict = fruits_on_board_dict

    ########## helper functions in class ##########
    @staticmethod
    def count_ones(board):
        counter = len(np.where(board == 1)[0])
        return counter

    def h_fruits_are_yummy(self, pos):
        if pos in self.fruits_dict:
            return self.fruits_dict[pos]
        elif pos in self.fruits_ate:
            return self.fruits_ate[pos]
        elif pos in self.rival_fruits_ate:
            return self.rival_fruits_ate[pos]
        else:
            return 0

    def h_simple_player(self, pos):
        num_steps_available = len(self.succ(pos))

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    ########## helper functions for AlphaBeta algorithm ##########
    def utility(self, state):
        pos = state[0]
        v1 = self.h_fruits_are_yummy(pos)
        v2 = self.h_simple_player(pos)
        return v1 + v2

    def succ(self, pos):
        next_poses = []

        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]

            # check legal move
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (self.board[i][j] not in [-1, 1, 2]):
                next_pos = (i, j)
                next_poses.append(next_pos)

        return next_poses

    def perform_move(self, pos, next_pos):
        assert (self.board[pos] in [1, 2])
        assert (self.board[next_pos] not in [1, 2])

        player_index = self.board[pos]

        if self.board[next_pos] != -1:  # moving forward
            self.board[pos] = -1
            if next_pos in self.fruits_dict:
                fruit_val = self.fruits_dict.pop(next_pos)
                if player_index == 1:
                    self.fruits_ate[next_pos] = fruit_val
                elif player_index == 2:
                    self.rival_fruits_ate[next_pos] = fruit_val

        else:  # returning backward
            if pos in self.fruits_ate:
                fruit_val = self.fruits_ate.pop(pos)
                self.board[pos] = fruit_val
                self.fruits_dict[pos] = fruit_val
            elif pos in self.rival_fruits_ate:
                fruit_val = self.rival_fruits_ate.pop(pos)
                self.board[pos] = fruit_val
                self.fruits_dict[pos] = fruit_val
            else:
                self.board[pos] = 0

        self.board[next_pos] = player_index

        if player_index == 1:
            self.pos = next_pos
        elif player_index == 2:
            self.rival_pos = next_pos

