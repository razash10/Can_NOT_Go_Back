"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import MiniMax
import numpy as np
import time


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.board = None
        self.pos = None
        self.fruits_states = {}
        self.time = 0
        self.start = 0
        self.buffer = 50
        self.rival = None
        self.init_pos = None
        self.fruits_ate = 0

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        # TODO: erase the following line and implement this function.
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        self.init_pos = self.pos
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] > 2:
                    self.fruits_states[(i, j)] = board[i][j]
                if board[i][j] == 2:
                    self.rival = (i, j)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        # TODO: erase the following line and implement this function.

        self.time = time_limit
        self.start = time.time()
        depth = 1
        best_direction = None
        best_score = float('-inf')
        limit = min(self.board.shape) * 2

        # At least "buffer" in ms left to run
        while self.time_left() > self.buffer and depth <= limit:
            minimax_algo = MiniMax(self.utility, self.succ, self.perform_move, None)
            ai_pos = self.get_player_position(1)
            player_pos = self.get_player_position(2)
            players_positions = [ai_pos, player_pos]
            score, direction = minimax_algo.search(players_positions, depth, True)
            print('score=' + str(score) + ' direction=' + str(direction) +
                  ' depth=' + str(depth) + ' fruits_ate=' + str(self.fruits_ate))
            depth += 1
            if score > best_score:
                best_score = score
                best_direction = direction

        assert (best_direction is not None)

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
        # TODO: erase the following line and implement this function.
        self.board[pos] = 2
        self.board[self.rival] = -1
        self.rival = pos

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_states = fruits_on_board_dict

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed
    def is_legal_move(self, next_pos):
        i = next_pos[0]
        j = next_pos[1]
        if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (self.board[i][j] not in [-1, 1, 2]):
            return True
        else:
            return False

    def time_left(self):
        #   Compute time left for the run in ms
        return (self.time - (time.time() - self.start)) * 1000

    def get_player_position(self, player_index):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == player_index:
                    player_pos = (i, j)
                    return player_pos

    def h_successors_by_depth(self, pos, depth):
        successors = set(self.succ(pos))

        if depth <= 0 or len(successors) == 0:
            return set(pos)

        for next_pos in successors:
            self.perform_move(pos, next_pos)
            next_successors = self.h_successors_by_depth(next_pos, depth - 1)
            successors.union(next_successors)
            self.perform_move(next_pos, pos)

        return successors

    def h_dist_from_rival(self):
        pos1 = self.pos
        pos2 = self.rival
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def h_directions_diff(self):
        return len(self.succ(self.pos)) - len(self.succ(self.rival))

    def manhattan_distance(self, pos):
        return np.abs(self.pos[0] - pos[0]) + np.abs(self.pos[1] - pos[1])

    def h_minimax(self, pos):
        v1 = len(self.h_successors_by_depth(pos, min(self.board.shape)))
        v2 = self.h_dist_from_rival() / self.board.size
        v3 = self.h_directions_diff() / 3
        v4 = self.fruits_ate
        return pow((v1 - v2 + v3), (v4 + 1))

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, pos):
        return self.h_minimax(pos)

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
            if next_pos in self.fruits_states and player_index == 1:
                self.fruits_ate += 1
            self.board[pos] = -1

        else:  # returning backward
            if pos in self.fruits_states:
                self.board[pos] = self.fruits_states[pos]
                if player_index == 1:
                    self.fruits_ate -= 1
            else:
                self.board[pos] = 0

        self.board[next_pos] = player_index
