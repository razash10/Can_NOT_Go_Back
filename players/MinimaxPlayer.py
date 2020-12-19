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
        limit = min(self.board.shape) * 3

        # At least "buffer" in ms left to run
        while self.time_left() > self.buffer and depth <= limit:
            minimax_algo = MiniMax(self.utility, self.succ, self.perform_move, None)
            players_positions = [self.pos, self.rival]
            score, direction = minimax_algo.search(players_positions, depth, True)
            print('score=' + str(score) + ' direction=' + str(direction) + ' depth=' + str(depth))
            depth += 1
            if best_direction is None or score > best_score:
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

    def h_successors_by_depth(self, pos, depth):
        temp_board = self.board
        queue = [pos]
        count_successors = 0

        while queue:
            s = queue.pop(0)
            count_successors += 1
            for i in self.succ(s):
                if self.board[i] != -2:
                    queue.append(i)
                    self.board[i] = -2
            depth -= 1
            if depth <= 0:
                break

        self.board = temp_board
        return count_successors

    def h_dist_from_rival(self):
        pos1 = self.pos
        pos2 = self.rival
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def h_directions_diff(self):
        return len(self.succ(self.pos)) - len(self.succ(self.rival))

    def manhattan_distance(self, pos):
        return np.abs(self.pos[0] - pos[0]) + np.abs(self.pos[1] - pos[1])

    def h_minimax(self, pos):
        v1 = self.h_successors_by_depth(pos, min(self.board.shape))
        v2 = self.h_dist_from_rival() / self.board.size
        v3 = self.h_directions_diff() / 3
        v4 = self.fruits_ate
        return pow((v1 - v2 + v3), (v4 + 1))

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, pos, is_it_me):
        if is_it_me:
            my_pos = pos
            rival_pos = self.rival
        else:
            my_pos = self.pos
            rival_pos = pos

        assert (self.board[my_pos] == 1)
        assert (self.board[rival_pos] == 2)

        my_moves = self.succ(my_pos)
        rival_moves = self.succ(rival_pos)

        win, lose = float('inf'), float('-inf')

        if len(rival_moves) == 0 and len(my_moves) > 0:
            return win
        elif len(my_moves) == 0 and len(rival_moves) > 0:
            return lose

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

        if player_index == 1:
            self.pos = next_pos
        elif player_index == 2:
            self.rival = next_pos
