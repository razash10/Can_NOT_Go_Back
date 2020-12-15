"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import MiniMax
import numpy as np


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        minimax_algo = MiniMax(self.utility, self.succ, self.perform_move, None)  # FIXME: Replace Nones with working functions
        self.board = None
        self.pos = None
        self.fruits_states = {}

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        # TODO: erase the following line and implement this function.
        # TODO: erase the following line and implement this function.
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)

        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] > 2:
                    self.fruits_states['(i,j)'] = board[i][j]

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # TODO: erase the following line and implement this function.
        self.board[pos] = -1

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
    @staticmethod
    def count_ones(board):
        counter = len(np.where(board == 1)[0])
        return counter

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, pos):
        assert (pos is not None)
        i = pos[0]
        j = pos[1]

        if self.board[i][j] == 0:
            return 1
        elif self.board[i][j] > 2:
            return self.board[i][j]

        return 0

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
        else:  # returning backward
            if pos in self.fruits_states:
                self.board[pos] = self.fruits_states[pos]
            else:
                self.board[pos] = 0

        self.board[next_pos] = player_index
