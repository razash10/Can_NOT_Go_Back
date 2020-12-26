"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import MiniMax
import numpy as np
import time
import heuristics


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.penalty_score = penalty_score
        self.board = None
        self.pos = None
        self.rival_pos = None
        self.fruits_dict = {}
        self.fruits_ate = {}
        self.rival_fruits_ate = {}
        self.fruits_score = 0
        self.rival_fruits_score = 0
        self.first_turn = True

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
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        buffer = 200
        depth = 1
        best_direction = None
        best_score = float('-inf')
        limit = (2 * self.board.size) / 3
        time_left = (time_limit - (time.time() - start_time)) * 1000

        print()
        print('fruits_score=' + str(self.fruits_score) + ' rival_fruits_score=' + str(self.rival_fruits_score))
        print('fruits_ate=' + str(self.fruits_ate) + ' rival_fruits_ate=' + str(self.rival_fruits_ate))
        print('fruits_dict=' + str(self.fruits_dict))
        print()

        # At least "buffer" in ms left to run
        while time_left > buffer and depth <= limit:
            minimax_algo = MiniMax(self.utility, self.succ, self.perform_move, None)
            state = [self.pos, self.rival_pos, start_time, time_limit]
            score, direction = minimax_algo.search(state, depth, True)
            print('score=' + str(score) + ' direction=' + str(direction) + ' depth=' + str(depth))
            depth += 1
            if best_direction is None or score > best_score:
                best_score = score
                best_direction = direction
            time_left = (time_limit - (time.time() - start_time)) * 1000

        assert (best_direction is not None)

        i = self.pos[0] + best_direction[0]
        j = self.pos[1] + best_direction[1]

        self.perform_move(self.pos, (i, j))

        self.pos = (i, j)

        print()
        print('fruits_score=' + str(self.fruits_score) + ' rival_fruits_score=' + str(self.rival_fruits_score))
        print('fruits_ate=' + str(self.fruits_ate) + ' rival_fruits_ate=' + str(self.rival_fruits_ate))
        print('fruits_dict=' + str(self.fruits_dict))
        print()

        return best_direction

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # TODO: erase the following line and implement this function.
        if self.first_turn:
            self.first_turn = False
            if self.board[pos] > 0:
                self.rival_fruits_ate[pos] = self.board[pos]
                self.rival_fruits_score += self.board[pos]
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
    # TODO: add here helper functions in class, if needed
    def update_fruits_scores(self, player_index):
        if player_index == 1:
            self.fruits_score = 0
            for fruit in self.fruits_ate:
                self.fruits_score += self.fruits_ate[fruit]
        elif player_index == 2:
            self.rival_fruits_score = 0
            for fruit in self.rival_fruits_ate:
                self.rival_fruits_score += self.rival_fruits_ate[fruit]

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    def utility(self, state):
        my_pos = state[0]
        rival_pos = state[1]

        assert (self.board[my_pos] == 1)
        assert (self.board[rival_pos] == 2)

        my_moves = self.succ(my_pos)
        rival_moves = self.succ(rival_pos)
        win, draw, lose = float('inf'), 0, float('-inf')

        if len(rival_moves) == 0 and len(my_moves) == 0:  # end of game and penalty goes to both
            if self.fruits_score > self.rival_fruits_score:
                return win
            elif self.fruits_score < self.rival_fruits_score:
                return lose
            else:
                return draw
        elif len(rival_moves) == 0 and len(my_moves) > 0:  # end of game and penalty goes to rival
            if self.fruits_score + self.penalty_score > self.rival_fruits_score:
                return win
            elif self.fruits_score + self.penalty_score < self.rival_fruits_score:
                return lose
            else:
                return draw
        elif len(my_moves) == 0 and len(rival_moves) > 0:  # end of game and penalty goes to me
            if self.fruits_score > self.rival_fruits_score + self.penalty_score:
                return win
            elif self.fruits_score < self.rival_fruits_score + self.penalty_score:
                return lose
            else:
                return draw

        return heuristics.h_minimax(self, my_pos)

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

        self.update_fruits_scores(player_index)
