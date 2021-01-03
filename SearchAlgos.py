"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import operator
import time


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        pos = None
        if maximizing_player:
            pos = state[0]
        else:
            pos = state[1]

        buffer = 200
        time_left = (state[3] - (time.time() - state[2])) * 1000

        if time_left < buffer or depth <= 0 or len(self.succ(pos)) == 0:
            res = (0, None)
            return res

        scores = []
        for next_pos in self.succ(pos):
            direction = None

            self.perform_move(pos, next_pos)

            if maximizing_player:
                state[0] = next_pos
                score = self.utility(state)
                direction = tuple(map(operator.sub, next_pos, pos))
                scores.append((score + self.search(state, depth - 1, not maximizing_player)[0], direction))
                state[0] = pos
            else:
                state[1] = next_pos
                score = self.utility(state)
                scores.append((score + self.search(state, depth - 1, not maximizing_player)[0], direction))
                state[1] = pos

            self.perform_move(next_pos, pos)

        assert len(scores) != 0

        return max(scores) if maximizing_player else min(scores)


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        pos = None
        if maximizing_player:
            pos = state[0]
        else:
            pos = state[1]

        buffer = 200
        time_left = (state[3] - (time.time() - state[2])) * 1000

        if time_left < buffer or depth <= 0 or len(self.succ(pos)) == 0:
            res = (0, None)
            return res

        scores = []
        for next_pos in self.succ(pos):
            direction = None
            self.perform_move(pos, next_pos)

            if maximizing_player:
                state[0] = next_pos
                direction = tuple(map(operator.sub, next_pos, pos))
                score = self.utility(state)
                scores.append((score + self.search(state, depth - 1, not maximizing_player)[0], direction))
                curr_max = max(scores)[0]
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    res = (float('inf'), direction)
                    state[0] = pos
                    self.perform_move(next_pos, pos)
                    return res
                state[0] = pos

            else:
                state[1] = next_pos
                score = self.utility(state)
                scores.append((score + self.search(state, depth - 1, not maximizing_player)[0], direction))
                curr_min = min(scores)[0]
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    res = (float('-inf'), direction)
                    state[1] = pos
                    self.perform_move(next_pos, pos)
                    return res
                state[1] = pos

            self.perform_move(next_pos, pos)

        assert len(scores) != 0

        return max(scores) if maximizing_player else min(scores)
