import operator
import numpy as np
import os

#TODO: edit the alpha and beta initialization values for AlphaBeta algorithm.
# instead of 'None', write the real initialization value, learned in class.
# hint: you can use np.inf
ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf


def get_directions():
    """Returns all the possible directions of a player in the game as a list of tuples.
    """
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]


def tup_add(t1, t2):
    """
    returns the sum of two tuples as tuple.
    """
    return tuple(map(operator.add, t1, t2))


def get_board_from_csv(board_file_name):
    """Returns the board data that is saved as a csv file in 'boards' folder.
    The board data is a list that contains: 
        [0] size of board
        [1] blocked poses on board
        [2] starts poses of the players
    """
    board_path = os.path.join('boards', board_file_name)
    board = np.loadtxt(open(board_path, "rb"), delimiter=" ")
    
    # mirror board
    board = np.flipud(board)
    i, j = len(board), len(board[0])
    blocks = np.where(board == -1)
    blocks = [(blocks[0][i], blocks[1][i]) for i in range(len(blocks[0]))]
    start_player_1 = np.where(board == 1)
    start_player_2 = np.where(board == 2)
    
    if len(start_player_1[0]) != 1 or len(start_player_2[0]) != 1:
        raise Exception('The given board is not legal - too many start locations.')
    
    start_player_1 = (start_player_1[0][0], start_player_1[1][0])
    start_player_2 = (start_player_2[0][0], start_player_2[1][0])

    return [(i, j), blocks, [start_player_1, start_player_2]]


def h_successors_by_depth(player, pos, depth):
    queue = [pos]
    count_successors = 0

    while queue:
        s = queue.pop(0)
        count_successors += 1
        for i in player.succ(s):
            if player.board[i] != -2:
                queue.append(i)
                player.board[i] = -2
        depth -= 1
        if depth <= 0:
            break

    return count_successors


def h_dist_from_rival(player):
    pos1 = player.pos
    pos2 = player.rival_pos
    return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


def h_directions_diff(player):
    return len(player.succ(player.pos)) - len(player.succ(player.rival_pos))


def h_diff_fruits_values(player, pos):
    assert player.board[pos] in [1, 2]
    if player.board[pos] == 1:
        return player.fruits_score - player.rival_fruits_score
    elif player.board[pos] == 2:
        return player.rival_fruits_score - player.fruits_score


def h_minimax(player, pos):
    v1 = h_successors_by_depth(player, pos, min(player.board.shape))
    v2 = h_dist_from_rival(player) / player.board.size
    v3 = h_directions_diff(player) / 3
    v4 = h_diff_fruits_values(player, pos) / player.penalty_score
    return v1 - v2 + v3 + v4
