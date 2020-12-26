import numpy as np


def h_successors_by_depth(player, pos, depth):
    temp_board = player.board
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

    player.board = temp_board
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
    v1 = h_successors_by_depth(player, pos, (min(player.board.shape) / 4))
    v2 = h_dist_from_rival(player) / player.board.size
    v3 = h_directions_diff(player) / 3
    v4 = h_diff_fruits_values(player, pos) / player.penalty_score
    return v1 - v2 + v3 + v4
