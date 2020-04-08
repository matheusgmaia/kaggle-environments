# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from os import path
from random import choice

EMPTY = 0


def play(board, column, mark, config):
    columns = config.columns
    rows = config.rows
    row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    board[column + (row * columns)] = mark


def is_win(board, column, mark, config, has_played=True):
    columns = config.columns
    rows = config.rows
    inarow = config.inarow - 1
    row = (
        min([r for r in range(rows) if board[column + (r * columns)] == mark])
        if has_played
        else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    )

    def count(offset_row, offset_column):
        for i in range(1, inarow + 1):
            r = row + offset_row * i
            c = column + offset_column * i
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= columns
                or board[c + (r * columns)] != mark
            ):
                return i - 1
        return inarow

    return (
        count(1, 0) >= inarow  # vertical.
        or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
        or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
        or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
    )


def random_agent(obs, config):
    return choice([c for c in range(config.columns) if obs.board[c] == EMPTY])


def negamax_agent(obs, config):
    columns = config.columns
    rows = config.rows
    size = rows * columns

    # Due to compute/time constraints the tree depth must be limited.
    max_depth = 4

    def negamax(board, mark, depth):
        moves = sum(1 if cell != EMPTY else 0 for cell in board)

        # Tie Game
        if moves == size:
            return (0, None)

        # Can win next.
        for column in range(columns):
            if board[column] == EMPTY and is_win(board, column, mark, config, False):
                return ((size + 1 - moves) / 2, column)

        # Recursively check all columns.
        best_score = -size
        best_column = None
        for column in range(columns):
            if board[column] == EMPTY:
                # Max depth reached. Score based on cell proximity for a clustering effect.
                if depth <= 0:
                    row = max(
                        [
                            r
                            for r in range(rows)
                            if board[column + (r * columns)] == EMPTY
                        ]
                    )
                    score = (size + 1 - moves) / 2
                    if column > 0 and board[row * columns + column - 1] == mark:
                        score += 1
                    if (
                        column < columns - 1
                        and board[row * columns + column + 1] == mark
                    ):
                        score += 1
                    if row > 0 and board[(row - 1) * columns + column] == mark:
                        score += 1
                    if row < rows - 2 and board[(row + 1) * columns + column] == mark:
                        score += 1
                else:
                    next_board = board[:]
                    play(next_board, column, mark, config)
                    (score, _) = negamax(next_board,
                                         1 if mark == 2 else 2, depth - 1)
                    score = score * -1
                if score > best_score or (score == best_score and choice([True, False])):
                    best_score = score
                    best_column = column

        return (best_score, best_column)

    _, column = negamax(obs.board[:], obs.mark, max_depth)
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == EMPTY])
    return column


# greedy agent
def greedy_agent(observation, cfg):
    import numpy as np
    
    # https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    def search_sequence_numpy(arr,seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------    
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------    
        Output : 1D Array of indices in the input array that satisfy the 
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() >0:
            return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
        else:
            return []         # No match found
    
    class Table():
        def __init__(self,rows,cols,inarow,board=None):
            self.inarow = inarow
            if board is None:
                self.board = np.zeros((rows,cols), dtype=int)
            else:
                self.board = np.reshape(board,(rows,cols))
    

        def _winning_rule(self, arr):
            win1rule = np.ones((self.inarow)) # 1 for 1s
            win2rule = np.ones((self.inarow)) + np.ones((self.inarow)) # 2 for 2s

            player1wins = len(search_sequence_numpy(arr,win1rule)) > 0
            player2wins = len(search_sequence_numpy(arr,win2rule)) > 0
            if player1wins or player2wins:
                return True
            else:
                return False

        def _get_diagonals(self, _board, i, j):
            diags = []
            diags.append(np.diagonal(_board, offset=(j - i)))
            diags.append(np.diagonal(np.rot90(_board), offset=-_board.shape[1] + (j + i) + 1))
            return diags

        def _get_axes(self, _board, i, j):
            axes = []
            axes.append(_board[i,:])
            axes.append(_board[:,j])
            return axes

        def _winning_check(self, i, j):
            '''
            Checks if there is four equal numbers in every
            row, column and diagonal of the matrix
            '''    
            all_arr = []
            all_arr.extend(self._get_axes(self.board, i, j))
            all_arr.extend(self._get_diagonals(self.board, i, j))

            for arr in all_arr:
                winner = self._winning_rule(arr)
                if winner:
                    return True
                else:
                    pass
        def check_win(self, player, column, inarow=None):
            if inarow is not None:
                self.inarow=inarow
            colummn_vec = self.board[:,column]
            non_zero = np.where(colummn_vec != 0)[0]

            if non_zero.size == 0:                        
                i = self.board.shape[0]-1
            else:                                          
                i = non_zero[0]-1

            self.board[i,column] = player # make move
            if self._winning_check(i, column):
                return True
            else:
                self.board[i,column] = 0 #unmove
                return False 
            
    us = observation.mark
    them = 2 if us == 1 else 1
        
    table = Table(cfg.rows,cfg.columns,cfg.inarow,board=observation['board'])
    valid_positions = [c for c in range(cfg.columns) if observation.board[c] == 0]

    #optimal first move
    if not np.any(table.board):
        return cfg.columns // 2

    #can we win with a move?
    for move in range(cfg.columns):
        if table.check_win(us,move): 
            #print("win")
            if move in valid_positions: 
                return move
            
    
    #would they with with a move? block it.
    for move in range(cfg.columns):
        if table.check_win(them,move):
            #print("block")
            if move in valid_positions: 
                return move
    
    #check for next greedy move
    for inarow in range(cfg.inarow-1,0,-1):
        for move in range(cfg.columns):
            if table.check_win(us,move,inarow=inarow): 
                if move in valid_positions: 
                    return move 
                
    #first lowest row
    counts = [np.count_nonzero(table.board[:,c]) for c in valid_positions]
    return valid_positions[np.argmin(counts)]


def rule_agent(observation, configuration):
    
    from random import choice
    
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_vertical_chance(me_or_enemy):
        for i in range(0, 7):
            if observation.board[i+7*5] == me_or_enemy \
            and observation.board[i+7*4] == me_or_enemy \
            and observation.board[i+7*3] == me_or_enemy \
            and observation.board[i+7*2] == 0:
                return i
            elif observation.board[i+7*4] == me_or_enemy \
            and observation.board[i+7*3] == me_or_enemy \
            and observation.board[i+7*2] == me_or_enemy \
            and observation.board[i+7*1] == 0:
                return i
            elif observation.board[i+7*3] == me_or_enemy \
            and observation.board[i+7*2] == me_or_enemy \
            and observation.board[i+7*1] == me_or_enemy \
            and observation.board[i+7*0] == 0:
                return i
        # no chance
        return -99
    
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_horizontal_chance(me_or_enemy):
        chance_cell_num = -99
        for i in [0,7,14,21,28,35]:
            for j in range(0, 4):
                val_1 = i+j+0
                val_2 = i+j+1
                val_3 = i+j+2
                val_4 = i+j+3
                if sum([observation.board[val_1] == me_or_enemy, \
                        observation.board[val_2] == me_or_enemy, \
                        observation.board[val_3] == me_or_enemy, \
                        observation.board[val_4] == me_or_enemy]) == 3:
                    for k in [val_1,val_2,val_3,val_4]:
                        if observation.board[k] == 0:
                            chance_cell_num = k
                            # bottom line
                            for l in range(35, 42):
                                if chance_cell_num == l:
                                    return l - 35
                            # others
                            if observation.board[chance_cell_num+7] != 0:
                                return chance_cell_num % 7
        # no chance
        return -99
    
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_slanting_chance(me_or_enemy, lag, cell_list):
        chance_cell_num = -99
        for i in cell_list:
            val_1 = i+lag*0
            val_2 = i+lag*1
            val_3 = i+lag*2
            val_4 = i+lag*3
            if sum([observation.board[val_1] == me_or_enemy, \
                    observation.board[val_2] == me_or_enemy, \
                    observation.board[val_3] == me_or_enemy, \
                    observation.board[val_4] == me_or_enemy]) == 3:
                for j in [val_1,val_2,val_3,val_4]:
                    if observation.board[j] == 0:
                        chance_cell_num = j
                        # bottom line
                        for k in range(35, 42):
                            if chance_cell_num == k:
                                return k - 35
                        # others
                        if chance_cell_num != -99 \
                        and observation.board[chance_cell_num+7] != 0:
                            return chance_cell_num % 7
        # no chance
        return -99
    
    def check_horizontal_first_enemy_chance():
        # enemy's chance
        if observation.board[38] == enemy_num:
            if sum([observation.board[39] == enemy_num, observation.board[40] == enemy_num]) == 1 \
            and observation.board[37] == 0:
                for i in range(39, 41):
                    if observation.board[i] == 0:
                        return i - 35
            if sum([observation.board[36] == enemy_num, observation.board[37] == enemy_num]) == 1 \
            and observation.board[39] == 0:
                for i in range(36, 38):
                    if observation.board[i] == 0:
                        return i - 35
        # no chance
        return -99

    def check_first_or_second():
        count = 0
        for i in observation.board:
            if i != 0:
                count += 1
        # first
        if count % 2 != 1:
            my_num = 1
            enemy_num = 2
        # second
        else:
            my_num = 2
            enemy_num = 1
        return my_num, enemy_num
    
    # check first or second
    my_num, enemy_num = check_first_or_second()
    
    def check_my_chances():
        # check my virtical chance
        result = check_vertical_chance(my_num)
        if result != -99:
            return result
        # check my horizontal chance
        result = check_horizontal_chance(my_num)
        if result != -99:
            return result
        # check my slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(my_num, 6, [3,4,5,6,10,11,12,13,17,18,19,20])
        if result != -99:
            return result
        # check my slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(my_num, 8, [0,1,2,3,7,8,9,10,14,15,16,17])
        if result != -99:
            return result
        # no chance
        return -99
    
    def check_enemy_chances():
        # check horizontal first chance
        result = check_horizontal_first_enemy_chance()
        if result != -99:
            return result
        # check enemy's vertical chance
        result = check_vertical_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's horizontal chance
        result = check_horizontal_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(enemy_num, 6, [3,4,5,6,10,11,12,13,17,18,19,20])
        if result != -99:
            return result
        # check enemy's slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(enemy_num, 8, [0,1,2,3,7,8,9,10,14,15,16,17])
        if result != -99:
            return result
        # no chance
        return -99
    
    if my_num == 1:
        result = check_my_chances()
        if result != -99:
            return result
        result = check_enemy_chances()
        if result != -99:
            return result
    if my_num == 2:
        result = check_enemy_chances()
        if result != -99:
            return result
        result = check_my_chances()
        if result != -99:
            return result
    
    # select center as priority (3 > 2 > 4 > 1 > 5 > 0 > 6)
    # column 3
    if observation.board[24] != enemy_num \
    and observation.board[17] != enemy_num \
    and observation.board[10] != enemy_num \
    and observation.board[3] == 0:
        return 3
    # column 2
    elif observation.board[23] != enemy_num \
    and observation.board[16] != enemy_num \
    and observation.board[9] != enemy_num \
    and observation.board[2] == 0:
        return 2
    # column 4
    elif observation.board[25] != enemy_num \
    and observation.board[18] != enemy_num \
    and observation.board[11] != enemy_num \
    and observation.board[4] == 0:
        return 4
    # column 1
    elif observation.board[22] != enemy_num \
    and observation.board[15] != enemy_num \
    and observation.board[8] != enemy_num \
    and observation.board[1] == 0:
        return 1
    # column 5
    elif observation.board[26] != enemy_num \
    and observation.board[19] != enemy_num \
    and observation.board[12] != enemy_num \
    and observation.board[5] == 0:
        return 5
    # column 0
    elif observation.board[21] != enemy_num \
    and observation.board[14] != enemy_num \
    and observation.board[7] != enemy_num \
    and observation.board[0] == 0:
        return 0
    # column 6
    elif observation.board[27] != enemy_num \
    and observation.board[20] != enemy_num \
    and observation.board[13] != enemy_num \
    and observation.board[6] == 0:
        return 6
    # random
    else:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

agents = {"random": random_agent, "negamax": negamax_agent, "greedy" : greedy_agent, "rules" : rule_agent}


def interpreter(state, env):
    columns = env.configuration.columns
    rows = env.configuration.rows

    # Ensure the board is properly initialized.
    board = state[0].observation.board
    if len(board) != (rows * columns):
        board = [EMPTY] * (rows * columns)
        state[0].observation.board = board

    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Active agent action.
    column = active.action

    # Invalid column, agent loses.
    if column < 0 or active.action >= columns or board[column] != EMPTY:
        active.status = f"Invalid column: {column}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    play(board, column, active.observation.mark, env.configuration)

    # Check for a win.
    if is_win(board, column, active.observation.mark, env.configuration):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = -1
        inactive.status = "DONE"
        return state

    # Check for a tie.
    if all(mark != EMPTY for mark in board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    columns = env.configuration.columns
    rows = env.configuration.rows
    board = state[0].observation.board

    def print_row(values, delim="|"):
        return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

    row_bar = "+" + "+".join(["---"] * columns) + "+\n"
    out = row_bar
    for r in range(rows):
        out = out + \
            print_row(board[r * columns: r * columns + columns]) + row_bar

    return out


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "connectx.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(dirpath, "connectx.js"))
    with open(jspath) as f:
        return f.read()
