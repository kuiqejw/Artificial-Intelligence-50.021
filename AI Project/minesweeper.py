import sys
sys.setrecursionlimit(1000000)
import random
import numpy as np

class Minesweeper(object):
    def __init__(self):
        super(Minesweeper, self).__init__()
        self.status = 0

        self.width = 8
        self.height = 8
        self.n_mines =  5 #10 was too hard for the DQN to compute
        self.uncleared_blocks = self.width * self.height - self.n_mines

        self.map = np.zeros((self.height, self.width)) 
        self.mines = np.zeros((self.height, self.width))
        self.mask = np.zeros((self.height, self.width)) 
        

        for index in random.sample(range(1, self.width * self.height), self.n_mines): 
            self.map[index // self.width][index % self.width] = 1

        for i in range(self.height):
            for j in range(self.width):
                self.mines[i][j] = self.get_mine_num(i, j)
                if self.map[i][j] == 1:
                    self.mines[i][j] = 9

    def action(self, a): #action is always a number from 0 to 63
        x = a // self.width
        y = a % self.width
        try:
            self.mask[x][y] = self.mines[x][y] #assign mines to mask to check if there was a omb
        except IndexError:
            print(x, y)

        if self.map[x][y] == 1:#basically bombed up
            self.status = -1
        else: #no bombs
            if self.mines[x][y] == -1: #since there is no bomb, and no number attached (close to a bomb) then clear the surrounding blocks
                self.clear_empty_blocks(x, y)
            self.uncleared_blocks = self.width * self.height - self.n_mines - (self.mask != 0).sum()
            if self.uncleared_blocks == 0: #all blocks have been uncovered
                self.status = 1

    def clear_empty_blocks(self, i, j): #recursion function to clear the surrounding blocks around it
        self.mask[i][j] = self.mines[i][j]
        if self.mines[i][j] != -1:
            return
        else:
            neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                          (i, j-1), (i, j+1),
                          (i+1, j-1), (i+1, j), (i+1, j+1)]
            for n in neighbours:
                if not self.is_in_range(n[0], n[1]) or self.mask[n[0]][n[1]] != 0:
                    continue
                self.clear_empty_blocks(n[0], n[1])#recurse to check the other blocks

    def get_score(self): #the reward
        sum = 0
        for i in self.mask:
            for j in i:
                if j != -1:
                    sum += 1
        return sum

    def get_state(self):
        return self.mask.copy()
        # return self.mask

    def get_status(self):
        return self.status

    def show(self): #to be removed
        print('======MAP======')
        for i in range(self.height):
          print(self.map[i])
        print('======MASK======')
        for i in range(self.height):
            print(self.mask[i])
        print('======MINE======')
        for i in range(self.height):
           print(self.mines[i])

    def get_mine_num(self, i, j):
        neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1), (i, j),   (i, j+1),
                      (i+1, j-1), (i+1, j), (i+1, j+1)]
        # mine_num = 0
        mine_num = sum(1 for (row_id, col_id) in neighbours if self.is_in_range(row_id, col_id) and self.map[row_id][col_id] == 1)
        if mine_num == 0:
            mine_num = -1
        return mine_num
    def is_in_range(self, row_id, col_id):
        return 0 <= row_id < self.width and 0 <= col_id < self.height

# game = Minesweeper()
# print(game.get_state())#nested 9 by 9 array
# game.show()
# while (game.get_status() == 0):
#    a = input('input a: ')
#    game.action(int(a))
#    game.show()