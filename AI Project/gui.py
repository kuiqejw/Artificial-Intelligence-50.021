from tkinter import *
from tkinter import messagebox
import random
import numpy as np
import argparse
from miner import Miner

class GUI:

    def __init__(self, master):

        # import images
        self.tile_plain = PhotoImage(file = "images/tile_plain.gif")
        self.tile_clicked = PhotoImage(file = "images/tile_clicked.gif")
        self.tile_mine = PhotoImage(file = "images/tile_mine.gif")
        self.tile_wrong = PhotoImage(file = "images/tile_wrong.gif")
        self.tile_no = []
        for x in range(1, 9):
            self.tile_no.append(PhotoImage(file = "images/tile_"+str(x)+".gif"))

        # set up frame
        frame = Frame(master)
        frame.pack()

        # show "Minesweeper" at the top
        self.label1 = Label(frame, text="Minesweeper")
        self.label1.grid(row = 0, column = 0, columnspan = 10)

        # create flag and clicked tile variables
        self.clicked = 0
        self.width = 8
        self.status = 0
        self.height = 8
        self.n_mines = 5
        self.uncleared_blocks = self.width*self.height - self.n_mines

        self.map = np.zeros((self.height, self.width)) 
        self.mines = np.zeros((self.height, self.width))
        self.mask = np.zeros((self.height, self.width))

        #randomly assign mines
        for index in random.sample(range(1, self.width * self.height), self.n_mines): 
            self.map[index // self.width][index % self.width] = 1
        # create buttons

        for i in range(self.height):
            for j in range(self.width):
                self.mines[i][j] = int(self.get_mine_num(i, j))
                if self.map[i][j] == 1:
                    self.mines[i][j] = 9

        self.buttons = dict({})
        self.n_mines = 0
        x_coord = 0
        y_coord = 0
        for i in range(0, 64):
            mine = 0
            # tile image changeable for debug reasons:
            gfx = self.tile_plain
            # currently random amount of mines
            # if random.uniform(0.0, 1.0) < 0.1:
            #     mine = 1
            #     self.mines += 1
            # 0 = Button widget
            # 1 = if a mine y/n (1/0)
            # 2 = state (0 = unclicked, 1 = clicked, 2 = flagged)
            # 3 = button id
            # 4 = [x, y] coordinates in the grid
            # 5 = nearby mines, 0 by default, calculated after placement in grid
            self.buttons[i] = [ Button(frame, image = gfx),
                                self.map[x_coord][y_coord], #mine yes no
                                0,
                                i,
                                [x_coord, y_coord],
                                self.mines[x_coord][y_coord]] #no of mines
            self.buttons[i][0].bind('<Button-1>', self.lclicked_wrapper(i))

            # calculate coords:
            y_coord += 1
            if y_coord == 8:
                y_coord = 0
                x_coord += 1
        
        # lay buttons in grid
        for key in self.buttons:
            self.buttons[key][0].grid( row = self.buttons[key][4][0], column = self.buttons[key][4][1] )

        # find nearby mines and display number on tile
        for key in self.buttons:
            self.buttons[key][5] = int(self.mines[key//8][key% 8])
            if self.mines[key//8][key% 8] < 0:
                self.buttons[key][5] = 0

        #add mine and count at the end
        self.label2 = Label(frame, text = "Mines: "+str(self.mines))
        self.show()
    ## End of __init__
    def get_mine_num(self, i, j):
        neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1), (i, j),   (i, j+1),
                      (i+1, j-1), (i+1, j), (i+1, j+1)]
        # mine_num = 0
        mine_num = sum(1 for (row_id, col_id) in neighbours if self.is_in_range(row_id, col_id) and self.map[row_id][col_id] == 1)
        if mine_num == 0:
            mine_num = -1
        return mine_num
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


    def is_in_range(self, row_id, col_id):
        return 0 <= row_id < self.width and 0 <= col_id < self.height
    def check_for_mines(self, key):
        try:
            if self.buttons[key][1] == 1:
                return True
        except KeyError:
            pass

    def lclicked_wrapper(self, x):
        return lambda Button: self.lclicked(x)

    def lclicked(self, a): #action is a number from 0 to 63
        button_data = self.buttons[a]
        x = a // self.width
        y = a % self.width
        try:
            self.mask[x][y] = self.mines[x][y]
        except IndexError:
            print(x,y)
        if button_data[1] == 1: #if a mine
            # show all mines and check for flags
            for key in self.buttons:
                if self.buttons[key][1] != 1 and self.buttons[key][2] == 2:
                    self.buttons[key][0].config(image = self.tile_wrong)
                if self.buttons[key][1] == 1 and self.buttons[key][2] != 2:
                    self.buttons[key][0].config(image = self.tile_mine)
            # end game
            self.status = -1
            self.gameover()
        else:
            #no bombs
            if button_data[5] == 0 and self.mines[x][y] == -1:
                button_data[0].config(image = self.tile_clicked)
                self.clear_empty_blocks(button_data[3])
            else:
                button_data[0].config(image = self.tile_no[button_data[5]-1])
            # if not already set as clicked, change state and count
            if button_data[2] != 1:
                button_data[2] = 1
                self.clicked += 1
            self.uncleared_blocks = self.width * self.height - self.n_mines - (self.mask != 0).sum()
            print(self.clicked, self.uncleared_blocks)
            if self.clicked == 58:
                self.status = 1
                self.victory()


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
    def check_tile(self, key):
        try:
            if self.buttons[key][2] == 0:
                if self.buttons[key][5] == 0:
                    self.buttons[key][0].config(image = self.tile_clicked)
                else:
                    self.buttons[key][0].config(image = self.tile_no[self.buttons[key][5]-1])
                self.buttons[key][2] = 1
                self.clicked += 1
        except KeyError:
            pass


    def clear_empty_blocks(self,key): #recursion function to clear the surrounding blocks around it
        i = key // 8
        j = key % 8
        self.check_tile(key)
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
                self.clear_empty_blocks(n[0] * 8 + n[1])#recurse to check the other blocks
    # def clear_empty_tiles(self, main_key):
    #     queue = deque([main_key])

    #     while len(queue) != 0:
    #         key = queue.popleft()
    #         self.check_tile(key-9, queue)      #top right
    #         self.check_tile(key-10, queue)     #top middle
    #         self.check_tile(key-11, queue)     #top left
    #         self.check_tile(key-1, queue)      #left
    #         self.check_tile(key+1, queue)      #right
    #         self.check_tile(key+9, queue)      #bottom right
    #         self.check_tile(key+10, queue)     #bottom middle
    #         self.check_tile(key+11, queue)     #bottom left
    
    def gameover(self):
        messagebox.showinfo("Game Over", "You Lose!")
        print(self.clicked)
        global root
        root.destroy()

    def victory(self):
        messagebox.showinfo("Game Over", "You Win!")
        global root
        root.destroy()

### END OF CLASSES ###

def main():
    global root
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.9, help='the probability to choose from memories')
    parser.add_argument('--memory_capacity', type=int, default=50000, help='the capacity of memories')
    parser.add_argument('--target_replace_iter', type=int, default=100, help='the iter to update the target net')
    parser.add_argument('--batch_size', type=int, default=16, help='sample amount')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=20000, help='training epoch number')
    parser.add_argument('--n_critic', type=int, default=100, help='evaluation point')
    parser.add_argument('--test', type=int, default=0, help='whether execute test')
    parser.add_argument('--conv', type=int, default=0, help = 'choose between linear and convolution')
    opt = parser.parse_args()
    print(opt)

    miner = Miner(opt.epsilon, opt.memory_capacity, opt.target_replace_iter, opt.batch_size, opt.lr, opt.conv)
    miner.load_params('eval_rms.pth')
    global root
    # create Tk widget
    root = Tk()
    # set program title
    root.title("Minesweeper")
    # create game instance
    game = GUI(root)
    old_a = None
    def sub_func():
        print('pray tell')
        s = game.get_state()
        a = miner.choose_action(s)
        game.lclicked(a)
        old_a = a
        root.after(1000, sub_func)
    # run event loop
    root.after(1000, sub_func)
    root.mainloop()

if __name__ == "__main__":
    main()
