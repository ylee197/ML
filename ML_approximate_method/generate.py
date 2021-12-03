import numpy as np
from random import randrange
from numpy.random import shuffle

class RecursiveBacktracker():
    def __init__(self, w, h):
        self.height = h
        self.width = w
        self.grid_height = (2 * self.height) + 1
        self.grid_width = (2 * self.width) + 1

    def peek_neighbours(self, row, col, grid, is_wall=False):        
        neighbourList = []
        if row > 1 and grid[row - 2][col] == is_wall:
            neighbourList.append((row - 2, col))
        if row < self.grid_height - 2 and grid[row + 2][col] == is_wall:
            neighbourList.append((row + 2, col))
        if col > 1 and grid[row][col - 2] == is_wall:
            neighbourList.append((row, col - 2))
        if col < self.grid_width - 2 and grid[row][col + 2] == is_wall:
            neighbourList.append((row, col + 2))  

        shuffle(neighbourList)     
        return neighbourList

    def generateMaze(self):
        grid = np.empty((self.grid_height, self.grid_width), dtype=np.int8)
        grid.fill(1)

        rand_row = randrange(1, self.grid_height, 2)
        rand_col = randrange(1, self.grid_width, 2)
        track = [(rand_row, rand_col)]
        grid[rand_row][rand_col] = 0

        while track:
            (rand_row, rand_col) = track[-1]
            neighbors = self.peek_neighbours(rand_row, rand_col, grid, True)

            if len(neighbors) == 0:
                track = track[:-1]
            else:
                nrow, ncol = neighbors[0]
                grid[nrow][ncol] = 0
                grid[(nrow + rand_row) // 2][(ncol + rand_col) // 2] = 0

                track += [(nrow, ncol)]

        return grid
    def visualise(self, grid):
        return_txt = []
        for row in grid:
            return_txt.append(''.join(['%' if cell else ' ' for cell in row]))
        return '\n'.join(return_txt)

if __name__ == "__main__":
    generator = RecursiveBacktracker(5,5)
    maze = generator.generateMaze()
    print(generator.visualise(maze))

