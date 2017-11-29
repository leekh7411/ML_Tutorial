import numpy as np
import random
class TicTacToeSingle:
    def __init__(self):
        self.Turn = 1
        self.PlayerA = 1
        self.PlayerB = -1
        self.Winner = 0
        self.GameOver = False
        self.Draw = 0
        self.Board = np.zeros(9)
        self.WinCase = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        self.ActionBoard = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]

    def __reset__(self):
        self.Turn = 1
        self.PlayerA = 1
        self.PlayerB = -1
        self.Winner = 0
        self.GameOver = False
        self.Board = np.zeros(9)
    def __checkWinner(self):
        for i in range(self.WinCase.__len__()):
            count = 0
            for j in range(self.WinCase[i].__len__()):
                if self.Board[self.WinCase[i][j]] == 1:
                    count += 1
            if count == 3 :
                if self.Turn == self.PlayerA:
                    print("--------- PlayerA Win! ---------")
                elif self.Turn == self.PlayerB:
                    print("--------- PlayerB Win! ---------")
                return True

        return False

    def __checkBoard__(self,action):
        if self.Board[action-1] == 0:
            return True
        else:
            return False

    def __changeBoardTurn__(self):
        for i in range(self.Board.__len__()):
            self.Board[i] *= -1

    def __printBoard__(self):
        _cnt = 0
        if self.Turn == self.PlayerA:
            print("> Player A : O")
        elif self.Turn == self.PlayerB:
            print("> Player B : O")

        for i in range(self.Board.__len__()):
            _end = ' '
            _stone = ''
            if _cnt == 2:
                _end = '\n'
                _cnt = 0
            else:
                _cnt += 1
                _end = ' '

            if self.Board[i] == 1:_stone = 'O'
            elif self.Board[i] == -1:_stone = 'X'
            else:_stone = '-'

            print(_stone,end =_end)

    def __step__(self,action):
        # Turn Change
        self.Turn = self.Turn * -1
        self.__changeBoardTurn__()

        # Set on the board
        if self.__checkBoard__(action):
            self.Board[action-1] = 1
        else:
            return False

        # Check Winner
        if self.__checkWinner():
            self.Winner = self.Turn
            self.GameOver = True

        # print board
        self.__printBoard__()

        return True


if __name__ == "__main__":
    my_game = TicTacToeSingle()
    while not my_game.GameOver:
        action = input("1~9: ")
        action = int(action)
        my_game.__step__(action)


