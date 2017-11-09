import numpy as np

class TicTacToe:
    def __init__(self):
        self._width = 3
        self._height = 3
        self._A = 1 # turn True
        self._B = 2 # turn False
        self._score_A = 0
        self._score_B = 0
        self._board = np.zeros((self._width,self._height))
        self._VERTICAL = 0
        self._HORIZONTAL = 1
        self._DIAGONAL_UD = 2
        self._DIAGONAL_DU = 3
        self._turn = True
        self._GAME_OVER = False

    def _start_game(self):
        while not self._GAME_OVER:
            print("======================================================")
            self._print_board()
            if self._turn:
                print("> Game Turn : Player A")
            else:
                print("> Game Turn : Player B")

            # prompt input case..
            wh = input("next location : ")
            wh = int(wh)
            _h = wh % 10
            _w = int((wh - _h)/10)

            _w = _w - 1
            _h = _h - 1
            print(">>> _w : ", _w, " / _h : ", _h)


            result = True
            if self._turn:
                result = self._game_update(self._A,_w,_h)
            else:
                result = self._game_update(self._B,_w,_h)

            if result:
                self._turn = not self._turn


        print("======================================================")
        self._print_board()
        if self._turn:
            print("> Winner : Player B")
        else:
            print("> Winner : Player A")


    def _print_board(self):
        print("> Game board")
        print(self._board)
        location_board = np.zeros((3,3))
        for h in range(3):
            for w in range(3):
                location_board[w,h] = (w+1)*10+(h+1)
        print("> location board")
        print(location_board)


    def _check_update(self,_w,_h):
        if self._board[_w,_h] == 0 :
            return True
        else :
            return False


    def _game_update(self,player,_w,_h):
        if self._check_update(_w,_h):
            self._board[_w,_h] = player
            self._check_winner(player, _w, _h, 0)
            return True
        else:
            return False


    def _check_rule(self,player,_w,_h,_score,_DIR,_DIR_V):
        if _DIR == self._VERTICAL:
            _h += _DIR_V

        if _DIR == self._HORIZONTAL:
            _w += _DIR_V

        if _DIR == self._DIAGONAL_DU:
            _h += _DIR_V
            _w += _DIR_V

        if _DIR == self._DIAGONAL_UD:
            _h -= _DIR_V
            _w += _DIR_V

        if _w >= 3 or _h >= 3 or _w < 0 or _h < 0 :
            print(">return out of index! score :" , _score)
            return _score

        if self._board[_w, _h] == player:
            print("> Found..(",_w,",",_h,")")
            _score += 1
            _score = self._check_rule(player, _w, _h, _score,_DIR ,_DIR_V)
        else:
            print("> Not found..(",_w,",",_h,"),player: ",player,",map : " ,self._board[_w,_h])
            return _score

        return _score

    def _check_winner(self,player,_w,_h,_score):
        win_score = 2
        start_score = 0

        # check vertical
        score = start_score # start point -> [+1 point]
        print("> check vertical ")
        scoreA = self._check_rule(player,_w,_h,score,self._VERTICAL, 1)
        print("> score : ", score)
        scoreB = self._check_rule(player,_w,_h,score,self._VERTICAL,-1)
        score = scoreA + scoreB
        print("> score : ", score)
        if score >= win_score :
            print("Winner : Player " , player)
            self._GAME_OVER = True

        # check horizontal
        print("> check horizontal")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._HORIZONTAL,  1)
        print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._HORIZONTAL, -1)
        score = scoreA + scoreB
        print("> score : ", score)
        if score >= win_score:
            print("Winner : Player ", player)
            self._GAME_OVER = True

        # check diagonal up -> down
        print("> check diagonal up -> down")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._DIAGONAL_UD,  1)
        print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._DIAGONAL_UD, -1)
        score = scoreA + scoreB
        print("> score : ", score)
        if score >= win_score:
            print("Winner : Player ", player)
            self._GAME_OVER = True

        # check diagonal down -> up
        print("> check diagonal down -> up")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._DIAGONAL_DU,  1)
        print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._DIAGONAL_DU, -1)
        score = scoreA + scoreB
        print("> score : ", score)
        if score >= win_score:
            print("Winner : Player ", player)
            self._GAME_OVER = True



my_game = TicTacToe()
my_game._start_game()

