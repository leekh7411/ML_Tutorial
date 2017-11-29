
class TicTacToe:
    def __init__(self):
        self._width = 3
        self._height = 3
        self._A =  1 # turn True
        self._B = -1 # turn False
        self._score_A = 0
        self._score_B = 0
        self._board = np.zeros((self._width,self._height))
        self._VERTICAL = 0
        self._HORIZONTAL = 1
        self._DIAGONAL_UD = 2
        self._DIAGONAL_DU = 3
        self._turn = True
        self._GAME_OVER = False
        self._board_count = self._board.size
        self._IsDraw = False
        self._IsPrint = True
        self._location_board = self._set_location_board()
        self._location_table = [11,12,13,21,22,23,31,32,33]

    def _reset(self):
        self._score_A = 0
        self._score_B = 0
        self._board = np.zeros((self._width, self._height))
        self._turn = True
        self._GAME_OVER = False
        self._board_count = self._board.size
        self._IsDraw = False
        return self._board

    def _get_action_location(self,action):
        if action >= 11 and action <= 33:
            _h = action % 10
            _w = int((action - _h) / 10)
            _w = _w - 1
            _h = _h - 1
            return(_h*3+_w)
        else:
            return -1

    def _get_flat_board(self,state):
        flat_board = np.zeros(9)
        idx = 0
        for _w in range(3):
            for _h in range(3):
                flat_board[idx]= state[_w,_h]
                idx += 1
        return flat_board

    def _get_location(self,action):
        return self._location_table[action]

    def _get_sample_action(self):
        rand_list = []
        for _w in range(3):
            for _h in range(3):
                if self._board[_w,_h] == 0 :
                    rand_list.append(self._location_board[_w,_h])

        if len(rand_list) > 0 :
            return random.choice(rand_list)
        else :
            return 0 # No more space

    def _get_GameOver(self):
        return self._GAME_OVER

    def _get_PlayerTurn(self):
        return self._turn

    def _get_observation_space(self):
        return self._board_count

    def _get_action_spcae(self):
        return self._board_count

    def _get_A(self):
        return self._A

    def _get_B(self):
        return self._B

    def _is_draw(self):
        return self._IsDraw

    def _step(self,action):
        _h = action % 10
        _w = int((action - _h) / 10)

        _w = _w - 1
        _h = _h - 1

        result = True
        if self._turn:
            result = self._game_update(self._A, _w, _h)
        else:
            result = self._game_update(self._B, _w, _h)

        if result:
            self._turn = not self._turn

        if self._board_count == 0 and (not self._GAME_OVER):
            self._GAME_OVER = True
            self._IsDraw = True

        next_player = self._turn
        done = self._GAME_OVER
        reward = 0
        if done :
            if self._IsDraw:
                reward = 0
            elif not self._turn: # End game at turn B --> A win
                reward = self._A
            else: # End game at turn A --> B win
                reward = self._B

        #self._print_board()

        return self._board,self._turn,reward, done

    def _start_game(self):
        while not self._GAME_OVER:
            input_msg = ''
            if self._IsPrint :
                print("======================================================")
                print("Board Size : ", self._board_count)
                self._print_board()

                if self._turn:
                    print("> Game Turn : Player A")
                else:
                    print("> Game Turn : Player B")
                input_msg = "next_location : "

            # prompt input case..


            wh = input(input_msg)
            wh = int(wh)
            _h = wh % 10
            _w = int((wh - _h)/10)

            _w = _w - 1
            _h = _h - 1
            if self._IsPrint:
                print(">>> _w : ", _w, " / _h : ", _h)


            result = True
            if self._turn:
                result = self._game_update(self._A,_w,_h)
            else:
                result = self._game_update(self._B,_w,_h)

            if result:
                self._turn = not self._turn

            if self._board_count == 0 and (not self._GAME_OVER):
                self._GAME_OVER = True
                self._IsDraw = True

        if self._IsPrint:
            print("======================================================")
            self._print_board()

            if self._IsDraw:
                print("> DRAW GAME")
            elif self._turn:
                print("> Winner : Player B")
            else:
                print("> Winner : Player A")


    def _set_location_board(self):
        location_board = np.zeros((3, 3))
        for h in range(3):
            for w in range(3):
                location_board[w, h] = (w + 1) * 10 + (h + 1)
        return location_board


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

    def _action_to_wh(self,action):
        _h = action % 10
        _w = int((action - _h) / 10)

        _w = _w - 1
        _h = _h - 1
        return _w,_h

    def _check_update_action(self,action):
        _w,_h = self._action_to_wh(action)
        return self._check_update(_w,_h)

    def _game_update(self,player,_w,_h):
        if self._check_update(_w,_h):
            self._board[_w,_h] = player
            self._board_count -= 1
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
            if self._IsPrint:print(">return out of index! score :" , _score)
            return _score

        if self._board[_w, _h] == player:
            if self._IsPrint:print("> Found..(",_w,",",_h,")")
            _score += 1
            _score = self._check_rule(player, _w, _h, _score,_DIR ,_DIR_V)
        else:
            if self._IsPrint:print("> Not found..(",_w,",",_h,"),player: ",player,",map : " ,self._board[_w,_h])
            return _score

        return _score

    def _check_winner(self,player,_w,_h,_score):
        win_score = 2
        start_score = 0

        # check vertical
        score = start_score # start point -> [+1 point]
        if self._IsPrint:print("> check vertical ")
        scoreA = self._check_rule(player,_w,_h,score,self._VERTICAL, 1)
        if self._IsPrint:print("> score : ", score)
        scoreB = self._check_rule(player,_w,_h,score,self._VERTICAL,-1)
        score = scoreA + scoreB
        if self._IsPrint:print("> score : ", score)
        if score >= win_score :
            if self._IsPrint:print("Winner : Player " , player)
            self._GAME_OVER = True

        # check horizontal
        if self._IsPrint:print("> check horizontal")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._HORIZONTAL,  1)
        if self._IsPrint:print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._HORIZONTAL, -1)
        score = scoreA + scoreB
        if self._IsPrint:print("> score : ", score)
        if score >= win_score:
            if self._IsPrint:print("Winner : Player ", player)
            self._GAME_OVER = True

        # check diagonal up -> down
        if self._IsPrint:print("> check diagonal up -> down")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._DIAGONAL_UD,  1)
        if self._IsPrint:print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._DIAGONAL_UD, -1)
        score = scoreA + scoreB
        if self._IsPrint:print("> score : ", score)
        if score >= win_score:
            if self._IsPrint:print("Winner : Player ", player)
            self._GAME_OVER = True

        # check diagonal down -> up
        if self._IsPrint:print("> check diagonal down -> up")
        score = start_score  # start point -> [+1 point]
        scoreA = self._check_rule( player, _w, _h,score, self._DIAGONAL_DU,  1)
        if self._IsPrint:print("> score : ", score)
        scoreB = self._check_rule( player, _w, _h,score, self._DIAGONAL_DU, -1)
        score = scoreA + scoreB
        if self._IsPrint:print("> score : ", score)
        if score >= win_score:
            if self._IsPrint:print("Winner : Player ", player)
            self._GAME_OVER = True