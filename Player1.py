# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 06:10:24 2019

@author: WallyKelm
"""

import DiceGame
import numpy as np
import copy

class Player1(DiceGame.BasePlayer):
    def __init__(self):
        DiceGame.BasePlayer.__init__(self)
        self.name = "Player1: Weight Outside Higher"
        
    def choose_option(self, gb, options):
        pass

    def choose_option_if_completes_col(self, gb, options):
        num = len(options)
        completes_col = np.zeros(num)
        for i in range(num):
            gb_try = copy(gb)
            token = gb_try.make_player_move(options[i])
            if len(token) > 0:
                for i in range(2):
                    col = gb_try.active_token_column[token[i]]
                    ht = gb_try.active_token_height[token[i]]
                    if ht == gb.heights[col-2]:
                        completes_col[i] = 1 + 1./gb.heights[col-2]
    
    def choose_option_by_added_value(self, gb, options):
        num = len(options)
        wt = np.zeros(num)
        for i in range(num):
            
            for opt in options[i]:
                wt[i] += 1./gb.heights[opt-2]
            #print(f"{i} {options[i]} {wt[i]}")
        choice = self.random_weighted_choose(wt)
        #print(f"{wt} {choice}")
        return choice
        
if __name__=="__main__":
    gb = DiceGame.GameBoard()
    p1 = Player1()
    
    p1.choose_option(gb,[[7],[2,3],[12,12]])
    
            