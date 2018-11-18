# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import random
import timeit
import pickle
import os

####################################################
#
#   Define class to store game state data
#
#
####################################################

class GameBoard():
    columns = range(2,13)
    heights = list(range(3,14,2)) + list(range(11,2,-2))
    version = 1

    def __init__(self, json_input = None):
        if json_input == None:
            #print("reset")
            self.reset()
        else:
            #print("try load")
            #self.reset()
            self.load(json_input)
        
            
            
    def reset(self, keep_players=False):
        self.player_heights = []
        if keep_players:
            for i, _ in enumerate(self.player_list):
                self.player_heights.append([0 for _ in enumerate(self.columns)])
        else:
            self.player_list = []
        self.completed_col  = [False for _ in enumerate(self.columns)]
        self.active_player = 0
        self.game_in_progress = False
        self.active_token_column = [-1,-1,-1] # use column labels 2,3,4,...12
        self.active_token_height = [0,0,0]
        self.completed = False
        self.winner = -1
        self.turn_count = 0



    def load(self, json_input):
        #print("json_input:"+json_input)
        input_state = json.loads(json_input)
        if "version" in input_state:
            if input_state["version"] == 1:
                self.loadV1(input_state)


    def loadV1(self, input_state):
        self.version = 1
        self.player_list = list(input_state["player_list"])
        self.player_heights = list(input_state["player_heights"])
        self.completed_col  = list(input_state["completed_col"])
        self.active_player = int(input_state["active_player"])
        self.game_in_progress = bool(input_state["game_in_progress"])
        self.active_token_column = list(input_state["active_token_column"])
        self.active_token_height = list(input_state["active_token_height"])
        self.completed = bool(input_state["completed"])
        self.winner = int(input_state["winner"])
        self.turn_count = int(input_state["turn_count"])
        
    def dump(self, method="json"):
        if method.lower() == "json":
            json_output = '{"version":'
            json_output += str(self.version)
            json_output += ',"player_list":['
            for p in self.player_list:
                json_output += str(p)
            json_output += ']'
            json_output += ',"player_heights":['
            for ph in self.player_heights:
                json_output += str(ph)
            json_output += ']'
            json_output += ',"completed_col":' + json.dumps(self.completed_col)
            json_output += ',"active_player":' + json.dumps(self.active_player)
            json_output += ',"game_in_progress":' + json.dumps(self.game_in_progress)
            json_output += ',"active_token_column":' + json.dumps(self.active_token_column)
            json_output += ',"active_token_height":' + json.dumps(self.active_token_height)
            json_output += ',"completed":' + json.dumps(self.completed)
            json_output += ',"winner":' + json.dumps(self.winner)
            json_output += ',"turn_count":' + json.dumps(self.turn_count)
            json_output += '}'
            #print("1"+json_output)
            #out=json.loads(json_output)
            #print(out)
            return json_output

    def get_player_height_str(self):
        return str(np.array(self.player_heights))

    def add_player(self, new_player):
        if (not self.game_in_progress) and ( len(self.player_list)<4):
            if new_player.is_cant_stop_player():
                self.player_list.append(new_player)
                return True
        return False

    def get_player_height(self, player_index=None, column=7):
        if player_index == None:
            player_index = self.active_player
        assert (column >=2 and column <= 12)
        return self.player_heights[player_index][column-2]

    def get_player_completed_column_sum(self, player_index=None):
        count = 0
        for col, ht in zip(self.columns, self.heights):
            if self.get_player_height(player_index, col) >= ht:
                count += 1
        return count

    def count_tokens_above_win(self):
        count = 0
        for t, h in zip(self.active_token_column, self.active_token_height):
            if t >=2 and t <= 12:
                if h >= self.heights[t-2]:
                    count +=1
        return count

    def count_free_tokens(self):
        num_free = 0
        for t in self.active_token_column:
            if t ==-1:
                num_free +=1
        return num_free

    def get_available_columns(self):
        available = []
        
        if self.count_free_tokens():
            for col, comp in zip(self.columns, self.completed_col):
                if not comp:
                    available.append(col)
        else:                
            for t in self.active_token_column:
                if not self.completed_col[t-2]:
                    available.append(t)
        return sorted(available)



####################################################
#
#   Define Game Rules
#
#
####################################################

class GameRules():
    
    def __init__(self, player_list=[]):
        self.gb = GameBoard()
        for p in player_list:
            self.gb.add_player(p)

    def add_player(self, new_player):
        return self.gb.add_player(new_player)

    def roll(self):
        """Roll 4 six sided die and calculate the 3 combinations of 2.  Compare
        to the available options and return a list of available options.
        """
        random_values = [random.randint(1,6) for _ in range(4)] # raw dice
        raw_options = [ [random_values[0]+random_values[1],
                     random_values[2]+random_values[3]],
                    [random_values[0]+random_values[2],
                     random_values[1]+random_values[3]],
                    [random_values[0]+random_values[3],
                     random_values[1]+random_values[2]]]
        available = self.gb.get_available_columns()
        available_options = []
        tokens = self.gb.active_token_column
        num_free = self.gb.count_free_tokens()
        
        for opt in raw_options: # check if combinations available
            opt = sorted(opt)
            both_available = (opt[0] in available) and (opt[1] in available)
            num_in_tokens =    int(opt[0] in tokens) + int(opt[1] in tokens)
                                    
            
            if (num_free + num_in_tokens >= 2) and ( both_available):
                    # both token and column available, score 2
                if opt not in available_options:
                    available_options.append(opt) # both die sums ok
            elif (opt[0] in available):
                if [opt[0]] not in available_options:
                    available_options.append([opt[0]]) # only first sum ok
            elif (opt[1] in available):
                if [opt[1]] not in available_options:
                    available_options.append([opt[1]]) # only second sum ok
        return random_values, sorted(available_options)


    def check_completed(self,player=None):
        player_iter = self.gb.player_heights
        if not( player == None):
            player_iter = [self.gb.player_heights[player]]
            # limit to one player if mid turn
        else:
            # Hard reset to check all players
            self.gb.completed_col = [False for _ in enumerate(self.gb.columns)]
            
        for ph in player_iter: # iterate over player heights
            for col_i, ph_col in enumerate(ph):
                if ph_col >= self.gb.heights[col_i]:
                    self.gb.completed_col[col_i] = True

    def check_mid_turn_completed_col(self):         
        for t, ht in zip( self.gb.active_token_column, self.gb.active_token_height):
            if t>=2 and t <=12:
                if ht >= self.gb.heights[t-2]:
                    self.gb.completed_col[t-2] = True
                
    def check_player_win(self,player=None):
        if self.gb.get_player_completed_column_sum(player) >=3:
            return True
        return False
        
        
    def next_player(self):
        # reset tokens
        self.gb.active_token_column = [-1,-1, -1]
        self.gb.active_token_height = [0,  0,  0]
        # next player
        self.gb.active_player = (self.gb.active_player + 1) % len(self.gb.player_list)
        if self.gb.active_player == 0:
            self.gb.turn_count +=1
        
    def player_bust(self):
        # Log?
        # No progress
#        print("BUST")
        self.next_player()

    def player_choose_option(self, options):
        """Ask the active player to pick from the options"""
        player_choice = int( self.gb.player_list[self.gb.active_player].choose_option(copy.copy(self.gb),options))
        #print(f"{options} {player_choice}")
        assert player_choice >=0 and player_choice < len(options)
        return player_choice

    def make_player_move(self, selected_option):
#        print("Before:" + str(self.gb.active_token_height))
        for opt in selected_option:
            for i in range(3): # check if active token and add if necessary
                active = self.gb.active_token_column[i]
#                print(f"{opt}, {active}")
                if opt == active: # Already an active token
                    self.gb.active_token_height[i] +=1
                    break
                elif active == -1: # add new token.  Must sort desc to keep -1's at end
                    self.gb.active_token_column[i] = opt
                    self.gb.active_token_column = sorted(self.gb.active_token_column, reverse=True)
                    self.gb.active_token_height[i] = 1 + self.gb.get_player_height(column=opt)
                    break
#        print("After:" + str(self.gb.active_token_height))
        self.check_mid_turn_completed_col()
            
            
    def player_choose_stop(self):
        """Ask the active player to continue or stop"""
        return bool( self.gb.player_list[self.gb.active_player].choose_stop(copy.copy(self.gb))) 
    
    def player_stop_save_progress(self):
        for token_col, token_ht in zip( self.gb.active_token_column, self.gb.active_token_height):
            if token_col>=2 and token_col <=12:
                # save progress to gameboard
                self.gb.player_heights[self.gb.active_player][token_col-2] =min(
                        (token_ht, self.gb.heights[token_col-2])) # limit to max ht
#        print("SAVE")
                
        
    
    def check_setup(self):
        if len( self.gb.player_list) > 0:
            return True # Stupid test... glad I checked
        else:
            return False
        
    def game_winner(self, log_type="STDOUT"):
        if log_type == "STDOUT":
            print(f"Game Winner is Player #{self.gb.active_player}")
            print(f"Congrats {self.gb.player_list[self.gb.active_player]}")
            
        # Log results?
    
    def start(self):
        if not self.check_setup():
            raise NameError("Invalid Setup")
        self.gb.reset(keep_players=True)
        self.gb.game_in_progress = True
        roll_count = 0
        while self.gb.game_in_progress:
            random_rolls, options = self.roll()
#            print(f"Turn: {self.gb.turn_count}  Player: {self.gb.active_player}  Roll: {roll_count}")
#            print(self.gb.get_player_height_str())
#            print(f"Token Columns: {self.gb.active_token_column}")
#            print(f"Token Heights: {self.gb.active_token_height}")
#            print(options)
            if len(options)==0:
                #print(self.gb.get_available_columns())
                self.player_bust()
            else:
                player_choice = self.player_choose_option(options)
#                print(player_choice)
                self.make_player_move(options[player_choice])
                
                if self.player_choose_stop():
                    # STOP
                    self.player_stop_save_progress()
                    
                    if self.check_player_win():
                        self.gb.winner = int( self.gb.active_player)
                        self.gb.game_in_progress = False
                        self.game_winner(log_type=None)
                        return self.gb.winner
                    else: # save progress but not a winner
                        self.next_player()
                        # start next loop
                    
                else:
                    # ROLL AGAIN
                    # next loop
                    pass
            roll_count +=1
            if roll_count > 250:
                return 0
                
####################################################
#
#   Base Player Class
#
#
####################################################            
        
class BasePlayer():
    def __init__(self):
        self.name="BasePlayer"
        self.sc = survive_calc() # calculate odds of surviving
        
    def __str__(self):
        return self.name

    def is_cant_stop_player(self):
        return True

    def choose_option(self, gameboard, options):
        """Return integer for selected option"""
        return random.randint(0,len(options)-1) #random choice

    def choose_stop(self, gameboard):
        """Return True to stop and False to roll again"""
        
        if gameboard.count_free_tokens() >0:
            return False # Free token so keep going
        elif gameboard.count_tokens_above_win():
            return True # any column finished so stop
        else:
            # 75% chance of rolling again.  3/(1+3)
            return bool(self.random_weighted_choose([3,1]))
        
    def random_weighted_choose(self, weights):
        """returns i with probability weights[i]/sum(weights)"""
        total = float(sum(weights))
        rnd   = total* random.random()
        for i,w in enumerate(weights):
            rnd -=w
            if rnd<=0:
                return i
#        print(rnd)
        return 0



class survive_calc():
    
    prob_index = [{} for _ in [0,1,2,3,4,5,6,7,8]] # dict for each len
    
    def __init__(self):
         if os.path.isfile("prob_index.pkl"):
             self.prob_index = pickle.load(open("prob_index.pkl","rb"))
         else:
             #calc combos
             for a in range(2,13): # go through 2-12
                 available = [a] # sets of single value
                 self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)                 
                 for b in range(a+1,13):
                     available = [a,b] # sets of 2
                     self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                     for c in range(b+1,13): # sets of 3
                         available = [a,b,c]
                         self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                         for d in range(c+1,13):
                             available = [a,b,c,d]
                             self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)                             
                             for e in range(d+1,13):
                                 available = [a,b,c,d,e]
                                 self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                                 for f in range(e+1,13):
                                     available = [a,b,c,d,e,f]
                                     self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                                     for g in range(f+1,13):
                                         available = [a,b,c,d,e,f,g]
                                         self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                                         for h in range(g+1,13):
                                             available = [a,b,c,d,e,f,g,h]
                                             self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
#                                             for i in range(h+1,13):
#                                                 available = [a,b,c,d,e,f,g,h,i]
#                                                 self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
#                                                 for j in range(i+1,13):
#                                                     available = [a,b,c,d,e,f,g,h,i,j]
#                                                     self.prob_index[len(available)][frozenset(available)] = self.calc_survive_probability(available)
                                                     
             pickle.dump(self.prob_index, open("prob_index.pkl","wb"))
             print("Building Probability Index")
             print("#Open   #Combos   Avg Prob")
             for i in range(1,9):
                 print("{0:5d} {1:9d} {2:8.2f}".format(i, len(self.prob_index[i].keys()), np.mean(list(self.prob_index[i].values()))))
         
    
    def survive(self, gameboard):
        available = gameboard.get_available_columns()
        available_set = frozenset(available)
        num = len(available_set)
        if num >= 9:
            return 1.
        elif num == 0:
            return 0.
        else:
            if available_set in self.prob_index[num].keys():
                return self.prob_index[num][available_set]
            else:
                prob = self.calc_survive_probability(available)
                self.prob_index[num][available_set] = prob
                return prob
                
        
    def calc_survive_probability(self, available):
        num_survive = 0
#        print("Available:"+str(available))
        if len(available)>0:
            for a in range(1, 7):
                for b in range(1, 7):
                    for c in range(1, 7):
                        for d in range(1, 7):  # roll 4 dice
                            options = [[a + b, c + d], [a + c, b + d], [a + d, b + c]]
                            survive_this_time = False
                            for opt in options:
                                if (opt[0] in available) or (opt[1] in available):
                                    survive_this_time = True
                                    break
                            num_survive += int(survive_this_time)
        return num_survive/1296.    

####################################################
#
#   TESTS
#
#
####################################################
def test_die_rolls(num):
    gr = GameRules()
    die_counts = np.zeros(6,dtype=np.int)
    counts = np.zeros(11)
    total = int(num)
    for _ in range(total):
        roll, options = gr.roll()
        for r in roll:
           die_counts[r-1]+=1

        for val in range(2,13):
            check=False
            for opt in options:
                if val in opt:
                    check=True
                    break
            if check:
                counts[val-2] += 1
    print("7 totals: "+str(counts[7-2]) + " "+ str(total))
    for  pct, num in zip(counts/total,range(2,13)):
        print(f"{num}: {pct}")
    print(die_counts)        

def test_roll_available_options():
    gr = GameRules()
    print(gr.gb.get_available_columns())
    for i in range(3):
        gr.gb.active_token_column[i]=i+2
        #gr.gb.completed_col[i]=True
        print(gr.gb.get_available_columns())
        for j in range(6):
            print(gr.roll())

def test_base_player_random_weighted_choose():
    bp = BasePlayer()
    choices = [bp.random_weighted_choose([1,3.]) for _ in range(1000)]
    ch_sum = sum(choices)
    print(ch_sum/1000.)
 
def test_survive_calc():
    gb = GameBoard()    
    sc = survive_calc()
    count = 0
    print("Combinations of 3 Columns:")
    for a in range(2,11):
        for b in range(a+1,12):
            for c in range(b+1,13):
                count +=1
                gb.active_token_column=[a,b,c]
                prob = sc.survive(gb)
                print(f"{a:2d},{b:2d},{c:2d}: {prob:4.2f}")
    print(f"Count of 3 Column Options: {count}")
    print(f"\nRandom sets of 4 Columns Open")
    gb.active_token_column=[-1,-1,-1]
    for j in range(10):
        
        gb.completed_col  = [False for _ in enumerate(gb.columns)]
        for col in random.choices( range(11),k=9): # pick up to 9 random column indices
            gb.completed_col[col] = True
        prob = sc.survive(gb)
        print(f"{j:4d} "+str(np.array(gb.completed_col,dtype=int))+f" {11-sum(gb.completed_col):2d}  {prob:4.2f}")
    
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped    
    
def test_base_player():
    gr = GameRules()

    bp = BasePlayer()

    gr.gb.active_token_column=[4,-1,-1]
    gr.check_completed()
    print("Stop probability with free tokens:")
    print(sum([bp.choose_stop(gr.gb) for _ in range(1000)])/1000.)
    
    gr.gb.active_token_column=[4,5,6]
    gr.check_completed()
    print("Stop probability without free tokens:")
    print(sum([bp.choose_stop(gr.gb) for _ in range(1000)])/1000.)

    gr.gb.active_token_column=[4,-1,-1]
    gr.check_completed()
    print("Survive probability with free tokens:")
    print(bp.sc.survive(gr.gb))
    
    gr.gb.active_token_column=[6,7,8]
    gr.check_completed()
    print("Stop probability with only 6,7,8:")
    print(bp.sc.survive(gr.gb))
    
    gr.gb.active_token_column=[2,3,12]
    gr.check_completed()
    print("Stop probability with only 2,3,12:")
    print(bp.sc.survive(gr.gb))
    
    gr.gb.active_token_column=[6,7,8]
    gr.check_completed()
    
    print("Time for 1000 evaluations:")
    prob = wrapper(bp.sc.survive,gr.gb)
    print(timeit.timeit(stmt=prob,number=1000))
    
    sc = survive_calc()
    
    print("Time for 1000 evaluations using index")
    prob_index = wrapper(sc.survive,gr.gb)
    print(timeit.timeit(stmt=prob_index,number=1000))
    
def test_game_rules():
    gr = GameRules()
    p1 = BasePlayer()
    p2 = BasePlayer()
    gr.add_player(p1)
    gr.add_player(p2)
    print(p1)
    print(p2)
    
    winner = gr.start()
    print(winner)
    print(np.array(gr.gb.player_heights[winner]))
    
    
def benchmark_one_player(player, num=2000):
    gr = GameRules()
    gr.add_player(player)
    turn_count = []    
    for i in range(num):
        gr.start()
        turn_count.append(gr.gb.turn_count)
    print(str(player) + "Avg Turns: "+ str( np.mean(turn_count))+" Std Turns: "+str(np.std(turn_count)))
    print("25%, 50%, 75%:"+str(np.percentile(turn_count,[25,50,75])))
    plt.hist(turn_count, bins=np.arange(0,25)+.5, density=True)
    plt.title(str(player))
    plt.xlabel("Turns to finsh single player")
    plt.show()
    
if __name__ == "__main__":
#    test_die_rolls(1000)
#    test_roll_available_options()
#    test_base_player_random_weighted_choose()
#    test_base_player()
#    test_survive_calc()
#    test_game_rules()
    bp = BasePlayer()
    benchmark_one_player(bp)
#    sc = survive_calc()