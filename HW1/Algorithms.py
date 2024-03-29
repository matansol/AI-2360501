import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

import csv
from rosman_nimrod_maps import small, medium, large, segel


class Node:
    def __init__(self, state, parent=None, cost=0, action=None, g=0, h=0, f=0) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        self.g = g
        self.h = h
        self.f = f
    
    
class Agent():
    def __init__(self) -> None:
        self.env = None
        self.MAX_CELL = 63
        self.MAX_DIST = 100000
    
    def _get_path(self, node: Node) -> (List[int], int):
        path = []
        total_cost = 0
        actions = []
        while(True):
            path.append(node.state)
            actions.append(node.action)
            
            total_cost += node.cost
            node = node.parent
            if node.parent is None:
                break
        
        return actions[::-1], total_cost
        
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        pass
    
    def calculate_L1(cell1: Tuple[int,int], cell2: Tuple[int,int]) -> int:
        """
        Returns the Manhattan distance between the 2 provided
        [row,col] Tuples.
        """
        y_distance = abs(cell1[0] - cell2[0])
        x_distance = abs(cell1[1] - cell2[1])
        return x_distance + y_distance
    
    def get_hsmap_value(self, current_state)->int:
        """
        Returns the hsmap heuristic value for each cell
        """
        current_row_col = self.env.to_row_col(current_state)
        if current_state[1] == True:
            L1_to_d1 = self.MAX_DIST
        else:
            L1_to_d1 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d1), current_row_col)
        
        if current_state[2] == True:
            L1_to_d2 = self.MAX_DIST
        else:
            L1_to_d2 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d2), current_row_col)
        
        min_dist = min(L1_to_d1, L1_to_d2)
        for goal_state in self.env.get_goal_states():
            L1_to_goal_state = WeightedAStarAgent.calculate_L1(self.env.to_row_col(goal_state),
                                                               current_row_col)
            min_dist = min(L1_to_goal_state, min_dist)
        return min_dist


class BFSAgent(Agent):
        
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        list_of_expended= []
        env.reset()
        self.env = env
        root = Node(env.get_state())
        expended = 0
        open = [root] # open is a queue initialized with the first state
        close = set()
        terminated = False
        while open:
            node = open.pop(0)    
            state = node.state 
            if state in close:
                continue       
            if node.parent and node.action is not None:
                prev_state = node.parent.state
                env.reset() # in order to reset the number of balls collected
                env.set_state(prev_state)
                state, cost, terminated = env.step(node.action)
                node.state = state
                node.cost = cost
            
            expended += 1
            close.add(state)
            
            if state[0] in [s[0] for s in env.get_goal_states()]: # we are in a goal cell but we didn't find all the balls
                continue
            
            succ_dict = env.succ(state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                if not next_state: # the cuerrent state is a hole
                    break
                ball1 = state[1]
                ball2 = state[2]
                if next_state[0] == env.d1[0]:
                    ball1 = True
                if next_state[0] == env.d2[0]:
                    ball2 = True
                next_state = (next_state[0], ball1, ball2)
                
                if env.is_final_state(next_state): # found the final state
                    child = Node(next_state, node, cost, action)
                    actions , total_cost = self._get_path(child)
                    #check if there is a tuple in list_of_expended that is twaice
                    return actions, total_cost, expended

                if next_state not in close and next_state not in [n.state for n in open]:
                    child = Node(state=next_state, parent=node, cost=cost, action=action) 
                    open.append(child)
        
        return [], 0, 0
    
    
def print_open(open):
    open_copy = heapdict.heapdict()
    for key, value in open.items():
        open_copy[key] = value
    print("print open")
    while open_copy:
        item = open_copy.popitem()
        print(f"key={item[0]}, state={item[1][1].state}")
    # for key in open:
    #     print(key)
    #     # print(f"f={key}, state={open[key][1].state}")
    print("end print open")

class WeightedAStarAgent(Agent):

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        start = env.get_state()
        root = Node(start, None, 0, None, 0, self.get_hsmap_value(start), h_weight*self.get_hsmap_value(start))
        expanded = 0
        open = heapdict.heapdict()
        key = (root.h, root.state) # arbitration is by h value and index
        open[key] = (key, root)
        close = dict()
        while open:
            # print()
            # print("new iteration")
            # print_open(open)
            curr_node = open.popitem()[1][1]
            state = curr_node.state
            
            if curr_node.parent and curr_node.action is not None:
                prev_state = curr_node.parent.state# if curr_node.parent else None
                env.reset()
                env.set_state(prev_state)
                state, cost, terminated = env.step(curr_node.action)
                curr_node.state = state
                curr_node.cost = cost
            
            
            if env.is_final_state(state):
                actions, total_cost = self._get_path(curr_node)
                return actions, total_cost, expanded
            
            if state[0] in [goal[0] for goal in env.get_goal_states()]: # we are in a goal cell but we didn't find all the balls
                expanded += 1
                key = (curr_node.f, curr_node.state) #added
                close[key] = curr_node # added
                # print("Reached goal state and break ", state)
                continue

            key = (curr_node.f, curr_node.state)
            close[key] = curr_node
            expanded += 1
            # print(f"state={state}")
            
            succ_dict = env.succ(state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                if not next_state: # we are in hole
                    break
                
                ball1 = state[1]
                ball2 = state[2]
                if next_state[0] == env.d1[0]:
                    ball1 = True
                if next_state[0] == env.d2[0]:
                    ball2 = True
                next_state = (next_state[0], ball1, ball2)
                new_g_value = curr_node.g + cost
                h_value = self.get_hsmap_value(next_state)
                new_f_value = (1-h_weight)*new_g_value + h_weight*h_value
                new_node = Node(next_state, curr_node, cost, action, new_g_value, h_value, new_f_value)

                
                if next_state in [val[1].state for val in open.values()]:  # the child is in open
                    # print(f"action={action}, next_state in open")
                    key = [key for key in open.keys() if open[key][1].state == next_state][0]
                    curr_node_in_open = open[key][1]
                    if new_f_value < curr_node_in_open.f:  # the new path is better
                        open.pop(key)
                        new_key = (new_f_value, next_state)
                        open[new_key] = (new_key, new_node)
                        
                else: 
                    if next_state in [val.state for val in close.values()]: # the child is in close
                        # print(f"action={action}, next_state in close")
                        key = [key for key in close.keys() if close[key].state == next_state][0]
                        curr_node_in_close = close[key]
                        if new_f_value < curr_node_in_close.f: # the new path is better so we remove the child from close and add it to open
                            close.pop(key)
                            new_key = (new_f_value, next_state)
                            open[new_key] = (new_key, new_node)
                            
                    else: # the child is not in open and not in close
                        # print(f"action={action}, next_state not in open and not in close add to open")
                        new_key = (new_f_value, next_state)
                        open[new_key] = (new_key, new_node)
        return [], 0, 0
    

class AStarEpsilonAgent(Agent):
    # h_focal(v) = g(v)
        
    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        start = env.get_state()
        root = Node(start, None, 0, None, 0, self.get_hsmap_value(start),self.get_hsmap_value(start))
        expanded = 0
        open = heapdict.heapdict()
        key = (root.h, root.state)  # arbitration is by h value and index
        open[key] = (key, root)
        close = dict()
        while open:
            [curr_node, open] = AStarEpsilonAgent.choose_min_from_focal(open, epsilon)
            
            if curr_node.parent and curr_node.action is not None:
                prev_state = curr_node.parent.state #if node.parent else None
                env.reset()
                env.set_state(prev_state)
                state, cost, terminated = env.step(curr_node.action)
                curr_node.state = state
                curr_node.cost = cost
            curr_state = curr_node.state
            
            if env.is_final_state(curr_state): # we found the final state
                actions, total_cost = self._get_path(curr_node)
                return actions, total_cost, expanded
              
            expanded += 1
            if curr_state[0] in [goal[0] for goal in env.get_goal_states()]: # we are in a goal cell but we didn't find all the balls
                continue  
            
            key = (curr_node.f, curr_state)
            close[key] = curr_node
            succ_dict = env.succ(curr_state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                if not next_state:
                    break
                
                ball1 = curr_state[1]
                ball2 = curr_state[2]
                if next_state[0] == env.d1[0]:
                    ball1 = True
                if next_state[0] == env.d2[0]:
                    ball2 = True
                next_state = (next_state[0], ball1, ball2)
                new_g_value = curr_node.g + cost
                h_value = self.get_hsmap_value(next_state)
                new_f_value =  new_g_value +  h_value
                new_node = Node(next_state, curr_node, cost, action, new_g_value, h_value, new_f_value)
                
                if next_state in [val[1].state for val in open.values()]:  # the child is in open
                    key = [key for key in open.keys() if open[key][1].state == next_state][0]
                    curr_node_in_open = open[key][1]
                    if new_f_value < curr_node_in_open.f:  # the new path is better
                        open.pop(key)
                        new_key = (new_f_value, next_state)
                        open[new_key] = (new_key, new_node)
                        
                else: 
                    if next_state in [val.state for val in close.values()]: # the child is in close
                        key = [key for key in close.keys() if close[key].state == next_state][0]
                        curr_node_in_close = close[key]
                        if new_f_value < curr_node_in_close.f: # the new path is better so we remove the child from close and add it to open
                            close.pop(key)
                            new_key = (new_f_value, next_state)
                            open[new_key] = (new_key, new_node)
                            
                    else: # the child is not in open and not in close
                        new_key = (new_f_value, next_state)
                        open[new_key] = (new_key, new_node)
        return [], 0, 0
    

    def choose_min_from_focal(open, epsilon):
        """
        Returns Min node based on h_focal heuristic (g)
        from focal set based on epsilon
        """
        node = open.peekitem()
        min_h_val = node[1][1].f # key is comprised of h_val and index
        focal_value = min_h_val * (1+epsilon)
        focal_dict = heapdict.heapdict()
        for key in open.keys():
            temp_node = open[key][1]
            if(temp_node.f <= focal_value):
                key_for_focal_dict = (temp_node.g,temp_node.state[0],key)
                focal_dict[key_for_focal_dict] = (key_for_focal_dict,temp_node)
        chosen_object = focal_dict.peekitem()
        original_key = (chosen_object[0][2])
        open.pop(original_key)
        return [chosen_object[1][1],open]

    def main():
        final = {}
        final.update(small)
        # final.update(medium)
        # final.update(large)
        final.update(segel)
        test_envs = {}
        for board_name, board in final.items():
            test_envs[board_name] = DragonBallEnv(board)
        for env_name, env in test_envs.items():
            if env_name == "15x17":
                WA_agent = WeightedAStarAgent()
                actions, total_cost, expanded = WA_agent.search(env, h_weight=0.5)
                print(f"Total_cost: {total_cost}")
                print(f"Expanded: {expanded}")
                print(f"Actions: {actions}")


    if __name__ == "__main__":
        main()
    
