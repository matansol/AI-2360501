import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state, parent=None, cost=0, action=None, g=0, h=0, f=0) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        self.g = g
        self.h = h
        self.f = f
    
    def print_node(self):
        print(f"state: {self.state}, parent: {self.parent.state if self.parent else None}, action: {self.action}, cost: {self.cost}")
    
    
class BFSAgent():
#     function Breadth-First-Search-Graph(problem):
#       node ← make_node(problem.init_state, null)
#       if problem.goal(node.state) then return solution(node)
#       OPEN ← {node} /* a FIFO queue with node as the only element */
#       CLOSE ← {} /* an empty set */
#       while OPEN is not empty do:
#           node ← OPEN.pop() /* chooses the shallowest node in OPEN */
#           CLOSE.add(node.state)
#           loop for s in expand(node.state):
#               child ← make_node(s, node)
#               if child.state is not in CLOSE and child is not in OPEN:
#               if problem.goal(child.state) then return solution(child)
#               OPEN.insert(child)
#       return failure

    def __init__(self) -> None:
        self.env = None
        
    def _get_path(self, node: Node) -> (List[int], int):
        path = []
        total_cost = 0
        actions = []
        while(True):
            # node.print_node()
            path.append(node.state)
            actions.append(node.action)
            
            total_cost += node.cost
            node = node.parent
            if node.parent is None:
                break
        print(f"all the actions:{actions[::-1]}")
        print(f"all the path:{path[::-1]}")
        
        return actions[::-1], total_cost
    
    def print_open_list(self, open):
        for node in open:
            parent = node.parent.state if node.parent else (0,False,False)
            print(f"{parent}-{node.action}->{node.state}")
        
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        start = env.get_state()
        root = Node(start)
        expended = 0
        # open is a queue
        open = [root]
        close = set()
        while open:
            node = open.pop(0)
            state = node.state
            
            if node.parent and node.action is not None:
                prev_state = node.parent.state if node.parent else None
                env.reset()
                env.set_state(prev_state)
                state, cost, terminated = env.step(node.action)
                node.state = state
                node.cost = cost
            expended += 1
            close.add(state)
            
            succ_dict = env.succ(state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                next_state = (next_state[0], state[1], state[2])
                # print(f"current state={state}, action={action}, next state={next_state}")
                #if the next state is a final state or hole
                if terminated:
                    if env.is_final_state(next_state):
                        child = Node(next_state, node, cost, action)
                        actions , total_cost = self._get_path(child)
                        return actions, total_cost, expended
                    else: # hole
                        continue
            
                if next_state not in close and next_state not in [n.state for n in open]:
                    child = Node(next_state, node, cost, action)
                    open.append(child)
        return None, -1, -1


class WeightedAStarAgent():
    # OPEN <- make_node(P.start, NIL, 0, h(P.start)) //order according to f-value
    # CLOSE <- {}
    # While OPEN != empty_set
    #   n <- OPEN.pop_min()
    #   CLOSE <- CLOSE + {n}
    #   If P.goal_test(n)
    #     Return path(n)
    #   For s in P.SUCC (n)
    #       new_g <- n.g() + P.COST(n.s,s); new_f = new_g + h(s) //newly-computed cost to reach s
    #      If s NOT IN ( OPEN + CLOSED)
    #            n' <- make_node(s, n, new_g, new_f )
    #           OPEN.insert(n’)
    #       Else if s IN OPEN
    #           n_curr <- node in OPEN with state s
    #           If new_f < n_curr.f () //found better path to s
    #               n_curr <- update_node(s, n, new_g , new_f)
    #               OPEN.update_key(n_curr) //don’t forget to update place in OPEN…
    #           Else
    #       Else // s IN CLOSED
    #           n_curr <- node in CLOSED with state s
    #           If new_f < n_curr.f () //found better path to s
    #               n_curr <- update_node(s, n, new_g , new_f)
        #           OPEN.insert(n_curr)
    #               CLOSED.remove(n_curr)
    def __init__(self) -> None:
        self.env = None

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        start = env.get_state()
        root = Node(start,None,0,None,0,self.get_hsmap_value(start),h_weight*self.get_hsmap_value(start))
        expanded = 0
        open = heapdict.heapdict()
        key = (root.h,root.state[0]) # arbitration is by h value and index
        open[key] = (key,root)
        close = dict()
        while open:
            node = open.popitem()[1][1]
            state = node.state
            if node.parent and node.action is not None:
                prev_state = node.parent.state if node.parent else None
                env.reset()
                env.set_state(prev_state)
                state, cost, terminated = env.step(node.action)
                node.state = state
                node.cost = cost
            expanded += 1
            key = (node.f,node.state[0])
            close[key]=(key,node)
            succ_dict = env.succ(state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                next_state = (next_state[0], state[1], state[2])
                if terminated:
                    if env.is_final_state(next_state):
                        child = Node(next_state, node, cost, action)
                        actions, total_cost = self._get_path(child)
                        return actions, total_cost, expanded
                    else:  # hole
                        continue
                new_g_value = node.g + cost
                h_value = self.get_hsmap_value(next_state)
                new_f_value = (1-h_weight)*new_g_value + h_weight*h_value
                existing_state = False
                for key in open.keys():
                    if open[key][1].state == next_state:
                        existing_state = True
                        current_state_in_open = open[key][1]
                        if new_f_value < current_state_in_open.f:
                            #updates better path to node
                            popped_node = open.pop(key)
                            key = (new_f_value,current_state_in_open.state[0])
                            open[key] = (key,popped_node[1])
                            break
                for key in close.keys():
                    if close[key][1].state == next_state:
                        existing_state = True
                        current_state_in_close = close[key][1]
                        if new_f_value < current_state_in_close.f:
                            # updates better path to node
                            popped_node = close.pop(key)
                            key = (new_f_value, current_state_in_close.state[0])
                            open[key] = (key,popped_node[1])
                            break
                if(existing_state == False):
                    # each node holds his own cost,action and parent only!
                    # Later when we return the solution, we will return the entire path
                    new_node = Node(next_state,node,cost,action,new_g_value,h_value,new_f_value)
                    key = (new_f_value, next_state[0])
                    open[key] = (key,new_node)
        return None, -1, -1

    def _get_path(self, node: Node) -> (List[int], int):
        path = []
        total_cost = 0
        actions = []
        while (True):
            # node.print_node()
            path.append(node.state)
            actions.append(node.action)
            total_cost += node.cost
            node = node.parent
            if node.parent is None:
                break
        print(f"all the actions:{actions[::-1]}")
        print(f"all the path:{path[::-1]}")

        return actions[::-1], total_cost

    def calculate_L1(cell1: Tuple[int,int], cell2: Tuple[int,int]) -> int:
        """
        Returns the Manhattan distance between the 2 provided
        [row,col] Tuples.
        """
        y_distance = abs(cell1[0]-cell2[0])
        x_distance = abs(cell1[1] - cell2[1])
        return x_distance+y_distance
    def get_hsmap_value(self,current_state)->int:
        """
        Returns the hsmap heuristic value for each cell
        """
        current_row_col = self.env.to_row_col(current_state)
        L1_to_d1 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d1),current_row_col)
        L1_to_d2 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d2),current_row_col)
        min_dist = min(L1_to_d1,L1_to_d2)
        for goal_state in self.env.get_goal_states():
            L1_to_goal_state = WeightedAStarAgent.calculate_L1(self.env.to_row_col(goal_state),
                                                               current_row_col)
            min_dist = min(L1_to_goal_state,min_dist)
        return min_dist


class AStarEpsilonAgent():
    # h_focal(v) = g(v)
    def __init__(self) -> None:
        self.env = None
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        start = env.get_state()
        root = Node(start, None, 0, None, 0, self.get_hsmap_value(start),self.get_hsmap_value(start))
        expanded = 0
        open = heapdict.heapdict()
        key = (root.h, root.state[0])  # arbitration is by h value and index
        open[key] = (key, root)
        close = dict()
        while open:
            #node = open.popitem()[1][1] # Automatically chooses min node
            [node,open] = AStarEpsilonAgent.choose_min_from_focal(open,epsilon)
            state = node.state
            if node.parent and node.action is not None:
                prev_state = node.parent.state if node.parent else None
                env.reset()
                env.set_state(prev_state)
                state, cost, terminated = env.step(node.action)
                node.state = state
                node.cost = cost
            expanded += 1
            key = (node.f, node.state[0])
            close[key] = (key, node)
            succ_dict = env.succ(state)
            actions = list(succ_dict.keys())
            for action in actions:
                next_state, cost, terminated = succ_dict[action]
                next_state = (next_state[0], state[1], state[2])
                if terminated:
                    if env.is_final_state(next_state):
                        child = Node(next_state, node, cost, action)
                        actions, total_cost = self._get_path(child)
                        return actions, total_cost, expanded
                    else:  # hole
                        continue
                new_g_value = node.g + cost
                h_value = self.get_hsmap_value(next_state)
                new_f_value =  new_g_value +  h_value
                existing_state = False
                for key in open.keys():
                    if open[key][1].state == next_state:
                        existing_state = True
                        current_state_in_open = open[key][1]
                        if new_f_value < current_state_in_open.f:
                            # updates better path to node
                            popped_node = open.pop(key)
                            key = (new_f_value, current_state_in_open.state[0])
                            open[key] = (key, popped_node[1])
                            break
                for key in close.keys():
                    if close[key][1].state == next_state:
                        existing_state = True
                        current_state_in_close = close[key][1]
                        if new_f_value < current_state_in_close.f:
                            # updates better path to node
                            popped_node = close.pop(key)
                            key = (new_f_value, current_state_in_close.state[0])
                            open[key] = (key, popped_node[1])
                            break
                if (existing_state == False):
                    # each node holds his own cost,action and parent only!
                    # Later when we return the solution, we will return the entire path
                    new_node = Node(next_state, node, cost, action, new_g_value, h_value, new_f_value)
                    key = (new_f_value, next_state[0])
                    open[key] = (key, new_node)
        return None, -1, -1

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

            #Notice! For same G values- should choose also by index?


    def calculate_L1(cell1: Tuple[int,int], cell2: Tuple[int,int]) -> int:
        """
        Returns the Manhattan distance between the 2 provided
        [row,col] Tuples.
        """
        y_distance = abs(cell1[0]-cell2[0])
        x_distance = abs(cell1[1] - cell2[1])
        return x_distance+y_distance
    def get_hsmap_value(self,current_state)->int:
        """
        Returns the hsmap heuristic value for each cell
        """
        current_row_col = self.env.to_row_col(current_state)
        L1_to_d1 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d1),current_row_col)
        L1_to_d2 = WeightedAStarAgent.calculate_L1(self.env.to_row_col(self.env.d2),current_row_col)
        min_dist = min(L1_to_d1,L1_to_d2)
        for goal_state in self.env.get_goal_states():
            L1_to_goal_state = WeightedAStarAgent.calculate_L1(self.env.to_row_col(goal_state),
                                                               current_row_col)
            min_dist = min(L1_to_goal_state,min_dist)
        return min_dist

    def _get_path(self, node: Node) -> (List[int], int):
        path = []
        total_cost = 0
        actions = []
        while (True):
            # node.print_node()
            path.append(node.state)
            actions.append(node.action)
            total_cost += node.cost
            node = node.parent
            if node.parent is None:
                break
        print(f"all the actions:{actions[::-1]}")
        print(f"all the path:{path[::-1]}")

        return actions[::-1], total_cost

def main():
    MAPS = {
        "4x4": ["SFFF",
                "FDFF",
                "FFFD",
                "FFFG"],
        "8x8": [
            "SFFFFFFF",
            "FFFFFTAL",
            "TFFHFFTF",
            "FFFFFHTF",
            "FAFHFFFF",
            "FHHFFFHF",
            "DFTFHDTL",
            "FLFHFFFG",
        ],
    }
    #env = DragonBallEnv(MAPS["8x8"])
    env = DragonBallEnv(MAPS["4x4"])
    state = env.reset()
    print('Initial state:', state)
    print('Goal states:', env.goals)
    print(env.render())
    #a_star_agent =WeightedAStarAgent()
    a_star_agent = AStarEpsilonAgent()
    result = a_star_agent.ssearch(env,1)

if __name__ == "__main__":
    main()