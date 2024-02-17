import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state, parent=None, cost=0, action=None) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        self.g = 0
        self.h = 0
        self.f = 0
    
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
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError