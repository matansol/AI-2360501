from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

MAX_DISTANCE = 10

def calc_f_value(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    p1 = env.packages[1]
    p0 = env.packages[0]
    if robot.package is None:
        facture = 2
        return max(facture*manhattan_distance(p1.position, p1.destination) - manhattan_distance(p1.position, robot.position),
               facture*manhattan_distance(p0.position, p0.destination) - manhattan_distance(p0.position, robot.position))
    
    return MAX_DISTANCE-manhattan_distance(robot.position, robot.package.destination)

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    if not env or robot_id < 0 or robot_id >= 2:
        return -1
    other_robot_id = (robot_id + 1) % 2
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot(other_robot_id)
    a = 1
    b = 10
    c = 1
    d = 10
    # return robot.position[0]
    return a*robot.battery + b*robot.credit + c*calc_f_value(env, robot_id) + d*(robot.package is not None)

class Node:
    def __init__(self, env,operator, agent_id, parent=None, value = 0, children = [],max=True,depth =0) -> None:
        self.env = env
        self.operator = operator
        self.agent_id=agent_id
        self.parent = parent
        self.value = value
        self.children = children
        self.max = max
        self.depth = depth
        self.policy =None

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        ## TO HANDLE TIME LIMIT I SHOULD DO IT WITH ITERATIONS
        start = time.time()
        depth_for_minimax = 1
        magic_number_for_depth = 1000
        root = Node(env,None,agent_id, None, 0, [],True,0)
        curr_node = root
        max_node = True
        root_node = self.create_tree_in_depth_j(root, 0)
        for depth_for_minimax in range(1,magic_number_for_depth): #determines depth
            root_node = self.create_tree_in_depth_j(root_node,depth_for_minimax)
            #smart_heuristic(env,robot_id)
            end = time.time()
            if end - start > time_limit * 0.98:
                return root_node.policy.operator
        raise NotImplementedError()
    def choose_min(self):
        raise NotImplementedError()
    def choose_max(self):
        raise NotImplementedError()
    def create_tree_in_depth_j(self,root:Node,j:int):
        if j == 0:
            root.value = smart_heuristic(root.env, root.agent_id)
            return root
        if j == 1 and len(root.children) == 0 : # node hasn`t been expanded yet
            successors = self.successors(root.env, root.agent_id)
            max_node = root.max
            for i in range(len(successors[0])):  # succ is a list of paires of operators and the matching env
                child_node = Node(successors[1][i],successors[0][i],root.agent_id^1, root, 0,[], (not root.max),root.depth+1)
                child_node = self.create_tree_in_depth_j(child_node, j - 1)
                root.children.append(child_node)
        else: # existing nodes
            for i in range(len(root.children)):
                child = self.create_tree_in_depth_j(root.children[i],j-1)
                root.children[i] = child
        root.policy = self.choose_policy(root)
        root.value = root.policy.value
        return root
    def expand_tree_another_level(root_node:Node,depth_for_expansion=-1):
        # Assumption that graph is balanced
        curr_node = root_node
        if len(curr_node.children) !=0 and curr_node.depth != depth_for_expansion:
            successors = Agent.successors(curr_node.env, curr_node.agent_id)
            for i in len(successors):  # succ is a list of paires of operators and the matching env
                child_node = Node(successors[0][i], successors[1][i], curr_node.agent_id ^ 1, curr_node,
                                  0, [], (not curr_node.max),
                                  curr_node.depth + 1)
                curr_node.append(child_node)
            return curr_node
        else: # expand till depth matches depth for expansion
            for child in root_node.children: # need to replace existing child with new
            # need to make sure, changes aren`t made on the copy
                return NotImplementedError
    def choose_policy(self,root:Node):
        if(root.max == True):
            max = root.children[0].value
            policy_child = root.children[0]
            for i in range(len(root.children)):
                if root.children[i].value > max:
                    max = root.children[i].value
                    policy_child = root.children[i]
            return policy_child
        else:
            min = root.children[0].value
            policy_child = root.children[0]
            for i in range(len(root.children)):
                if root.children[i].value < min:
                    min = root.children[i].value
                    policy_child = root.children[i]
            return policy_child


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)