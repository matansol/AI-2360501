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
    def __init__(self, env , operator , agent_id, parent=None, value = 0, children = [] , max=True,depth =0) -> None:
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
        start = time.time()
        depth_for_minimax = 1
        magic_number_for_depth = 1000
        root = Node(env, None , agent_id , None , 0, [] , True , 0 )
        curr_node = root
        max_node = True
        current_solution = None
        root_node = self.create_tree_in_depth_j(root, 0, start, time_limit)
        for depth_for_minimax in range(1, magic_number_for_depth): #determines depth
            root_node = self.create_tree_in_depth_j(root_node, depth_for_minimax, start, time_limit)
            if( root_node == None): # if folded, because of time shortage, return previous optimal solution
                return current_solution
            current_solution = root_node.policy.operator
            end = time.time()

        raise NotImplementedError()

    def create_tree_in_depth_j(self, root:Node, j:int, start_time, time_limit):
        if root == None:
            return None
        if j == 0:
            root.value = smart_heuristic(root.env, root.agent_id)
            return root

        if j == 1 and len(root.children) == 0 : # node hasn`t been expanded yet
            successors = self.successors(root.env, root.agent_id)
            max_node = root.max
            for operator, child in zip(*successors):  # succ is a list of paires of operators and the matching env
                if (time.time() - start_time > 0.9 * time_limit): # time is over, must fold
                    return None
                child_node = Node(child, operator, root.agent_id^1,
                                  root, 0, [], (not root.max), root.depth+1)
                child_node = self.create_tree_in_depth_j(child_node, j - 1, start_time, time_limit)
                if(child_node == None):
                    return None
                root.children.append(child_node)

        else: # existing nodes
            for i in range(len(root.children)):
                if(time.time() - start_time > 0.9 *time_limit): # time is over, must fold
                    return None
                child = self.create_tree_in_depth_j(root.children[i], j-1, start_time, time_limit)
                if(child == None):
                    return None
                root.children[i] = child

        root.policy = self.choose_policy(root)
        root.value = root.policy.value
        return root

    def choose_policy(self,root:Node):
        if(root.max == True):
            max = root.children[0].value
            policy_child = root.children[0]
            for child in root.children:
                if child.value > max:
                    max = child.value
                    policy_child = child
            return policy_child
        else:
            min = root.children[0].value
            policy_child = root.children[0]
            for child in root.children:
                if child.value < min:
                    min = child.value
                    policy_child = child
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