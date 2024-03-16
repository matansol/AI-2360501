from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import sys

MAX_DISTANCE = 10
MAX_VALUE = sys.maxsize
MIN_VALUE = -sys.maxsize-1
WIN_VALUE = 1000

def calc_f_value(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    p1 = env.packages[1]
    p0 = env.packages[0]
    if robot.package is None:
        facture = 2
        pd1 = manhattan_distance(p1.position, p1.destination)
        rd1 = manhattan_distance(p1.position, robot.position)
        left =  facture*pd1 - rd1

        pd2 = manhattan_distance(p0.position, p0.destination)
        rd2 = manhattan_distance(p0.position, robot.position)
        right = facture*pd2 - rd2
        return max((facture*manhattan_distance(p1.position, p1.destination) - manhattan_distance(p1.position, robot.position))*p0.on_board,
                   (facture*manhattan_distance(p0.position, p0.destination) - manhattan_distance(p0.position, robot.position))*p1.on_board)
    
    return MAX_DISTANCE - manhattan_distance(robot.position, robot.package.destination)

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    if not env or robot_id < 0 or robot_id >= 2:
        return -1
    
    robot = env.get_robot(robot_id)
    a = 0
    b = 20
    c = 1
    d = 10
    f = calc_f_value(env, robot_id)
    val = a*robot.battery + b*robot.credit + c*calc_f_value(env, robot_id) + d*(robot.package is not None)
    return val

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
class Node:
    def __init__(self, env, operator, agent_id, parent=None, value=0, children=[], node_max=True, depth=0, alpha=MIN_VALUE, beta=MAX_VALUE) -> None:
        self.env = env
        self.operator = operator
        self.agent_id = agent_id
        self.parent = parent
        self.value = value
        self.children = children
        self.node_max = node_max
        self.depth = depth
        self.policy = None
        self.alpha = alpha
        self.beta = beta


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        magic_number_for_depth = 20
        root = Node(env, None, agent_id, None, 0, [], True, 0)
        current_solution = None
        root_node = self.create_tree_in_depth_j(root, 0, start, time_limit, agent_id)

        for depth_for_minimax in range(1, magic_number_for_depth + 1):  # determines depth
            root_node = self.create_tree_in_depth_j(root_node, depth_for_minimax, start, time_limit, agent_id)
            if (root_node == None):  # if folded, because of time shortage, return previous optimal solution
                return current_solution
            current_solution = root_node.policy.operator
        return current_solution

    def create_tree_in_depth_j(self, curr_node: Node, j: int, start_time, time_limit, root_agent_id):
        if curr_node == None:
            return None
        if curr_node.env.done():
            balance = curr_node.env.get_balances()
            if balance[0] == balance[1]:  # draw
                curr_node.value = 0
                return curr_node
            wining_agent = balance.index(max(balance))
            if wining_agent == curr_node.agent_id:
                curr_node.value = WIN_VALUE #balance[wining_agent] - balance[wining_agent^1]
            else:
                curr_node.value = -WIN_VALUE #(balance[wining_agent] - balance[wining_agent^1])
            return curr_node

        if j == 0 or curr_node.env.robots[curr_node.agent_id].battery == 0:
            curr_node.value = smart_heuristic(curr_node.env, root_agent_id)
            return curr_node

        TIME_PERCENTAGE = 0.9
        if j == 1 and len(curr_node.children) == 0:  # node hasn`t been expanded yet
            successors = self.successors(curr_node.env, curr_node.agent_id)
            for operator, child in zip(*successors):  # succ is a list of pairs of operators and the matching env
                if (time.time() - start_time > TIME_PERCENTAGE * time_limit): # time is over, must fold
                    return None
                child_node = Node(child, operator, curr_node.agent_id ^ 1,
                                  curr_node, 0, [], (not curr_node.node_max), curr_node.depth + 1)
                child_node = self.create_tree_in_depth_j(child_node, j - 1, start_time, time_limit, root_agent_id)
                if (child_node == None):
                    return None
                curr_node.children.append(child_node)

        else: # existing nodes
            for i in range(len(curr_node.children)):
                if (time.time() - start_time > TIME_PERCENTAGE *time_limit):  # time is over, must fold
                    return None
                child = self.create_tree_in_depth_j(curr_node.children[i], j - 1, start_time, time_limit, root_agent_id)
                if (child == None):
                    return None
                curr_node.children[i] = child

        curr_node.policy = self.find_best_child(curr_node)
        curr_node.value = curr_node.policy.value
        return curr_node

    # def choose_policy(self, root: Node):
    #     if (root.max == True):
    #         max_val = root.children[0].value
    #         policy_child = root.children[0]
    #         for child in root.children:
    #             if child.value > max_val:
    #                 max_val = child.value
    #                 policy_child = child
    #         return policy_child
    #     else:
    #         min_val = root.children[0].value
    #         policy_child = root.children[0]
    #         for child in root.children:
    #             if child.value < min_val:
    #                 min_val = child.value
    #                 policy_child = child
    #         return policy_child

    def find_best_child(self, curr_node: Node):
        if curr_node.node_max == True:
            big_small = lambda x,y: x>y
        else:
            big_small = lambda x,y: x<y
        max_val = curr_node.children[0].value
        policy_child = curr_node.children[0]
        for child in curr_node.children:
            if big_small(child.value, max_val):
                max_val = child.value
                policy_child = child
        return policy_child


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(AgentMinimax):
    # TODO: section d : 1
    def find_best_child(self, curr_node: Node):
        succ_list = curr_node.children
        policy_child = succ_list[0]
        if curr_node.node_max == True:
            max_val = policy_child.value
            for child in curr_node.children:
                if child.value > max_val:
                    max_val = child.value
                    policy_child = child
            return policy_child

        n = len(succ_list)
        special_operators = ['move west', 'pick up']
        E_values = 0
        for operator in special_operators:
            if operator in [node.operator for node in succ_list]:
                n += 1

        for succ in succ_list:
            p = 1/n
            if succ.operator in special_operators:
                p = 2/n
            E_values += p*succ.value
        policy_child.value = E_values
        return policy_child


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



