from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

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

    
    

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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