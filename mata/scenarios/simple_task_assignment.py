import numpy as np
from mata.core import World, Agent, Task
from mata.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_tasks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add tasks
        world.tasks = [Task() for i in range(num_tasks)]
        for i, task in enumerate(world.tasks):
            task.name = 'task %d' % i
            task.collide = False
            task.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_step = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for tasks
        for i, task in enumerate(world.tasks):
            task.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.ability = np.random.randint(1, 10)
            agent.route = [agent.state.p_pos]
        for i, task in enumerate(world.tasks):
            task.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            task.state.p_vel = np.zeros(world.dim_p)
            task.amount = np.random.randint(10, 20)
            task.exec_state = 0
            task.amount_list = [task.amount]

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_tasks = 0
        min_dists = 0
        for l in world.tasks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_tasks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_tasks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each task, penalized for collisions
        rew = 0
        for t in world.tasks:
            if t.exec_state == 0:  # 任务没有被执行完成
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - t.state.p_pos))) for a in world.agents]
                rew -= min(dists)
                d = np.sqrt(np.sum(np.square(agent.state.p_pos - t.state.p_pos)))
                if d < 0.3:
                    rew += min(dists) / 2
                    t.amount -= agent.ability
                    if t.amount <= 0:
                        t.exec_state = 1
                        rew += 5
        # sum_t = 0
        # for t in world.tasks:
        #     if t.exec_state == 1:
        #         sum_t += 1
        # if sum_t == len(world.tasks):
        #     rew += 50
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        task_pos = []
        for task in world.tasks:  # world.tasks:
            task_pos.append(task.state.p_pos - agent.state.p_pos)
        # task state and size
        task_state, task_amount = [], []
        for task in world.tasks:  # world.tasks:
            task_state.append(np.array([task.exec_state]))
            task_amount.append(np.array([task.amount]))
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        agent_ability = np.array([agent.ability])
        # print('v: ', [agent.state.p_vel])
        # print('p: ', [agent.state.p_pos])
        # print('a: ', [agent_ability])
        # print('ts: ', task_state)
        # print('tp: ', task_pos)
        # print('ta: ', task_amount)
        return np.concatenate([agent.state.p_vel] + [
            agent.state.p_pos] + [agent_ability] + task_state + task_pos + task_amount + other_pos + comm)

    def info(self, agent, world):
        inf = {'total_step': world.num_step, 'mv_step': [], 'exec_step': [], 'route': [], 'task_pos': [], 'task_amount': []}
        agent.route.append(list(agent.state.p_pos))
        inf['route'] = agent.route
        for t in world.tasks:
            inf['task_pos'].append(list(t.state.p_pos))
            if agent.name == 'agent 0':     # 任务的大小列表只用更新一次
                t.amount_list.append(t.amount)
            inf['task_amount'].append(t.amount_list)
        return inf

    def done(self, agent, world):
        for t in world.tasks:
            if t.exec_state != 1:
                return False
        return True
