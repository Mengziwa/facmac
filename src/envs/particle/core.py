# From https://github.com/openai/multiagent-particle-envs
import numpy as np


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        #self.power = []
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        # self.c = None
        # 簇外干扰
        self.inter = []
        # 簇内干扰
        self.intra = []


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # power control
        self.power = []
        # RB allocation
        self.rb = None
        # communication action
        # self.c = None


# from https://github.com/shariqiqbal2810/multiagent-particle-envs/blob/master/multiagent/core.py
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents  # + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            # agent.action = agent.action_callback(agent, self)
            agent.action.u = agent.action_callback(agent, self)  # NOTE: need this in the apply_action_force() function
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u  # + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # integrate physical state
    #def integrate_state(self, p_force):
    #    sensitivity = 100.0
    #    for i, entity in enumerate(self.entities):
    #        if entity.movable:
    #            noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
    #            noise *= sensitivity
    #            old_pos = entity.state.p_pos
    #            entity.state.p_pos = entity.state.p_pos + entity.action.u + noise
    #            if entity.state.p_pos[0] < 0 or entity.state.p_pos[0] >= 1000:
    #                entity.state.p_pos = old_pos
    #            if entity.state.p_pos[1] < 0 or entity.state.p_pos[1] >= 1000:
    #                entity.state.p_pos = old_pos
                # todo 3D
                # if entity.state.p_pos[2] < 100 or entity.state.p_pos[2] >= 300:
                #    entity.state.p_pos = old_pos


    #def update_state(self, agents):
    #    p_force = [None] * len(self.agents)
    #    sensitivity = 1.0
    #    for i, agent in enumerate(self.agents):
    #        if agent.movable:
    #            noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
    #            noise *= sensitivity
    #            old_pos = agent.state.p_pos
    #            agent.state.p_pos = agent.state.p_pos + agent.action.u + noise
    #            if agent.state.p_pos[0] < 0 or agent.state.p_pos[0] >= 1000:  # area_width
    #                agent.state.p_pos = old_pos
    #            if agent.state.p_pos[1] < 0 or agent.state.p_pos[1] >= 1000:  # area_width
    #                agent.state.p_pos = old_pos
                # todo 3D
                # if agent.state.p_pos[2] < 100 or agent.state.p_pos[2] >= 200:
                #    agent.state.p_pos = old_pos

    #    for agent in self.agents:
            # set communication state (directly for now)
    #        if agent.silent:
    #            agent.state.c = np.zeros(self.dim_c)
    #        else:
    #            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
    #            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # all below from https://github.com/shariqiqbal2810/multiagent-particle-envs/blob/master/multiagent/core.py
    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)
