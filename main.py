import numpy as np
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, m, q, x, v):
        self.m = m
        self.q = q
        self.x = x
        self.v = v

    def move(self, force, delta_t):
        v = self.v + force / self.m * delta_t
        x = self.x + self.v * delta_t
        return Particle(self.m, self.q, x, v)


class System:
    def __init__(self, initial_state, delta_t):
        self.states = [initial_state]
        self.delta_t = delta_t

    def calc_next_step(self):
        last_state = self.states[-1]
        self.states.append(last_state.move_particles(self.delta_t))

    def calc_n_next_steps(self, n):
        for i in range(n):
            self.calc_next_step()

    def plot_states(self):
        _t = np.array(list(map(lambda s: s.t, self.states)))
        _x = np.array(list(map(lambda s: [part.x for part in s.particles], self.states)))
        for i in range(0, len(_x[0])):
            plt.plot(_t, _x[:, i])
        plt.show()


class State:
    def __init__(self, particles, t):
        self.particles = particles
        self.t = t
        self.forces = self.__calc_forces()

    def __calc_forces(self):
        c = 0.3
        forces = [0] * len(self.particles)
        for index_f, particle_f in enumerate(self.particles, start=0):
            for index_i, particle_i in enumerate(self.particles, start=0):
                if index_f != index_i:
                    forces[index_f] += c * particle_i.q * particle_f.q / (particle_i.x - particle_f.x) ** 2 * np.sign(particle_f.x - particle_i.x)
        return forces

    def move_particles(self, delta_t):
        particles_new = []
        for index, particle in enumerate(self.particles, start=0):
            particles_new.append(particle.move(self.forces[index], delta_t))
        return State(particles_new, self.t + delta_t)


init_particles = [Particle(4, 1, 0, 0), Particle(1, -1, 3, 0)]
init_state = State(init_particles, 0)
sys = System(init_state, 0.1)
sys.calc_n_next_steps(100)
sys.plot_states()