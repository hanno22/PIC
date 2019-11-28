import numpy as np


class particle:
    def __init__(self, m, q, x, v):
        self.m = m
        self.q = q
        self.x = x
        self.v = v

    def move(self, force, delta_t):
        v = self.v + force / self.m * delta_t
        x = self.x + self.v * delta_t
        return particle(self.m, self.q, x, v)


class system:
    def __init__(self, initial_state, delta_t):
        self.states = [initial_state]
        self.delta_t = delta_t

    def calc_next_step(self):
        last_state = self.states[-1]
        self.states.append(last_state.move_particles(self.delta_t))


class state:
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
                    forces[index_f] += c * particle_i.q * particle_f.q * 1 / (
                            particle_i.x - particle_f.x) ** 2 * np.sign(particle_f.x - particle_i.x)
        return forces

    def move_particles(self, delta_t):
        particles_new = []
        for index, partic in enumerate(self.particles, start=0):
            particles_new.append(partic.move(self.forces[index], delta_t))
        return state(particles_new, self.t + delta_t)


init_particles = [particle(0.3, 1, 0, 0), particle(0.3, -1, 3, 0)]
init_state = state(init_particles, 0)
sys = system(init_state, 0.1)

