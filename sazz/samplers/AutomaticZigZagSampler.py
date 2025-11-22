from functools import partial

import numpy as np
from tqdm.auto import tqdm

from sazz.utils.utils import brent, sample_trajectory_at_regular_intervals


class AutomaticZigZagSampler:
    """
    Basic Automatic ZigZag Sampler
    """
    def __init__(self, N: int, D: int, grad_target, t_max: float = 0.01, gamma: float = 0.01):
        self.N = N
        self.D = D
        self.Position = np.zeros((N+1,D))
        self.Velocity = np.zeros((N+1,D))
        self.Time = np.zeros(N+1)
        self.gamma = gamma
        self.grad_target = grad_target
        self.t_max = t_max
        # And actually initialize
        self.Position[0,:] = np.random.normal(0,1,size=D)
        self.Velocity[0,:] = -1.0 + 2.0 * np.random.randint(2,size=D)
        self.Time[0] = 0.0
        # Set current iteration, which after init is 1
        self.iteration = 1

    def rate(self, time, pos0, vel):
        pos = pos0 + time*vel
        rate = np.sum(np.maximum(0,vel*self.grad_target(pos)) + self.gamma)
        return -rate

    def sample(self):
        time_passed = 0.0
        # Set up a progress bar
        pbar = tqdm(total=self.N, desc="Zig-Zag time", unit="iter")

        while self.iteration <= self.N:
            n = self.iteration
            # Current velocity and position
            pos = self.Position[(n-1),:]
            vel = self.Velocity[(n-1),:]
            # Define a single-argument function for Brent
            rate_time = partial(self.rate,pos0=pos,vel=vel)

            # Optimize bound on rate
            x_star = brent(rate_time,0,self.t_max) # Find time point at which rate is maximum
            lambda_max = -rate_time(x_star) # Rate at this time, basically a flat bound
            # Compute event time
            tau_star = np.random.exponential(1.0/lambda_max,1)
            # Thinning
            u = np.random.random()
            lambda_star = -rate_time(tau_star)
            p = lambda_star / lambda_max
            acc = (u <= p)
            if acc and (tau_star < self.t_max):
                # Event <3
                # Which component is flipping
                rates = np.maximum(0,vel*self.grad_target(pos + vel*tau_star)) + self.gamma
                rates = np.asarray(rates).astype('float64')
                rates = rates / np.sum(rates)
                i0 = np.argmax(np.random.multinomial(1, rates)).item()
                # Setting new values for positions and parameters
                self.Position[n,:] = pos + vel*tau_star
                self.Time[n] = self.Time[(n-1)] + time_passed + tau_star.item()
                # Velocity flips
                vel[i0] *= -1.0
                self.Velocity[n,:] = vel
                # Increase iteration and reset time_passed
                self.iteration += 1
                time_passed = 0.0
                pbar.update(1)
            else:
                time_passed += self.t_max
        print("Time passed: " + str(self.Time[n]))

    def getSamples(self,N_samples: int = 1000):
        time = np.max(self.Time).item()
        dt = time / (N_samples-1)
        _, samples, _ = sample_trajectory_at_regular_intervals(self.Position, self.Velocity, self.Time, dt)
        return samples






