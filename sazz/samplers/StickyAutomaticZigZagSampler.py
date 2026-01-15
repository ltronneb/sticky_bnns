from functools import partial

import numpy as np
from tqdm.auto import tqdm

from sazz.samplers.AutomaticZigZagSampler import AutomaticZigZagSampler
from sazz.utils.utils import brent


class StickyAutomaticZigZagSampler(AutomaticZigZagSampler):
    """
    Sticky version of the automatic ZigZag

    Parameters:
        kappa:
    """
    def __init__(self, N: int, D: int, grad_target, kappa,
                 t_max: float = 0.01, gamma: float = 0.01,
                 tempering: bool = True):
        super().__init__(N, D, grad_target, t_max=t_max, gamma=gamma)
        self.kappa = kappa
        self.tempering = tempering
        # Also here set up clocks to keep track of frozen states etc.
        self.freezing = self.freezing_time(self.Position[0,:],self.Velocity[0,:])
        self.thawing = np.full((D), np.inf, float)
        self.active = np.full((D), True, bool)
        self.vel_at_freeze = np.zeros((D))

    def freezing_time(self, pos, vel):
        valid = (pos*vel < 0) # were current position and velocity means they'll freeze at some point
        freeze = np.full((pos.size), np.inf, float)
        freeze[valid] = (-pos[valid] / vel[valid]) # at current velocity here is when they will freeze
        return freeze


    def sample(self):
        time_passed = 0.0
        # Set up a progress bar
        pbar = tqdm(total=self.N, desc="Sticky Zig-Zag time", unit="iter")

        while self.iteration <= self.N:
            n = self.iteration

            # First look at our active set
            n_active = sum(self.active).item()
            n_frozen = sum(~self.active).item()

            # Current position and velocity
            vel = self.Velocity[(n - 1), :]  # Current velocity
            pos = self.Position[(n - 1), :] + vel * time_passed  # Current position

            # Define a single-argument function for Brent
            rate_time = partial(self.neg_rate, pos0=pos, vel=vel)

            # Update our freezing times
            self.freezing = self.freezing_time(pos,vel)
            self.freezing = np.where(self.freezing > 0, self.freezing, np.inf)

            # Optimize bound on rate
            x_star = brent(rate_time, 0, self.t_max)  # Find time point at which rate is maximum
            if (x_star > self.t_max):
                x_star = self.t_max
            lambda_max = -rate_time(x_star)  # Rate at this time, basically a flat bound
            # Compute event time
            tau_star = np.random.exponential(1.0/lambda_max,1)
            # Thinning
            u = np.random.random()
            lambda_star = -rate_time(tau_star)
            p = lambda_star / lambda_max
            acc = (u <= p)

            # Okay, so if we have an accepted time, we need to handle three separate cases
            #   i) Regular rejection step
            #  ii) Freezing event
            # iii) Thawing event
            # Set up flags for these
            regular_event = False
            thawing_event = False
            freezing_event = False
            # Is there a regular event
            if acc and (tau_star < self.t_max):
                regular_event = True
            else:
                tau_star = self.t_max  # tau_star now just becomes the end of the interval

            # Are there freezing or thawing events before this?
            # Find times for next freezing and thawing events
            next_freeze = self.freezing[self.active].min()
            next_thaw = self.thawing[~self.active].min(initial=np.inf)

            if min(next_freeze, next_thaw) < tau_star:
                # Handle the freeze event
                if next_freeze < next_thaw:
                    freezing_event = True
                    regular_event = False
                    thawing_event = False
                    tau_star = next_freeze

                # Handle the thaw event
                elif next_thaw < next_freeze:
                    thawing_event = True
                    regular_event = False
                    freezing_event = False
                    tau_star = next_thaw

            # Now we handle the events
            # If regular event:
            if regular_event:
                # Which component is flipping velocity?
                rates = np.maximum(0,vel*self.grad_target(pos + vel*tau_star))
                # NB note that gamma should only enter on active rates!
                rates[self.active] += self.gamma
                rates = np.asarray(rates).astype('float64')
                rates = rates / np.sum(rates)
                i0 = np.argmax(np.random.multinomial(1, rates)).item()
                # Setting new values for positions and parameters
                self.Position[n, :] = pos + vel * tau_star
                self.Time[n] = self.Time[(n-1)] + time_passed + tau_star.item()
                # Velocity now flips!
                if not self.active[i0]:
                    print("something is horribly wrong")
                    break
                vel[i0] *= -1.0  # Velocity flip!
                self.Velocity[n, :] = vel

                pbar.update(1)
                # Increasing iteration and setting the local clock to zero again
                self.iteration += 1
                time_passed = 0.0

                # Adapt t_max
                self.t_max *= 0.96

            # If there is a thawing event
            elif thawing_event:
                i_thaw = np.argmin(np.where(~self.active, self.thawing, np.inf))
                self.Position[n, :] = pos + vel * tau_star
                self.Time[n] = self.Time[(n-1)] + time_passed + tau_star
                self.Velocity[n, :] = vel
                self.Velocity[n, i_thaw] = self.vel_at_freeze[i_thaw]  # set velocity for the thawed component
                self.active[i_thaw] = True  # set component as active
                self.thawing[i_thaw] = np.inf  # this is no longer frozen

                pbar.update(1)

                # Increasing iteration and setting the local clock to zero again
                self.iteration += 1
                time_passed = 0.0

                # If there is a freezing event
            elif freezing_event:
                i_freeze = np.argmin(np.where(self.active, self.freezing, np.inf))
                self.Position[n, :] = pos + vel * tau_star
                self.Time[n] = self.Time[(n-1)] + time_passed + tau_star
                self.Velocity[n, :] = vel
                self.Position[n, i_freeze] = 0.0  # set position for frozen component
                self.Velocity[n, i_freeze] = 0.0  # set velocity for frozen component
                self.vel_at_freeze[i_freeze] = vel[i_freeze]  # store the velocity it had before it froze
                self.active[i_freeze] = False  # set component as inactive
                self.thawing[i_freeze] = np.random.exponential(1.0 / self.kappa[i_freeze])  # How long will it stay frozen?

                pbar.update(1)

                # Increasing iteration and setting the local clock to zero again
                self.iteration += 1
                time_passed = 0.0

            # If nothing happens we increase the time passed by the tmax
            if not (regular_event or freezing_event or thawing_event):
                time_passed += self.t_max
                # Adapt t_max
                self.t_max *= 1.01

            # And in any case, thawing clocks tick down
            self.thawing = self.thawing - tau_star

        print("Time passed: " + str(self.Time[n]))
        pbar.close()