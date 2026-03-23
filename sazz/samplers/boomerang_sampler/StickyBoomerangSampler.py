import numpy as np
from functools import partial
from autograd import hessian
from scipy.optimize import minimize
from scipy.linalg import cholesky
import autograd.numpy as anp
from tqdm import tqdm

from sazz.utils.utils import brent
from sazz.samplers.boomerang_sampler.BoomerangSampler import BoomerangSampler


class StickyBoomerangSampler(BoomerangSampler):
    """
    Sticky Boomerang sampler
    """
    def __init__(self, E, N:int, D:int, grad_target, kappa=1.0,
                 refresh_rate=1.0, t_max: float = 0.5,
                 ):
        super().__init__(
            E=E, N=N, D=D, grad_target=grad_target,
            refresh_rate=refresh_rate, t_max=t_max,
        )
        
        # --- Sticky specifics ---
        kappa_arr = np.array([kappa])
        self.kappa = np.full(D, kappa_arr)
        
        self.frozen_mask = np.zeros(D, dtype=bool)
        self.frozen_velocity = np.zeros(D, dtype=float)


    # --- Sticky specific dynamics ---
    def trajectory_sticky(self, t, x, v):
        """
        Same as parent trajectory, but frozen coordinates stay at 0.
        """
        x_t, v_t = self.trajectory(t, x, v)
        # Overwrite frozen coordinates
        x_t[self.frozen_mask] = 0.0
        v_t[self.frozen_mask] = 0.0
        return x_t, v_t
    
    def freeze(self, i, v):
        """
        Freeze coordinate i: store its velocity, then set x_i = v_i = 0.
        """
        self.frozen_mask[i] = True
        self.frozen_velocity[i] = v[i]
        
    def thaw(self, i):
        """
        Thaw coordinate i: restore its stored velocity (non-reversible jump).
        Position stays at 0.
        """
        self.frozen_mask[i] = False
        
    def reflect_velocity_sticky(self, v, gradU):
        """
        Boomerang reflection on the active subspace only. Fall back to parent if no frozen coordinates
        """
        if not np.any(self.frozen_mask):
            return self.reflect_velocity(v, gradU)

        active = ~self.frozen_mask
        v_new = v.copy()

        # Restrict to active block
        v_a = v[active]
        g_a = gradU[active]
        Sigma_a = self.Sigma[np.ix_(active, active)]

        rate = float(np.dot(v_a, g_a))
        Sigma_g = Sigma_a @ g_a
        denom = float(np.dot(g_a, Sigma_g))

        if denom <= 1e-14:
            print("denom was zero")
            return v_new

        v_new[active] = v_a - 2.0 * rate / denom * Sigma_g
        v_new[self.frozen_mask] = 0.0
        return v_new

    def next_hitting_event(self, x, v):
        """
        First time an active coordinate hits x_i = 0, and which coordinate
        """
        t_hit = np.inf
        i_hit = None

        for i in np.flatnonzero(~self.frozen_mask):
            a = float(x[i] - self.x_ref[i])
            b = float(v[i])
            c = float(-self.x_ref[i])

            # Solve: a cos(t) + b sin(t) = c
            R = np.hypot(a, b)
            if R < 1e-14:
                continue
            if abs(c) > R + 1e-14:
                continue  # no solution exists

            phi = np.arctan2(b, a)
            delta = np.arccos(np.clip(c / R, -1.0, 1.0))

            # Collect smallest positive root
            t_i = np.inf
            for base in (phi - delta, phi + delta):
                # Reduce to (0, 2*pi]
                candidate = base % (2.0 * np.pi)
                if candidate < 1e-10:
                    candidate += 2.0 * np.pi
                t_i = min(t_i, candidate)

            if t_i < t_hit:
                t_hit = t_i
                i_hit = i

        return t_hit, i_hit
    
    def sample_thaw_event(self):
        """
        Sample when and which frozen coordinate thaws.
        Each frozen coordinate i has thaw rate kappa_i * |v_i_stored|.
        """
        frozen_idx = np.flatnonzero(self.frozen_mask)

        # Nothing frozen — no thaw event possible
        if len(frozen_idx) == 0:
            return np.inf, None

        # Individual thaw rates
        rates = self.kappa[frozen_idx] * np.abs(self.frozen_velocity[frozen_idx])
        total_rate = float(np.sum(rates))

        # All frozen velocities were ~zero — stuck forever
        if total_rate < 1e-14:
            return np.inf, None

        # Time until first thaw: min of competing exponentials
        t_thaw = np.random.exponential(1.0 / total_rate)

        # Which coordinate wins: proportional to its rate
        probs = rates / total_rate
        i_thaw = int(np.random.choice(frozen_idx, p=probs))

        return t_thaw, i_thaw


    def sample_auto(self):
        """
        Sticky Automatic Boomerang sampler.
        
        Four competing clocks determine what happens next:
            1. Bounce proposal (Poisson thinning on local rate bound)
            2. Velocity refresh (exponential clock)
            3. Coordinate freeze (deterministic: orbit crosses x_i = 0)
            4. Coordinate thaw (exponential clock per frozen coordinate)
        
        Whichever fires first within [0, t_max] wins.
        """
        self.Position[0,:] = np.random.normal(0,1,size=self.D)
        self.Velocity[0,:] = self.Sigma_sqrt @ np.random.randn(self.D)
        self.Time[0] = 0.0
        #self.frozen_mask[:] = False
        #self.frozen_velocity[:] = 0.0
 
        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
        
        time_passed = 0.0
        pbar = tqdm(total=self.N, desc="Sticky Boomerang time", unit="iter")
        while self.iteration < self.N:
            n = self.iteration
            pos, vel = self.trajectory_sticky(time_passed, self.Position[(n-1), :], self.Velocity[(n-1), :])
            
            dt_hit, i_hit = self.next_hitting_event(pos, vel)
            dt_thaw , i_thaw = self.sample_thaw_event()
            horizon = min(self.t_max, dt_refresh, dt_hit, dt_thaw)
            
            rate_time = partial(self.neg_rate, x=pos, v=vel)
            x_star = brent(rate_time, 0, horizon)
            lambda_max = max(-rate_time(x_star), 0.0) 
            
            regular_event_proposal = False
            thawing_event = False
            freezing_event = False
            
            if lambda_max <= 1e-14:
                tau_star = np.inf
            else:
                tau_star = -np.log(np.random.rand()) / lambda_max
                
            if tau_star < horizon: # Regular event
                regular_event_proposal = True
                u = np.random.random()
                lambda_star = max(-rate_time(tau_star), 0.0)
                p = lambda_star / lambda_max
                
                if u <= p:
                    # Accepted velocity bounce
                    self.current_time += tau_star
                    
                    pos_prop, vel_prop = self.trajectory_sticky(
                        time_passed + tau_star,
                        self.Position[(n-1), :],
                        self.Velocity[(n-1), :]
                    )
                    self.Position[n, :] = pos_prop
                    self.Velocity[n, :] = self.reflect_velocity_sticky(vel_prop, self.gradU(pos_prop))
                    self.Time[n] = self.Time[(n-1)] + time_passed + tau_star
                    
                    self.iteration += 1
                    time_passed = 0.0
                    dt_refresh -= tau_star
                    pbar.update(1)
                else:
                    time_passed += tau_star
                    self.current_time += tau_star
                    dt_refresh -= tau_star
            else: # Either thaw, freeze or refresh fired and became the horizon
                time_passed += horizon
                self.current_time += horizon
                dt_refresh -= horizon
                
                tol = 1e-12
                
                if abs(horizon - dt_hit) < tol and i_hit is not None:
                    freezing_event = True
                    pos_now, vel_now = self.trajectory_sticky(
                        time_passed, self.Position[n-1, :], self.Velocity[n-1, :]
                    )
                    self.freeze(i_hit, vel_now)
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    self.Position[n, :] = pos_now
                    self.Velocity[n, :] = vel_now
                    self.Time[n] = self.Time[n-1] + time_passed

                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)
                    
                elif abs(horizon - dt_thaw)<tol and i_thaw is not None:
                    thawing_event = True
                    pos_now, vel_now = self.trajectory_sticky(
                        time_passed, self.Position[n-1, :], self.Velocity[n-1, :]
                    )
                    self.thaw(i_thaw)
                    #pos_now[i_thaw] = 0.0
                    vel_now[i_thaw] = self.frozen_velocity[i_thaw]
                    
                    self.Position[n, :] = pos_now
                    self.Velocity[n, :] = vel_now
                    self.Time[n] = self.Time[n-1] + time_passed
                    
                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)
                
            if dt_refresh <= 1e-14:
                if self.iteration < self.N:
                    n = self.iteration
                    pos_ref, _ = self.trajectory_sticky(
                        time_passed,
                        self.Position[(n-1), :],
                        self.Velocity[(n-1), :]
                    )
                    
                    v_refreshed = np.zeros(self.D)
                    active = ~self.frozen_mask
                    if np.any(active):
                        Sigma_a = self.Sigma[np.ix_(active, active)]
                        chol_a = np.linalg.cholesky(
                            Sigma_a + 1e-12 * np.eye(np.sum(active))
                        )
                        v_refreshed[active] = chol_a @ np.random.randn(np.sum(active))
                    
                    self.Position[n, :] = pos_ref
                    self.Velocity[n, :] = v_refreshed
                    self.Time[n] = self.Time[(n-1)] + time_passed
                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
            pbar.set_postfix_str(f"time={self.current_time:.3f}", refresh=False)

        print("Time passed: " + str(self.Time[self.iteration - 1]))
        pbar.close()
            
            

            



    def _default_temperature(self, t, T_min=0.1):
        """
        Exponential ramp for tempering:
        Starts flat (T_min) and increase to 1.0
        """
        return 1.0 - (1.0 - T_min) * np.exp(-t / self.t0)
 
