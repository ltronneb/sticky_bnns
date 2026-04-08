import numpy as np
from functools import partial
from autograd import hessian
from scipy.optimize import minimize
from scipy.linalg import cholesky
import autograd.numpy as anp
from tqdm import tqdm

import pandas as pd

from sazz.utils.utils import brent

class BoomerangSampler:
    """
    Boomerang sampler
    """
    def __init__(self, E, N:int, D:int, grad_target, refresh_rate=1.0, 
                 t_max: float = 0.5,
                 gamma: float = 0.01,
                 temper: bool=False, temperature=None,
                 t0: float = 100.0):
        self.E = E
        self.N = N
        self.D = D
        self.Position = np.zeros((N,D))
        self.Velocity = np.zeros((N,D))
        self.Time = np.zeros(N)
        self.gamma = gamma
        self.grad_target = grad_target
        self.t_max = t_max
        
        self.iteration = 1
        # Deal with temperature
        if temperature is None:
            self.temperature = self._default_temperature
        else:
            self.temperature = temperature
        self.temper = temper
        # Also need a global thing for the current time
        self.current_time = 0.0
        self.t0 = t0
        
        # --- Boomerang specific ---
        self.refresh_rate = refresh_rate
        
        self.hessE = hessian(E)
        self.x_ref = None
        self.Sigma_inv = None
        self.Sigma = None
        self.Sigma_sqrt = None

    # Preprocessing
    def preprocess(self, x0=None, method="diagonal", x_ref=None, Sigma_inv=None, jitter=1e-8):
        """
        Construct the Gaussian reference measure used by Boomerang
        """
        if x0 is None:
            x0 = np.zeros(self.D)
        if method == "diagonal":
            res = minimize(self.E, x0, jac=lambda x: np.array(self.grad_target(x)))
            
            x_ref = np.array(res.x, dtype=float)
            H = np.array(self.hessE(x_ref), dtype=float)
            # Clip zero eigenvalues
            diag_H = np.clip(np.diag(H), jitter, None)
            
            Sigma_inv = np.diag(diag_H)

        elif method == "manual":
            x_ref = np.array(x_ref, dtype=float)
            Sigma_inv = np.array(Sigma_inv, dtype=float)
            # Force exact symmetry to keep Cholesky stable
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
            Sigma_inv = Sigma_inv + jitter * np.eye(self.D)

        else:
            raise ValueError("Unknown method. Use 'diagonal' or 'manual'.")

        Sigma = np.linalg.inv(Sigma_inv)
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma_sqrt = cholesky(Sigma, lower=True)

        self.x_ref = x_ref
        self.Sigma_inv = Sigma_inv
        self.Sigma = Sigma
        self.Sigma_sqrt = Sigma_sqrt
        
    # --- Boomerang dynamics and rates
    def gradU(self, x):
        """
        Gradient accounting for reference measure
        """
        return self.grad_target(x) - self.Sigma_inv@(x-self.x_ref)
    
    def trajectory(self, t, x, v):
        """
        Boomerang trajectory
        """
        x_t = self.x_ref + (x - self.x_ref) * anp.cos(t) + v * anp.sin(t)
        v_t = -(x - self.x_ref) * anp.sin(t) + v * anp.cos(t)
        return x_t, v_t

    def neg_rate(self, t, x, v):
        """
        Boomerang negative rate
        """
        x_t, v_t = self.trajectory(t, x, v)
        inner = anp.dot(v_t, self.gradU(x_t))
        return -inner

    def reflect_velocity(self, v, gradU):
        """
        Boomerang reflection
        """
        rate = float(np.dot(v, gradU))
        skew = self.Sigma_sqrt.T @ gradU
        denom = float(np.dot(skew, skew))

        return v - 2.0 * rate / denom * (self.Sigma_sqrt @ skew)

    def sample_auto(self, diagnostics=True):
        """
        Automatic Boomerang sampler
        """
        self.Position[0,:] = np.random.normal(0,1,size=self.D)
        self.Velocity[0,:] = self.Sigma_sqrt @ np.random.randn(self.D)
        self.Time[0] = 0.0
 
        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
        
        time_passed = 0.0
        pbar = tqdm(total=self.N, desc="Boomerang time", unit="iter")
        
        diag_log = []
        event_counts = {'bounce': 0, 'refresh': 0, 'max_iter': 0}
        grad_evals, accept, reject = 0, 0, 0
        
        while self.iteration < self.N:
            n = self.iteration
            pos, vel = self.trajectory(time_passed, self.Position[(n-1), :], self.Velocity[(n-1), :])
            horizon = min(self.t_max, dt_refresh) # Need this for refreshments
            
            rate_time = partial(self.neg_rate, x=pos, v=vel)
            x_star, stats = brent(rate_time, 0, horizon, diagnostics=diagnostics)
            
            lambda_max = max(-rate_time(x_star), 0.0) 
            
            if lambda_max <= 1e-14:                  
                tau_star = np.inf
            else:
                tau_star = -np.log(np.random.rand()) / lambda_max
                
            grad_evals += stats['rate_evals']

            stats['horizon'] = horizon
            stats['time'] = self.current_time
                
            if tau_star >= horizon:
                stats['event_type'] = 'no_event'
                # No proposal in this window
                time_passed += horizon
                self.current_time += horizon
                dt_refresh -= horizon
            else:
                stats['event_type'] = 'bounce'
                event_counts['bounce'] += 1
                # Proposal happened
                u = np.random.random()
                lambda_star = max(-rate_time(tau_star), 0.0)
                p = lambda_star / lambda_max
                
                grad_evals += 1

                if u <= p:
                    accept += 1
                    # Accepted velocity bounce
                    self.current_time += tau_star
                    
                    pos_prop, vel_prop = self.trajectory(
                        time_passed + tau_star,
                        self.Position[(n-1), :],
                        self.Velocity[(n-1), :]
                    )
                    self.Position[n, :] = pos_prop
                    self.Velocity[n, :] = self.reflect_velocity(vel_prop, self.gradU(pos_prop))
                    self.Time[n] = self.Time[(n-1)] + time_passed + tau_star
                    
                    grad_evals += 1
                    
                    self.iteration += 1
                    time_passed = 0.0
                    dt_refresh -= tau_star
                    pbar.update(1)
                else:
                    reject += 1
                    # Rejected — advance by tau_star
                    time_passed += tau_star
                    self.current_time += tau_star
                    dt_refresh -= tau_star
 
            # Refresh velocity regularly
            if dt_refresh <= 1e-14:
                stats['event_type'] = 'refresh'
                event_counts['refresh'] += 1
                if self.iteration < self.N:
                    n = self.iteration
                    pos_ref, _ = self.trajectory(
                        time_passed,
                        self.Position[(n-1), :],
                        self.Velocity[(n-1), :]
                    )
                    self.Position[n, :] = pos_ref
                    self.Velocity[n, :] = self.Sigma_sqrt @ np.random.randn(self.D)
                    self.Time[n] = self.Time[(n-1)] + time_passed
                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
            
            pbar.set_postfix_str((f"time={self.current_time:.3f}"),refresh=False)
            diag_log.append(stats)

        df = pd.DataFrame(diag_log)
        print("\n=== Sampler Diagnostics ===")
        print(f"Events: {event_counts}")
        print(f"Total gradient evals: {grad_evals}")
        print(f"Grad evals per skeleton point: {grad_evals / self.N:.1f}")

        print("\n=== Thinning Diagnostics ===")
        print(f"Brent gradient per call: {df['rate_evals'].mean():.1f} mean, {df['rate_evals'].max()} max")
        print(f"Accept/Reject: {accept}/{reject}")

        print("\n=== Event-type breakdown ===")
        for etype in ['bounce', 'refresh']:
            sub = df[df['event_type'] == etype]
            if len(sub) > 0:
                print(f"  {etype:8s}: {len(sub):5d} events, "
                    f"mean horizon={sub['horizon'].mean():.4f}, "
                    )

        self.diagnostics_df = df
        
        print("Time passed: " + str(self.Time[n]))
        pbar.close()

    def _default_temperature(self, t, T_min=0.1):
        """
        Exponential ramp for tempering:
        Starts flat (T_min) and increase to 1.0
        """
        return 1.0 - (1.0 - T_min) * np.exp(-t / self.t0)
 
