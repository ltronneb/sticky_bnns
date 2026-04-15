import numpy as np
from functools import partial
from autograd import hessian
from scipy.optimize import minimize
from scipy.linalg import cholesky
import autograd.numpy as anp
from tqdm import tqdm
import time as _time

import pandas as pd

from sazz.utils.linear_envelope import piecewise_thinning
from sazz.samplers.boomerang_sampler.BoomerangSampler import BoomerangSampler

class BoomerangSampler_PLI(BoomerangSampler):
    """
    Boomerang sampler
    """

    def rate_func(self, t, x, v):
        """
        Boomerang negative rate
        """
        x_t, v_t = self.trajectory(t, x, v)
        inner = anp.dot(v_t, self.gradU(x_t))
        return inner

    def sample_auto(self, diagnostics=True):
        """
        Automatic Boomerang sampler
        """
        #self.Position[0,:] = np.random.normal(0,1,size=self.D)
        self.Position[0,:] = self.x_ref + self.Sigma_sqrt @ np.random.randn(self.D)
        self.Velocity[0,:] = self.Sigma_sqrt @ np.random.randn(self.D)
        self.Time[0] = 0.0
 
        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
        
        time_passed = 0.0
        pbar = tqdm(total=self.N, desc="Boomerang time", unit="iter")
        
        # diag_log = []
        # event_counts = {'bounce': 0, 'refresh': 0, 'max_iter': 0}
        # grad_evals, accept, reject = 0, 0, 0
        diag_log = []
        grad_evals = 0
        
        while self.iteration < self.N:
            _iter_start = _time.perf_counter()
            n = self.iteration
            pos, vel = self.trajectory(time_passed, self.Position[(n-1), :], self.Velocity[(n-1), :])
            #horizon = min(self.t_max, dt_refresh) # Need this for refreshments
            
            horizon = dt_refresh
            rate_fn = partial(self.rate_func, x=pos, v=vel)

            tau_star, stats = piecewise_thinning(rate_fn, horizon, diagnostics=True)
            
            grad_evals += stats['rate_evals']

            row = {
                'rate_evals': stats['rate_evals'],
                'horizon': horizon,
                'time': self.current_time,
                'event_type': None,
                'accepted': None,
                'wall_seconds': None,
                # PLI-specific
                'proposals': stats.get('proposals', 0),
                'bound_violations': stats.get('bound_violations', 0),
                'max_ratio': stats.get('max_ratio', 0.0),
            }
                
            if tau_star < horizon:
                row['event_type'] = 'bounce'
                row['accepted'] = True  # PLI only returns accepted events
                # Accepted bounce
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
                row['rate_evals'] += 1
                
                self.iteration += 1
                time_passed = 0.0
                dt_refresh -= tau_star
                pbar.update(1)
            else:
                time_passed += horizon
                self.current_time += horizon
                dt_refresh -= horizon
 
            # Refresh velocity regularly
            if dt_refresh <= 1e-14:
                row['wall_seconds'] = _time.perf_counter() - _iter_start
                diag_log.append(row)
                refresh_row = {
                    'rate_evals': 0,
                    'horizon': 0.0,
                    'time': self.current_time,
                    'event_type': 'refresh',
                    'accepted': None,
                    'wall_seconds': 0.0,
                    'proposals': 0,
                    'bound_violations': 0,
                    'max_ratio': 0.0,
                }
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
                diag_log.append(refresh_row)
                pbar.set_postfix_str((f"time={self.current_time:.3f}"), refresh=False)
            
            # pbar.set_postfix_str((f"time={self.current_time:.3f}"),refresh=False)
            # diag_log.append(stats)
            row['wall_seconds'] = _time.perf_counter() - _iter_start
            pbar.set_postfix_str((f"time={self.current_time:.3f}"), refresh=False)
            diag_log.append(row)

        df = pd.DataFrame(diag_log)
        self.diagnostics_df = df
        self.gradient_evals = grad_evals

        n_accept = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)].shape[0]
        
        if diagnostics:
            print("\n=== Sampler Diagnostics ===")
            print(f"Total gradient evals: {grad_evals}")
            print(f"Grad evals per skeleton point: {grad_evals / self.N:.1f}")

            print("\n=== Thinning Diagnostics (PLI) ===")
            print(f"Bound violations: {df['bound_violations'].sum()} across {self.N:.1f} calls")
            print(f"Max ratio: {df['max_ratio'].max():.4f}")
            print(f"Rate evals per call: {df['rate_evals'].mean():.1f} mean, {df['rate_evals'].max()} max")
            print(f"Proposals per call: {df['proposals'].mean():.1f} mean, {df['proposals'].max()} max")
            print(f"Accepted bounces: {n_accept}")

            print("\n=== Event-type breakdown ===")
            for etype in ['bounce', 'no_event', 'refresh']:
                sub = df[df['event_type'] == etype]
                if len(sub) > 0:
                    print(f"  {etype:10s}: {len(sub):5d} events, "
                        f"mean horizon={sub['horizon'].mean():.4f}")

            print("\n=== Timing ===")
            print(f"Mean wall-seconds per iteration: {df['wall_seconds'].mean():.6f}")
            print(f"Total wall-seconds: {df['wall_seconds'].sum():.2f}")

        print(f"\nTime passed: {self.Time[self.iteration - 1]}")
        pbar.close()
