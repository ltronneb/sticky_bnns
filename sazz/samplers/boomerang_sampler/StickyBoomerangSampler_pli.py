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


class StickyBoomerangSampler_pli(BoomerangSampler):
    """
    Sticky Boomerang sampler
    """
    def __init__(self, E, N:int, D:int, grad_target, kappa=1.0,
                 refresh_rate=0.1, t_max: float = 1.0,
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
        self.thaw_deadline = np.full(D, np.inf)


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
    
    def freeze(self, i, v, current_time):
        self.frozen_mask[i] = True
        self.frozen_velocity[i] = v[i]
        rate_i = self.kappa[i] * abs(v[i])
        if rate_i > 1e-14:
            self.thaw_deadline[i] = current_time + np.random.exponential(1.0 / rate_i)
        else:
            self.thaw_deadline[i] = np.inf
            
    def thaw(self, i):
        self.frozen_mask[i] = False
        self.thaw_deadline[i] = np.inf
        
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
    
    def next_thaw_event(self):
        frozen_idx = np.flatnonzero(self.frozen_mask)
        if len(frozen_idx) == 0:
            return np.inf, None
        i_min = frozen_idx[np.argmin(self.thaw_deadline[frozen_idx])]
        return self.thaw_deadline[i_min], i_min

    def rate_func(self, t, x, v):
        """
        Boomerang negative rate
        """
        x_t, v_t = self.trajectory_sticky(t, x, v)
        inner = anp.dot(v_t, self.gradU(x_t))
        return inner
    
    def sample_auto(self):
        self.Position[0, :] = np.random.normal(0, 1, size=self.D)
        self.Velocity[0, :] = self.Sigma_sqrt @ np.random.randn(self.D)
        self.Time[0] = 0.0
        self.thaw_deadline[:] = np.inf  # reset schedule

        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

        time_passed = 0.0
        pbar = tqdm(total=self.N, desc="Sticky Boomerang PLI time", unit="iter")
        
        diag_log = []
        grad_evals = 0
        
        while self.iteration < self.N:
            _iter_start = _time.perf_counter()
            n = self.iteration
            pos, vel = self.trajectory_sticky(
                time_passed, self.Position[n - 1, :], self.Velocity[n - 1, :]
            )

            dt_hit, i_hit = self.next_hitting_event(pos, vel)

            # --- thaw: lookup from schedule instead of sampling ---
            T_thaw_abs, i_thaw = self.next_thaw_event()
            dt_thaw = T_thaw_abs - self.current_time  # time until thaw from *now*
            if dt_thaw < 0:
                dt_thaw = 0.0  # safety clamp

            #horizon = min(self.t_max, dt_refresh, dt_hit, dt_thaw)
            horizon = min(dt_refresh, dt_hit, dt_thaw)
            rate_fn = partial(self.rate_func, x=pos, v=vel)

            tau_star, stats = piecewise_thinning(rate_fn, horizon)
            
            grad_evals += stats['rate_evals']

            n_frozen = self.frozen_mask.sum()
            n_active = self.D - n_frozen

            row = {
                'rate_evals': stats['rate_evals'],
                'horizon': horizon,
                'time': self.current_time,
                'event_type': None,
                'accepted': None,
                'n_frozen': n_frozen,
                'n_active': n_active,
                'sparsity': n_frozen / self.D,
                'wall_seconds': None,
                # PLI-specific
                'proposals': stats.get('proposals', 0),
                'bound_violations': stats.get('bound_violations', 0),
                'max_ratio': stats.get('max_ratio', 0.0),
            }

            if tau_star < horizon:  # Regular event (bounce proposal)
                row['event_type'] = 'bounce'
                row['accepted'] = True  # PLI only returns accepted events
                # Accepted bounce
                self.current_time += tau_star
                pos_prop, vel_prop = self.trajectory_sticky(
                    time_passed + tau_star,
                    self.Position[n - 1, :],
                    self.Velocity[n - 1, :],
                )
                self.Position[n, :] = pos_prop
                self.Velocity[n, :] = self.reflect_velocity_sticky(
                    vel_prop, self.gradU(pos_prop)
                )
                self.Time[n] = self.Time[n - 1] + time_passed + tau_star
                
                grad_evals += 1
                row['rate_evals'] += 1

                self.iteration += 1
                time_passed = 0.0
                dt_refresh -= tau_star
                pbar.update(1)

            else:  # horizon fired: freeze, thaw, or refresh
                time_passed += horizon
                self.current_time += horizon
                dt_refresh -= horizon

                tol = 1e-12

                if abs(horizon - dt_hit) < tol and i_hit is not None:
                    # --- Freeze: draw thaw deadline here ---
                    row['event_type'] = 'freeze'
                    row['accepted'] = None
                    row['frozen_coord'] = i_hit
                    pos_now, vel_now = self.trajectory_sticky(
                        time_passed, self.Position[n - 1, :], self.Velocity[n - 1, :]
                    )
                    self.freeze(i_hit, vel_now, self.current_time)
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    self.Position[n, :] = pos_now
                    self.Velocity[n, :] = vel_now
                    self.Time[n] = self.Time[n - 1] + time_passed

                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                elif abs(horizon - dt_thaw) < tol and i_thaw is not None:
                    row['event_type'] = 'thaw'
                    row['accepted'] = None
                    row['thawed_coord'] = i_thaw
                    # --- Thaw: just clear the deadline ---
                    pos_now, vel_now = self.trajectory_sticky(
                        time_passed, self.Position[n - 1, :], self.Velocity[n - 1, :]
                    )
                    self.thaw(i_thaw)
                    vel_now[i_thaw] = self.frozen_velocity[i_thaw]

                    self.Position[n, :] = pos_now
                    self.Velocity[n, :] = vel_now
                    self.Time[n] = self.Time[n - 1] + time_passed

                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

            if dt_refresh <= 1e-14:
                row['wall_seconds'] = _time.perf_counter() - _iter_start
                diag_log.append(row)

                refresh_row = {
                    'rate_evals': 0,
                    'horizon': 0.0,
                    'time': self.current_time,
                    'event_type': 'refresh',
                    'accepted': None,
                    'n_frozen': self.frozen_mask.sum(),
                    'n_active': self.D - self.frozen_mask.sum(),
                    'sparsity': self.frozen_mask.sum() / self.D,
                    'wall_seconds': 0.0,
                    'proposals': 0,
                    'bound_violations': 0,
                    'max_ratio': 0.0,
                }
                if self.iteration < self.N:
                    n = self.iteration
                    pos_ref, _ = self.trajectory_sticky(
                        time_passed,
                        self.Position[n - 1, :],
                        self.Velocity[n - 1, :],
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
                    self.Time[n] = self.Time[n - 1] + time_passed
                    self.iteration += 1
                    time_passed = 0.0
                    pbar.update(1)
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
                diag_log.append(refresh_row)
                pbar.set_postfix_str(f"time={self.current_time:.3f}", refresh=False)
            row['wall_seconds'] = _time.perf_counter() - _iter_start
            pbar.set_postfix_str(f"time={self.current_time:.3f}", refresh=False)
            diag_log.append(row)
            
        df = pd.DataFrame(diag_log)
        self.diagnostics_df = df

        n_accept = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)].shape[0]

        print("\n=== Sampler Diagnostics ===")
        print(f"Total gradient evals: {grad_evals}")
        print(f"Grad evals per skeleton point: {grad_evals / self.N:.1f}")
        print(f"Mean sparsity (fraction frozen): {df['sparsity'].mean():.3f}")
        print(f"Max simultaneous frozen: {df['n_frozen'].max()} / {self.D}")

        print("\n=== Thinning Diagnostics (PLI) ===")
        print(f"Bound violations: {df['bound_violations'].sum()} across {(df['bound_violations'] > 0).sum()} calls")
        print(f"Max ratio: {df['max_ratio'].max():.4f}")
        print(f"Rate evals per call: {df['rate_evals'].mean():.1f} mean, {df['rate_evals'].max()} max")
        print(f"Proposals per call: {df['proposals'].mean():.1f} mean, {df['proposals'].max()} max")
        print(f"Accepted bounces: {n_accept}")

        print("\n=== Event-type breakdown ===")
        for etype in ['bounce', 'freeze', 'thaw', 'refresh', 'no_event']:
            sub = df[df['event_type'] == etype]
            if len(sub) > 0:
                print(f"  {etype:10s}: {len(sub):5d} events, "
                    f"mean horizon={sub['horizon'].mean():.4f}")

        print("\n=== Timing ===")
        print(f"Mean wall-seconds per iteration: {df['wall_seconds'].mean():.6f}")
        print(f"Total wall-seconds: {df['wall_seconds'].sum():.2f}")

        print(f"\nTime passed: {self.Time[self.iteration - 1]}")
        pbar.close()
    
    def _default_temperature(self, t, T_min=0.1):
        """
        Exponential ramp for tempering:
        Starts flat (T_min) and increase to 1.0
        """
        return 1.0 - (1.0 - T_min) * np.exp(-t / self.t0)
 
