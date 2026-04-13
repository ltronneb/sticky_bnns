import numpy as np
from functools import partial
from tqdm import tqdm
import time as _time
import pandas as pd

from sazz.samplers.boomerang_sampler.BoomerangSampler import BoomerangSampler
from sazz.utils.utils import brent


class FactorizedBoomerangSampler(BoomerangSampler):
    """
    Factorized Boomerang sampler.
    """
    # --- Per-component rate ---
    def neg_rate_i(self, t, x, v, i):
        """Negative of component-i rate: -max(0, v_i(t) * gradU_i(x(t)))"""
        x_t, v_t = self.trajectory(t, x, v)
        return -(v_t[i] * self.gradU(x_t)[i])

    # --- Per-component reflection ---
    def reflect_velocity_i(self, v, gradU, i):
        """
        Reflect only component i:  v_i -> v_i - 2 * v_i * gradU_i / (Sigma_ii * gradU_i^2) * Sigma_ii * gradU_i
        
        For diagonal Sigma this simplifies to flipping the sign of v_i.
        For general Sigma we use the full formula restricted to component i.
        """
        g_i = gradU[i]
        if abs(g_i) < 1e-14:
            return v
        
        v_new = v.copy()
        # General formula: v - 2 * (v_i * g_i) / (g_i * Sigma_{ii} * g_i) * Sigma[:, i] * g_i
        # = v - 2 * v_i / (Sigma_{ii} * g_i) * Sigma[:, i] * g_i
        # But for factorized boomerang, the reflection for component i is:
        # v_new = v - 2 * (v_i * g_i) / (Sigma_{ii} * g_i^2) * Sigma[:, i] * g_i
        numerator = v[i] * g_i
        denom = g_i * (self.Sigma[i, :] @ (g_i * np.eye(self.D)[i]))
        # Simplifies to: denom = g_i^2 * Sigma_{ii}
        denom = g_i**2 * self.Sigma[i, i]
        
        if abs(denom) < 1e-14:
            return v
        
        v_new -= 2.0 * numerator / denom * (self.Sigma[:, i] * g_i)
        return v_new

    def sample_auto(self, diagnostics=True):
        """
        Factorized Boomerang: D independent Poisson clocks via Brent.
        """
        self.Position[0, :] = np.random.normal(0, 1, size=self.D)
        self.Velocity[0, :] = self.Sigma_sqrt @ np.random.randn(self.D)
        self.Time[0] = 0.0

        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

        time_passed = 0.0
        pbar = tqdm(total=self.N, desc="Factorized Boomerang", unit="iter")

        diag_log = []
        grad_evals = 0

        while self.iteration < self.N:
            _iter_start = _time.perf_counter()
            n = self.iteration
            pos, vel = self.trajectory(
                time_passed, self.Position[n - 1, :], self.Velocity[n - 1, :]
            )
            horizon = min(self.t_max, dt_refresh)

            # --- Factorized: Brent on each component ---
            tau_stars = np.full(self.D, np.inf)
            lambda_maxes = np.zeros(self.D)
            total_rate_evals = 0

            for i in range(self.D):
                rate_i = partial(self.neg_rate_i, x=pos, v=vel, i=i)
                x_star_i, stats_i = brent(rate_i, 0, horizon, diagnostics=diagnostics)
                lam_i = max(-rate_i(x_star_i), 0.0)
                lambda_maxes[i] = lam_i
                total_rate_evals += stats_i['rate_evals']

                if lam_i > 1e-14:
                    tau_stars[i] = -np.log(np.random.rand()) / lam_i

            grad_evals += total_rate_evals

            # Winning component: smallest proposed time
            i_win = np.argmin(tau_stars)
            tau_star = tau_stars[i_win]

            row = {
                'rate_evals': total_rate_evals,
                'horizon': horizon,
                'time': self.current_time,
                'event_type': None,
                'accepted': None,
                'component': None,
                'wall_seconds': None,
            }

            if tau_star >= horizon:
                row['event_type'] = 'no_event'
                row['accepted'] = False
                time_passed += horizon
                self.current_time += horizon
                dt_refresh -= horizon
            else:
                row['event_type'] = 'bounce'
                row['component'] = i_win

                # Accept/reject for the winning component
                rate_i_win = partial(self.neg_rate_i, x=pos, v=vel, i=i_win)
                lambda_star = max(-rate_i_win(tau_star), 0.0)
                p = lambda_star / lambda_maxes[i_win]
                u = np.random.rand()

                grad_evals += 1
                row['rate_evals'] += 1

                if u <= p:
                    row['accepted'] = True
                    self.current_time += tau_star

                    pos_prop, vel_prop = self.trajectory(
                        time_passed + tau_star,
                        self.Position[n - 1, :],
                        self.Velocity[n - 1, :],
                    )
                    gradU_prop = self.gradU(pos_prop)
                    self.Position[n, :] = pos_prop
                    self.Velocity[n, :] = self.reflect_velocity_i(vel_prop, gradU_prop, i_win)
                    self.Time[n] = self.Time[n - 1] + time_passed + tau_star

                    grad_evals += 1
                    row['rate_evals'] += 1

                    self.iteration += 1
                    time_passed = 0.0
                    dt_refresh -= tau_star
                    pbar.update(1)
                else:
                    row['accepted'] = False
                    time_passed += tau_star
                    self.current_time += tau_star
                    dt_refresh -= tau_star

            # --- Refresh ---
            if dt_refresh <= 1e-14:
                row['wall_seconds'] = _time.perf_counter() - _iter_start
                diag_log.append(row)

                refresh_row = {
                    'rate_evals': 0,
                    'horizon': 0.0,
                    'time': self.current_time,
                    'event_type': 'refresh',
                    'accepted': None,
                    'component': None,
                    'wall_seconds': 0.0,
                }
                if self.iteration < self.N:
                    n = self.iteration
                    pos_ref, _ = self.trajectory(
                        time_passed,
                        self.Position[n - 1, :],
                        self.Velocity[n - 1, :],
                    )
                    self.Position[n, :] = pos_ref
                    self.Velocity[n, :] = self.Sigma_sqrt @ np.random.randn(self.D)
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

        # --- Diagnostics ---
        df = pd.DataFrame(diag_log)
        self.diagnostics_df = df

        n_accept = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)].shape[0]
        n_reject = df[(df['event_type'] == 'bounce') & (df['accepted'] == False)].shape[0]

        print("\n=== Sampler Diagnostics ===")
        print(f"Total gradient evals: {grad_evals}")
        print(f"Grad evals per skeleton point: {grad_evals / self.N:.1f}")

        print("\n=== Thinning Diagnostics ===")
        bounce_df = df[df['event_type'] == 'bounce']
        if len(bounce_df) > 0:
            print(f"Rate evals per call: {df['rate_evals'].mean():.1f} mean, {df['rate_evals'].max()} max")
        print(f"Accept/Reject: {n_accept}/{n_reject}")

        # Per-component bounce counts
        if n_accept > 0:
            accepted = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)]
            print("Bounces per component:", dict(accepted['component'].value_counts().sort_index()))

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
