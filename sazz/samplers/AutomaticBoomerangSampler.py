"""
AutomaticBoomerangSampler.py

Unified Boomerang sampler in PyTorch. Merges the base (Brent) and PLI
thinning variants into a single class controlled by `thinning_method`.

Design principles
-----------------
- The **trajectory**, **gradU**, and **reflect_velocity** methods live ON the
  torch computational graph so that gradients flow back through skeleton
  points to e.g. neural-network parameters.
- The **bounding / thinning** logic (Brent optimisation, PLI envelope
  construction and the thinning loop) runs OFF the graph — these are
  stochastic scheduling decisions, not differentiable maps. They receive
  *detached* tensors and return plain Python floats.
- The sampling loop re-evaluates the trajectory at the accepted event
  time *on the graph*, so the chain
      x_{n-1} -> trajectory(tau) -> gradU -> reflect -> v_n
  is fully differentiable.

"""

import math
import time as _time
from functools import partial
from typing import Callable, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bounding utilities live in utils/utils.py and are pure-numpy, off-graph.
# They are expected to expose:
#   brent(rate_fn, a, b, diagnostics=True) -> (x_star, stats)
#   piecewise_thinning_sinusoidal_second_order(
#       rate_fn, horizon, ..., diagnostics=True) -> (tau, stats)
# ---------------------------------------------------------------------------
from sazz.utils.bounding import (
    brent,
    piecewise_thinning_sinusoidal_second_order,
)


class AutomaticBoomerangSampler(nn.Module):
    """
    Boomerang sampler with automatic rate bounding.

    Parameters
    ----------
    grad_target : callable  (x: Tensor) -> Tensor
        Gradient of the (negative) log-target.  Must be torch-differentiable
        w.r.t. any parameters you want to learn.
    D : int
        Dimensionality of the target.
    refresh_rate : float
        Rate of velocity refreshment (Poisson clock).
    thinning : {"brent", "pli"}
        Which bounding strategy to use for Poisson thinning.
    t_max : float
        Horizon cap per iteration (only used by brent thinning).
    pli_kwargs : dict | None
        Extra keyword arguments forwarded to the PLI thinning function
        (alpha, n_segments, R, max_iter, second_harmonic).
    dtype : torch.dtype
        Working precision.
    device : torch.device | str
        Device for all tensors.
    """

    def __init__(
        self,
        grad_target: Callable[[Tensor], Tensor],
        D: int,
        refresh_rate: float = 0.1,
        thinning: Literal["brent", "pli"] = "pli",
        t_max: float = 0.1,
        pli_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.D = D
        self.grad_target = grad_target
        self.refresh_rate = refresh_rate
        self.thinning = thinning
        self.t_max = t_max
        self.pli_kwargs = pli_kwargs or {
            "alpha": 2.0,
            "R": 1.5,
            "n_segments": 8,
            "second_harmonic": True,
        }
        self.dtype = dtype
        self.device = torch.device(device)

        # Reference-measure quantities (set by preprocess)
        self.x_ref: Optional[Tensor] = None
        self.Sigma_inv: Optional[Tensor] = None
        self.Sigma: Optional[Tensor] = None
        self.Sigma_sqrt: Optional[Tensor] = None  # lower-triangular Cholesky

    # ------------------------------------------------------------------
    # Preprocessing — find reference point & covariance (OFF graph)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def preprocess(
        self,
        x_ref: Optional[Tensor] = None,
        Sigma_inv: Optional[Tensor] = None,
        jitter: float = 1e-8,
    ):
        """
        Set the Gaussian reference measure N(x_ref, Sigma).

        In the torch version we expect the user to supply x_ref and Sigma_inv
        directly (the old scipy.optimize path is removed — use torch.optim
        or similar upstream to find the mode / Hessian).

        Parameters
        ----------
        x_ref : Tensor [D]
            Reference point (e.g. MAP estimate).
        Sigma_inv : Tensor [D, D]
            Precision matrix of the reference Gaussian.
        jitter : float
            Diagonal jitter for numerical stability.
        """
        if x_ref is None or Sigma_inv is None:
            raise ValueError(
                "Supply both x_ref and Sigma_inv. "
                "Use torch.optim to find the MAP / Hessian upstream."
            )

        x_ref = x_ref.to(dtype=self.dtype, device=self.device).detach()
        Sigma_inv = Sigma_inv.to(dtype=self.dtype, device=self.device).detach()

        if Sigma_inv.ndim == 1:
            # Diagonal path: Sigma_inv is a 1-D vector of diagonal entries.
            # All operations are elementwise — no [D, D] matrix ever allocated.
            Sigma_inv_diag = Sigma_inv + jitter
            Sigma_diag     = 1.0 / Sigma_inv_diag
            Sigma_sqrt_diag = Sigma_diag.sqrt()
            self.x_ref      = x_ref
            self.Sigma_inv  = Sigma_inv_diag   # 1-D
            self.Sigma      = Sigma_diag        # 1-D
            self.Sigma_sqrt = Sigma_sqrt_diag   # 1-D
        else:
            # Original full-matrix path (kept for non-diagonal Sigma_inv).
            # Force symmetry + jitter
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
            Sigma_inv = Sigma_inv + jitter * torch.eye(
                self.D, dtype=self.dtype, device=self.device
            )
            Sigma = torch.linalg.inv(Sigma_inv)
            Sigma = 0.5 * (Sigma + Sigma.T)
            Sigma_sqrt = torch.linalg.cholesky(Sigma)  # lower triangular
            self.x_ref      = x_ref
            self.Sigma_inv  = Sigma_inv
            self.Sigma      = Sigma
            self.Sigma_sqrt = Sigma_sqrt

    # ------------------------------------------------------------------
    # Dynamics — ON the computational graph
    # ------------------------------------------------------------------
    def trajectory(self, t: float, x: Tensor, v: Tensor):
        """
        Boomerang (elliptical) trajectory.

        Returns (x_t, v_t) — both remain on the graph.
        """
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        dx = x - self.x_ref
        x_t = self.x_ref + dx * cos_t + v * sin_t
        v_t = -dx * sin_t + v * cos_t
        return x_t, v_t

    def gradU(self, x: Tensor) -> Tensor:
        """
        Gradient of the *excess* potential (target minus reference Gaussian).
        Stays on the graph.
        """
        # Original: self.Sigma_inv @ (x - self.x_ref)  [full matrix]
        dx = x - self.x_ref
        prec_dx = self.Sigma_inv * dx if self.Sigma_inv.ndim == 1 else self.Sigma_inv @ dx
        return self.grad_target(x) - prec_dx

    def reflect_velocity(self, v: Tensor, grad: Tensor) -> Tensor:
        """
        Specular reflection of v w.r.t. the Boomerang metric.
        Stays on the graph.
        """
        rate = torch.dot(v, grad)
        # Original: skew = self.Sigma_sqrt.T @ grad; self.Sigma_sqrt @ skew  [full matrix]
        if self.Sigma_sqrt.ndim == 1:
            skew = self.Sigma_sqrt * grad
            denom = torch.dot(skew, skew)
            return v - 2.0 * rate / denom * (self.Sigma_sqrt * skew)
        else:
            skew = self.Sigma_sqrt.T @ grad
            denom = torch.dot(skew, skew)
            return v - 2.0 * rate / denom * (self.Sigma_sqrt @ skew)

    # ------------------------------------------------------------------
    # Rate function — thin wrapper used OFF graph by the bounding code
    # ------------------------------------------------------------------
    def _rate_numpy(self, t: float, x_np: np.ndarray, v_np: np.ndarray) -> float:
        """
        Evaluate the signed rate at time t along the trajectory anchored
        at (x, v).  Runs in no_grad, converts to/from numpy for the
        bounding utilities.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory(t, x, v)
            inner = torch.dot(v_t, self.gradU(x_t))
        return float(inner)

    # ------------------------------------------------------------------
    # Event-time finders — OFF graph, return plain floats
    # ------------------------------------------------------------------
    def _find_event_time(
        self, pos: Tensor, vel: Tensor, horizon: float
    ) -> tuple[float, dict]:
        """
        Dispatch to the configured bounding strategy.
        pos, vel are already detached.
        Returns (tau_star, stats_dict).
        """
        pos_np = pos.cpu().numpy()
        vel_np = vel.cpu().numpy()

        if self.thinning == "brent":
            return self._brent_bound(pos_np, vel_np, horizon)
        elif self.thinning == "pli":
            return self._pli_bound(pos_np, vel_np, horizon)
        else:
            raise ValueError(f"Unknown thinning method: {self.thinning}")

    def _brent_bound(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        """
        Classic Brent-based global bounding.
        Find the maximum of the (negative) rate in [0, horizon], then do
        Poisson thinning with that constant bound.
        """
        neg_rate_fn = lambda t: -max(self._rate_numpy(t, pos_np, vel_np), 0.0)

        x_star, stats = brent(neg_rate_fn, 0.0, horizon, diagnostics=True)
        lambda_max = max(-neg_rate_fn(x_star), 0.0)

        if lambda_max <= 1e-14:
            tau_star = float("inf")
        else:
            tau_star = -math.log(np.random.random()) / lambda_max

        stats["lambda_max"] = lambda_max
        stats["tau"] = tau_star
        return tau_star, stats

    def _pli_bound(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        """
        Piecewise-linear (sinusoidal-aware) adaptive thinning.
        """
        rate_fn = partial(self._rate_numpy, x_np=pos_np, v_np=vel_np)
        tau, stats = piecewise_thinning_sinusoidal_second_order(
            rate_fn, horizon, diagnostics=True, **self.pli_kwargs
        )
        return tau, stats

    # ------------------------------------------------------------------
    # Velocity refresh — OFF graph (stochastic, no grad needed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_velocity(self) -> Tensor:
        """Sample a fresh velocity from N(0, Sigma)."""
        z = torch.randn(self.D, dtype=self.dtype, device=self.device)
        # Original: self.Sigma_sqrt @ z  [full matrix]
        return self.Sigma_sqrt * z if self.Sigma_sqrt.ndim == 1 else self.Sigma_sqrt @ z

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(
        self,
        N: int,
        x0: Optional[Tensor] = None,
        diagnostics: bool = True,
    ) -> dict:
        """
        Run the Boomerang sampler for N skeleton points.

        Parameters
        ----------
        N : int
            Number of skeleton points to collect.
        x0 : Tensor [D] | None
            Starting position.  Defaults to a draw from the reference measure.
        diagnostics : bool
            Whether to print summary statistics.

        Returns
        -------
        dict with keys:
            "positions"  : Tensor [N, D]
            "velocities" : Tensor [N, D]
            "times"      : Tensor [N]
            "diagnostics": list[dict]   (per-iteration stats)
            "gradient_evals": int
        """
        assert self.x_ref is not None, "Call preprocess() first."

        # --- Storage (detached, we just record the skeleton) ---
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        # --- Initialise ---
        if x0 is None:
            positions[0] = self.x_ref + self._refresh_velocity()
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._refresh_velocity()
        times[0] = 0.0

        dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        diag_log = []

        pbar = tqdm(total=N, desc="Boomerang", unit="skel")
        pbar.update(1)  # count the initial point
        iteration = 1

        while iteration < N:
            _t0 = _time.perf_counter()
            n = iteration

            # Current anchor on the trajectory (still on graph for the
            # re-evaluation below, but the bounding only sees detached).
            x_prev = positions[n - 1]
            v_prev = velocities[n - 1]

            # Advance along trajectory to current sub-position
            with torch.no_grad():
                pos, vel = self.trajectory(time_passed, x_prev, v_prev)

            # Determine horizon
            if self.thinning == "brent":
                horizon = min(self.t_max, dt_refresh)
            else:
                horizon = dt_refresh

            # --- OFF graph: find candidate event time ---
            tau, stats = self._find_event_time(pos.detach(), vel.detach(), horizon)
            grad_evals += stats.get("rate_evals", 0)

            row = {
                "rate_evals": stats.get("rate_evals", 0),
                "horizon": horizon,
                "time": current_time,
                "event_type": None,
                "accepted": None,
                "wall_seconds": None,
                # PLI extras (gracefully absent for brent)
                "proposals": stats.get("proposals", 0),
                "bound_violations": stats.get("bound_violations", 0),
                "max_ratio": stats.get("max_ratio", 0.0),
            }

            event_accepted = False

            if self.thinning == "brent":
                # Brent: tau is from Poisson thinning with constant bound.
                # Must do accept/reject at the proposed time.
                if tau < horizon:
                    row["event_type"] = "bounce"
                    u = np.random.random()

                    # Re-evaluate rate at tau (1 extra grad eval, off graph)
                    pos_np = pos.detach().cpu().numpy()
                    vel_np = vel.detach().cpu().numpy()
                    lambda_star = max(self._rate_numpy(tau, pos_np, vel_np), 0.0)
                    lambda_max = stats.get("lambda_max", 1.0)
                    p_accept = lambda_star / lambda_max if lambda_max > 1e-14 else 0.0
                    grad_evals += 1
                    row["rate_evals"] += 1

                    if u <= p_accept:
                        event_accepted = True
                        row["accepted"] = True
                    else:
                        row["accepted"] = False
                        time_passed += tau
                        current_time += tau
                        dt_refresh -= tau
                else:
                    row["event_type"] = "no_event"
                    row["accepted"] = False
                    time_passed += horizon
                    current_time += horizon
                    dt_refresh -= horizon

            else:  # PLI
                if tau < horizon:
                    row["event_type"] = "bounce"
                    row["accepted"] = True
                    event_accepted = True
                else:
                    row["event_type"] = "no_event"
                    time_passed += horizon
                    current_time += horizon
                    dt_refresh -= horizon

            # --- ON graph: materialise the accepted skeleton point ---
            if event_accepted:
                current_time += tau

                # Re-evaluate trajectory ON the graph at the accepted time
                pos_prop, vel_prop = self.trajectory(
                    time_passed + tau, x_prev, v_prev
                )
                grad_at_prop = self.gradU(pos_prop)
                vel_reflected = self.reflect_velocity(vel_prop, grad_at_prop)

                # Store (detach for the skeleton storage; the graph is
                # consumed upstream by whoever called .sample() or kept
                # references to grad_target's parameters).
                positions[n] = pos_prop.detach()
                velocities[n] = vel_reflected.detach()
                times[n] = times[n - 1] + time_passed + tau

                grad_evals += 1
                row["rate_evals"] += 1

                iteration += 1
                time_passed = 0.0
                dt_refresh -= tau
                pbar.update(1)

            # --- Velocity refresh ---
            if dt_refresh <= 1e-14:
                row["wall_seconds"] = _time.perf_counter() - _t0
                diag_log.append(row)

                refresh_row = {
                    "rate_evals": 0,
                    "horizon": 0.0,
                    "time": current_time,
                    "event_type": "refresh",
                    "accepted": None,
                    "wall_seconds": 0.0,
                    "proposals": 0,
                    "bound_violations": 0,
                    "max_ratio": 0.0,
                }

                if iteration < N:
                    n = iteration
                    with torch.no_grad():
                        pos_ref, _ = self.trajectory(
                            time_passed, positions[n - 1], velocities[n - 1]
                        )
                    positions[n] = pos_ref
                    velocities[n] = self._refresh_velocity()
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
                diag_log.append(refresh_row)
                pbar.set_postfix_str(f"t={current_time:.3f}", refresh=False)
                continue

            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(f"t={current_time:.3f}", refresh=False)
            diag_log.append(row)

        pbar.close()

        # --- Diagnostics ---
        if diagnostics:
            self._print_diagnostics(diag_log, N, grad_evals, times[iteration - 1])

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "diagnostics": diag_log,
            "gradient_evals": grad_evals,
        }

    # ------------------------------------------------------------------
    # Pretty-print diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _print_diagnostics(
        diag_log: list[dict], N: int, grad_evals: int, final_time: Tensor
    ):
        import pandas as pd

        df = pd.DataFrame(diag_log)

        n_accept = len(
            df[(df["event_type"] == "bounce") & (df["accepted"] == True)]
        )
        n_reject = len(
            df[(df["event_type"] == "bounce") & (df["accepted"] == False)]
        )

        print("\n=== Sampler Diagnostics ===")
        print(f"Total gradient evals  : {grad_evals}")
        print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")

        print("\n=== Thinning ===")
        if "proposals" in df.columns and df["proposals"].sum() > 0:
            print(f"Bound violations : {int(df['bound_violations'].sum())}")
            print(f"Max ratio        : {df['max_ratio'].max():.4f}")
            print(
                f"Proposals/call   : {df['proposals'].mean():.1f} mean, "
                f"{int(df['proposals'].max())} max"
            )
        if n_reject > 0:
            print(f"Accept / Reject  : {n_accept} / {n_reject}")
        else:
            print(f"Accepted bounces : {n_accept}")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "no_event", "refresh"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(
                    f"  {etype:10s}: {len(sub):5d} events, "
                    f"mean horizon={sub['horizon'].mean():.4f}"
                )

        print("\n=== Timing ===")
        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"Mean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
