import time as _time
from functools import partial
from typing import Callable, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bounding utilities live in utils/bounding.py and are pure-numpy, off-graph.
#   brent(f, a, b, diagnostics=True) -> (x_star, stats)
#       Minimises f over [a, b]; we pass f = -rate to find the max of rate.
#   piecewise_thinning(rate_fn, horizon, ..., diagnostics=True) -> (tau, stats)
#       Adaptive piecewise-linear upper envelope; returns +inf if no event
#       in [0, horizon].
# ---------------------------------------------------------------------------
from sazz.utils.bounding import brent, brent_monotone_aware, piecewise_thinning#, grid_thinning


class AutomaticZigZagSampler(nn.Module):
    """
    Zig-Zag sampler with automatic rate bounding.

    Parameters
    ----------
    grad_target : callable  (x: Tensor) -> Tensor
        Gradient of the (negative) log-target.  Must be torch-differentiable
        w.r.t. any parameters you want to learn.
    D : int
        Dimensionality of the target.
    t_max : float
        Horizon cap per iteration. For Brent this is the search interval
        for the rate maximum; for PLI it is the upper bound on the
        adaptive thinning horizon.
    gamma : float
        Refreshment rate added uniformly to every coordinate's switching
        rate. Ensures irreducibility.
    thinning : {"brent", "pli"}
        Which bounding strategy to use for Poisson thinning.
    adapt_t_max : bool
        If True, multiplicatively shrink ``t_max`` on accepted events
        (×0.96) and grow on no-event horizons (×1.01). Matches the legacy
        numpy zig-zag behaviour. Default False (static horizon, like the
        merged Boomerang).
    pli_kwargs : dict | None
        Extra keyword arguments forwarded to ``piecewise_thinning``
        (alpha, R, max_iter, t_init).
    dtype : torch.dtype
        Working precision.
    device : torch.device | str
        Device for all tensors.
    """

    def __init__(
        self,
        grad_target: Callable[[Tensor], Tensor],
        D: int,
        t_max: float = 0.1,
        gamma: float = 0.01,
        thinning: Literal["brent", "pli", "brent_monotone", "grid"] = "pli",
        adapt_t_max: bool = False,
        pli_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.D = D
        self.grad_target = grad_target
        self.t_max = float(t_max)
        self.gamma = float(gamma)
        self.thinning = thinning
        self.adapt_t_max = adapt_t_max
        self.pli_kwargs = pli_kwargs or {
            "alpha": 1.2,
            "R": 2.0,
            "max_iter": 200,
        }
        self.dtype = dtype
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Dynamics — ON the computational graph
    # ------------------------------------------------------------------
    def trajectory(self, t: float, x: Tensor, v: Tensor):
        """
        Linear Zig-Zag trajectory. Velocity is constant between events.
        Returns (x_t, v_t) — both remain on the graph.
        """
        x_t = x + t * v
        v_t = v
        return x_t, v_t

    def gradU(self, x: Tensor) -> Tensor:
        """
        Gradient of the (negative log) target. Stays on the graph.
        Aliased for parity with the Boomerang interface.
        """
        return self.grad_target(x)

    def flip_velocity(self, v: Tensor, i: int) -> Tensor:
        """
        Flip the i-th coordinate of the velocity. Stays on the graph
        (a sign flip is a no-op for autograd, but we keep the API
        symmetric with Boomerang's reflect_velocity).
        """
        v_new = v.clone()
        v_new[i] = -v_new[i]
        return v_new

    # ------------------------------------------------------------------
    # Rate function — thin wrapper used OFF graph by the bounding code
    # ------------------------------------------------------------------
    def _rate_numpy(self, t: float, x_np: np.ndarray, v_np: np.ndarray) -> float:
        """
        Total Zig-Zag switching rate at time t along the linear
        trajectory anchored at (x, v):
            lambda(t) = sum_i max(0, v_i * grad_i U(x + t v)) + D * gamma
        Runs in no_grad, converts to/from numpy for the bounding utilities.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t = x + t * v
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v * grad, min=0.0) + self.gamma
            rate = per_coord.sum()
        return float(rate)

    def _per_coord_rates_numpy(
        self, t: float, x_np: np.ndarray, v_np: np.ndarray
    ) -> np.ndarray:
        """
        Per-coordinate switching rates at time t. Used to draw the
        flipped index after an accepted event.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t = x + t * v
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v * grad, min=0.0) + self.gamma
        return per_coord.detach().cpu().numpy().astype(np.float64)

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
        if self.thinning == "brent_monotone":
            return self._brent_monotone_bound(pos_np, vel_np, horizon)
        elif self.thinning == "pli":
            return self._pli_bound(pos_np, vel_np, horizon)
        elif self.thinning == "grid":
            return self._grid_bound(pos_np, vel_np, horizon)
        else:
            raise ValueError(f"Unknown thinning method: {self.thinning}")

    def _brent_bound(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        """
        Classic Brent-based global bounding.
        Find the maximum of the rate in [0, horizon], then do Poisson
        thinning with that constant bound. The accept/reject step at
        the proposed time happens in the main loop.
        """
        neg_rate_fn = lambda t: -max(self._rate_numpy(t, pos_np, vel_np), 0.0)

        x_star, stats = brent(neg_rate_fn, 0.0, horizon, diagnostics=True)
        # Guard the boundary (legacy numpy zig-zag clamps too)
        # if x_star > horizon:
        #     x_star = horizon
        lambda_max = max(-neg_rate_fn(x_star), 0.0)

        if lambda_max <= 1e-14:
            tau_star = float("inf")
        else:
            tau_star = -np.log(np.random.random()) / lambda_max

        stats["lambda_max"] = lambda_max
        stats["tau"] = tau_star
        return tau_star, stats
    
    def _brent_monotone_bound(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        """
        Classic Brent-based global bounding.
        Find the maximum of the rate in [0, horizon], then do Poisson
        thinning with that constant bound. The accept/reject step at
        the proposed time happens in the main loop.
        """
        neg_rate_fn = lambda t: -max(self._rate_numpy(t, pos_np, vel_np), 0.0)

        x_star, stats = brent_monotone_aware(neg_rate_fn, 0.0, horizon, diagnostics=True)
        # Guard the boundary (legacy numpy zig-zag clamps too)
        # if x_star > horizon:
        #     x_star = horizon
        lambda_max = max(-neg_rate_fn(x_star), 0.0)

        if lambda_max <= 1e-14:
            tau_star = float("inf")
        else:
            tau_star = -np.log(np.random.random()) / lambda_max

        stats["lambda_max"] = lambda_max
        stats["tau"] = tau_star
        return tau_star, stats

    def _pli_bound(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        """
        Piecewise-linear adaptive thinning. Returns the accepted event
        time (or +inf if no event in [0, horizon]).
        """
        rate_fn = partial(self._rate_numpy, x_np=pos_np, v_np=vel_np)
        tau, stats = piecewise_thinning(
            rate_fn, horizon, diagnostics=True, **self.pli_kwargs
        )
        return tau, stats
    
    # def _grid_bound(
    #     self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    # ) -> tuple[float, dict]:
    #     """
    #     Grid bound by (Andral & Kamatani 2024)
    #     NOT READY YET
    #     """
    #     rate_fn = partial(self._rate_numpy, x_np=pos_np, v_np=vel_np)
    #     tau, stats = grid_thinning(
    #         rate_fn, horizon, diagnostics=True, **self.pli_kwargs
    #     )
    #     return tau, stats

    # ------------------------------------------------------------------
    # Initial velocity — ±1 per coordinate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _initial_velocity(self) -> Tensor:
        signs = torch.randint(
            0, 2, (self.D,), device=self.device, dtype=torch.int64
        )
        v = (2 * signs - 1).to(dtype=self.dtype)
        return v

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
        Run the Zig-Zag sampler for N skeleton points.

        Parameters
        ----------
        N : int
            Number of skeleton points to collect.
        x0 : Tensor [D] | None
            Starting position. Defaults to a draw from N(0, I).
        diagnostics : bool
            Whether to print summary statistics.

        Returns
        -------
        dict with keys:
            "positions"    : Tensor [N, D]
            "velocities"   : Tensor [N, D]
            "times"        : Tensor [N]
            "diagnostics"  : list[dict]   (per-iteration stats)
            "gradient_evals": int
        """
        # --- Storage (detached, we just record the skeleton) ---
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        # --- Initialise ---
        if x0 is None:
            positions[0] = torch.randn(self.D, dtype=self.dtype, device=self.device)
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._initial_velocity()
        times[0] = 0.0

        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        diag_log = []

        # Local working copy of t_max so adaptation doesn't mutate self.t_max
        t_max = self.t_max

        pbar = tqdm(total=N, desc=f"ZigZag[{self.thinning}]", unit="skel")
        pbar.update(1)
        iteration = 1

        while iteration < N:
            _t0 = _time.perf_counter()
            n = iteration

            x_prev = positions[n - 1]
            v_prev = velocities[n - 1]

            # Advance along trajectory to current sub-position (off graph;
            # we re-do this on graph below if the event is accepted).
            with torch.no_grad():
                pos, vel = self.trajectory(time_passed, x_prev, v_prev)

            horizon = t_max

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

            if self.thinning == "brent" or self.thinning == "brent_monotone":
                if tau < horizon:
                    row["event_type"] = "bounce"
                    u = np.random.random()

                    pos_np = pos.detach().cpu().numpy()
                    vel_np = vel.detach().cpu().numpy()
                    lambda_star = max(self._rate_numpy(tau, pos_np, vel_np), 0.0)
                    lambda_max = stats.get("lambda_max", 1.0)
                    p_accept = (
                        lambda_star / lambda_max if lambda_max > 1e-14 else 0.0
                    )
                    grad_evals += 1
                    row["rate_evals"] += 1

                    if u <= p_accept:
                        event_accepted = True
                        row["accepted"] = True
                    else:
                        row["accepted"] = False
                        time_passed += tau
                        current_time += tau
                else:
                    row["event_type"] = "no_event"
                    row["accepted"] = False
                    time_passed += horizon
                    current_time += horizon

            else:  # PLI
                if tau < horizon:
                    row["event_type"] = "bounce"
                    row["accepted"] = True
                    event_accepted = True
                else:
                    row["event_type"] = "no_event"
                    time_passed += horizon
                    current_time += horizon

            # --- ON graph: materialise the accepted skeleton point ---
            if event_accepted:
                current_time += tau

                # Re-evaluate trajectory ON the graph at the accepted time
                pos_prop, vel_prop = self.trajectory(
                    time_passed + tau, x_prev, v_prev
                )

                # Pick which coordinate flips: categorical proportional to
                # per-coord rates at the event time. Off graph (discrete).
                pos_np = pos_prop.detach().cpu().numpy()
                vel_np = vel_prop.detach().cpu().numpy()
                rates_i = self._per_coord_rates_numpy(0.0, pos_np, vel_np)
                total = rates_i.sum()
                if total <= 1e-14:
                    # Degenerate fallback: uniform pick
                    i_flip = int(np.random.randint(self.D))
                else:
                    probs = rates_i / total
                    i_flip = int(np.argmax(np.random.multinomial(1, probs)))

                vel_flipped = self.flip_velocity(vel_prop, i_flip)
                grad_evals += 1
                row["rate_evals"] += 1
                row["flipped_coord"] = i_flip

                positions[n] = pos_prop.detach()
                velocities[n] = vel_flipped.detach()
                times[n] = times[n - 1] + time_passed + tau

                iteration += 1
                time_passed = 0.0
                pbar.update(1)

                if self.adapt_t_max:
                    t_max *= 0.96
            else:
                if self.adapt_t_max:
                    t_max *= 1.01

            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(f"t={current_time:.3f}", refresh=False)
            diag_log.append(row)

        pbar.close()

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
        for etype in ["bounce", "no_event"]:
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
