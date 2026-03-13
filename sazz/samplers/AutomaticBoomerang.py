import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal
import autograd.numpy as anp
from autograd import grad, hessian

from tqdm import tqdm


class AutomaticBoomerangSampler:
    """
    Boomerang sampler with two options:

    1. sample(...)      : analytic affine bound using user/global m1
    2. sample_auto(...) : Automatic-Zig-Zag style local constant bound
                          based only on numerical maximization of the
                          rate on [0, t_max].

    Only requires:
        E(x) = negative log density (up to constant)
    """

    def __init__(self, E, dim, refresh_rate=1.0, m1=None, gradE=None):
        self.E = E
        self.dim = dim
        self.refresh_rate = refresh_rate

        if gradE is None:
            self.gradE = grad(E)
        else:
            self.gradE = gradE
        #self.gradE = grad(E)
        self.hessE = hessian(E)

        self.x_ref = None
        self.Sigma_inv = None
        self.Sigma = None
        self.Sigma_sqrt = None

        self.m1 = m1

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    def _estimate_m1(self, n_samples=20, proposal_scale=1.0):
        """
        Crude Hessian-norm estimate used only by the analytic-affine sampler.

        This is a heuristic. It is not part of the Boomerang reference measure
        itself, and it is mainly useful for the original affine-bound version.
        """
        vals = []
        for _ in range(n_samples):
            z = self.x_ref + np.random.normal(scale=proposal_scale, size=self.dim)
            H = np.array(self.hessE(z), dtype=float)
            vals.append(norm(H, 2))
        self.m1 = max(vals)

    def _set_reference_from_precision(self, x_ref, Sigma_inv, jitter=1e-8):
        """
        Set the Gaussian reference measure from a mean and precision matrix.

        The matrix is symmetrized and lightly regularized if needed, so this
        helper is slightly more forgiving numerically than a raw inversion.
        """
        self.x_ref = np.array(x_ref, dtype=float)
        Sigma_inv = np.array(Sigma_inv, dtype=float)
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
        Sigma_inv = Sigma_inv + jitter * np.eye(self.dim)

        self.Sigma_inv = Sigma_inv
        self.Sigma = np.linalg.inv(self.Sigma_inv)
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        self.Sigma_sqrt = cholesky(self.Sigma, lower=True)

    def preprocess(
        self,
        x0=None,
        method="laplace",
        x_ref=None,
        Sigma=None,
        Sigma_inv=None,
        jitter=1e-8,
        estimate_m1=False,
        m1_samples=20,
        m1_scale=1.0,
    ):
        """
        Construct the Gaussian reference measure used by Boomerang.

        Parameters
        ----------
        x0 : array_like, optional
            Starting point for mode finding when `method` is ``"laplace"`` or
            ``"diagonal"``. If omitted, the zero vector is used.
        method : {"laplace", "diagonal", "manual"}
            Strategy used to construct the Gaussian reference.

            - ``"laplace"``:
              Use the mode and the full Hessian at the mode. This is the
              standard Laplace-style choice and is usually the best default for
              smooth, reasonably well-behaved unimodal targets.

            - ``"diagonal"``:
              Use the mode but keep only the diagonal of the Hessian. This is a
              useful fallback when a full Hessian is too expensive, too noisy,
              poorly conditioned, or when you want a simpler preconditioner.
              It is more robust but less geometry-aware than ``"laplace"``.

            - ``"manual"``:
              Provide `x_ref` together with either `Sigma` or `Sigma_inv`.
              This is suitable when problem-specific structure is known in
              advance, when you already have a good Gaussian approximation, or
              when the target is awkward for Laplace preprocessing.

        x_ref : array_like, optional
            Reference location for ``method="manual"``.
        Sigma : array_like, optional
            Reference covariance for ``method="manual"``.
        Sigma_inv : array_like, optional
            Reference precision for ``method="manual"``.
        jitter : float, optional
            Small diagonal regularization added to the precision matrix before
            inversion/Cholesky. Useful for numerical stability.
        estimate_m1 : bool, optional
            If True and `self.m1` is still None, estimate the crude Hessian
            bound used by the original analytic-affine sampler.
        m1_samples : int, optional
            Number of probe points used in the crude `m1` estimate.
        m1_scale : float, optional
            Proposal scale around `x_ref` used when estimating `m1`.

        Notes
        -----
        Suitability guide:

        - Smooth target, clear mode, informative local curvature:
          use ``method="laplace"``.
        - Little curvature information, unstable Hessian, or large/simple
          problems where you want a cheap preconditioner:
          use ``method="diagonal"``.
        - Case-specific knowledge, custom reference, multimodal target, or an
          external approximation you trust more than a local Hessian:
          use ``method="manual"``.

        The Gaussian reference is part of the Boomerang construction. It is not
        specific to any one target, but its quality does depend strongly on the
        target and on how well the chosen reference captures its geometry.
        """
        if x0 is None:
            x0 = np.zeros(self.dim)

        method = method.lower()

        if method in {"laplace", "diagonal"}:
            res = minimize(self.E, x0, jac=lambda x: np.array(self.gradE(x)))
            x_mode = np.array(res.x, dtype=float)
            H_ref = np.array(self.hessE(x_mode), dtype=float)

            if method == "laplace":
                precision = H_ref
            else:
                diag_H = np.clip(np.diag(H_ref), jitter, None)
                precision = np.diag(diag_H)

            self._set_reference_from_precision(x_mode, precision, jitter=jitter)

        elif method == "manual":
            if x_ref is None:
                raise ValueError("For method='manual', x_ref must be provided.")
            if (Sigma is None) == (Sigma_inv is None):
                raise ValueError(
                    "For method='manual', provide exactly one of Sigma or Sigma_inv."
                )

            if Sigma_inv is not None:
                self._set_reference_from_precision(x_ref, Sigma_inv, jitter=jitter)
            else:
                Sigma = np.array(Sigma, dtype=float)
                Sigma = 0.5 * (Sigma + Sigma.T)
                Sigma = Sigma + jitter * np.eye(self.dim)
                self.x_ref = np.array(x_ref, dtype=float)
                self.Sigma = Sigma
                self.Sigma_inv = np.linalg.inv(Sigma)
                self.Sigma_inv = 0.5 * (self.Sigma_inv + self.Sigma_inv.T)
                self.Sigma_sqrt = cholesky(self.Sigma, lower=True)
        else:
            raise ValueError(
                "Unknown preprocess method. Use 'laplace', 'diagonal', or 'manual'."
            )

        if estimate_m1 and self.m1 is None:
            self._estimate_m1(n_samples=m1_samples, proposal_scale=m1_scale)

    # def preprocess(self, x0=None):
    #     if x0 is None:
    #         x0 = np.zeros(self.dim)

    #     res = minimize(self.E, x0, jac=lambda x: np.array(self.gradE(x)))
    #     self.x_ref = np.array(res.x, dtype=float)

    #     self.Sigma_inv = np.array(self.hessE(self.x_ref), dtype=float)
    #     self.Sigma = np.linalg.inv(self.Sigma_inv)
    #     self.Sigma_sqrt = cholesky(self.Sigma, lower=True)

    #     # crude global Hessian bound for the analytic affine version
    #     if self.m1 is None:
    #         vals = []
    #         for _ in range(20):
    #             z = self.x_ref + np.random.normal(scale=1.0, size=self.dim)
    #             H = np.array(self.hessE(z), dtype=float)
    #             vals.append(norm(H, 2))
    #         self.m1 = max(vals)

    # ------------------------------------------------------------
    # Shared dynamics / rate helpers
    # ------------------------------------------------------------

    def elliptic_step(self, dt, y, v):
        """
        Evolve the centered state y = x - x_ref and velocity v
        for time dt under the Boomerang deterministic flow.
        """
        c = np.cos(dt)
        s = np.sin(dt)
        y_new = c * y + s * v
        v_new = -s * y + c * v
        return y_new, v_new

    def gradU(self, x):
        """
        Residual gradient U relative to the Gaussian reference.
        """
        return np.array(self.gradE(x), dtype=float) - self.Sigma_inv @ (x - self.x_ref)

    def rate(self, x, v):
        """
        True switching rate lambda(x,v) = <v, gradU(x)>_+.
        """
        return max(float(np.dot(v, self.gradU(x))), 0.0)

    def state_on_orbit(self, x, v, dt):
        """
        Deterministic state after dt from current state (x,v).
        """
        y = x - self.x_ref
        y_new, v_new = self.elliptic_step(dt, y, v)
        x_new = y_new + self.x_ref
        return x_new, v_new

    def reflect_velocity(self, v, gradU):
        """
        Full Boomerang reflection:
            v <- v - 2 <v,gradU> / |Sigma^{1/2} gradU|^2 * Sigma gradU
        """
        rate = float(np.dot(v, gradU))
        skew = self.Sigma_sqrt.T @ gradU
        denom = float(np.dot(skew, skew))

        if denom <= 1e-14:
            return v.copy()

        return v - 2.0 * rate / denom * (self.Sigma_sqrt @ skew)

    # ------------------------------------------------------------
    # Original affine-switching-time helper
    # ------------------------------------------------------------

    def switching_time(self, a, b):
        u = np.random.rand()

        if a <= 0 and b <= 1e-14:
            return np.inf

        if abs(b) < 1e-14:
            if a <= 0:
                return np.inf
            return -np.log(u) / a

        disc = a**2 - 2.0 * b * np.log(u)
        if disc < 0:
            return np.inf

        tau = (-a + np.sqrt(disc)) / b
        if tau < 0 or not np.isfinite(tau):
            return np.inf

        return tau

    # ------------------------------------------------------------
    # Original analytic-affine sampler
    # ------------------------------------------------------------

    def sample(self, T=10.0):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")

        x = self.x_ref.copy()
        v = self.Sigma_sqrt @ np.random.randn(self.dim)
        t = 0.0

        x_skeleton = [x.copy()]
        v_skeleton = [v.copy()]
        t_skeleton = [t]

        accepted_switches = 0
        rejected_switches = 0
        refreshes = 0

        gradU = self.gradU(x)
        M2 = norm(gradU)

        phase_norm = np.sqrt(norm(x - self.x_ref) ** 2 + norm(v) ** 2)
        a = float(np.dot(v, gradU))
        b = float(self.m1 * phase_norm**2 + M2 * phase_norm)

        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

        while t < T:
            dt_switch = self.switching_time(a, b)
            dt = min(dt_switch, dt_refresh)

            if t + dt > T:
                dt = T - t
                y, v = self.elliptic_step(dt, x - self.x_ref, v)
                x = y + self.x_ref
                t += dt

                x_skeleton.append(x.copy())
                v_skeleton.append(v.copy())
                t_skeleton.append(t)
                break

            y, v = self.elliptic_step(dt, x - self.x_ref, v)
            x = y + self.x_ref
            t += dt

            gradU = self.gradU(x)

            if dt_switch < dt_refresh:
                simulated_rate = max(a + b * dt, 0.0)
                true_rate = max(float(np.dot(v, gradU)), 0.0)

                if simulated_rate > 0.0 and np.random.rand() * simulated_rate <= true_rate:
                    v = self.reflect_velocity(v, gradU)
                    accepted_switches += 1

                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                else:
                    rejected_switches += 1

                phase_norm = np.sqrt(norm(x - self.x_ref) ** 2 + norm(v) ** 2)
                a = float(np.dot(v, gradU))
                b = float(self.m1 * phase_norm**2 + M2 * phase_norm)

                dt_refresh -= dt

            else:
                v = self.Sigma_sqrt @ np.random.randn(self.dim)
                refreshes += 1

                phase_norm = np.sqrt(norm(x - self.x_ref) ** 2 + norm(v) ** 2)
                a = float(np.dot(v, gradU))
                b = float(self.m1 * phase_norm**2 + M2 * phase_norm)

                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

                x_skeleton.append(x.copy())
                v_skeleton.append(v.copy())
                t_skeleton.append(t)

        return {
            "t": np.array(t_skeleton),
            "x": np.array(x_skeleton),
            "v": np.array(v_skeleton),
            "x_ref": self.x_ref.copy(),
            "accepted_switches": accepted_switches,
            "rejected_switches": rejected_switches,
            "refreshes": refreshes,
            "method": "analytic_affine",
        }

    # ------------------------------------------------------------
    # Automatic local bound helpers
    # ------------------------------------------------------------

    def _max_rate_on_interval(
        self,
        x,
        v,
        horizon,
        n_grid,
        safety,
        refine,
        n_candidates=2,
    ):
        """
        Automatic local constant bound:
            lambda_bar >= sup_{t in [0,horizon]} lambda(x_t, v_t)

        Strategy:
        1. evaluate on a grid,
        2. optionally refine near the top few grid cells,
        3. inflate by a safety factor.
        """
        if horizon <= 0:
            return 0.0, np.array([0.0]), np.array([0.0])

        ts = np.linspace(0.0, horizon, n_grid + 1)
        vals = np.empty_like(ts)

        for k, tk in enumerate(ts):
            xt, vt = self.state_on_orbit(x, v, tk)
            vals[k] = self.rate(xt, vt)
        grad_evals = len(vals)
        
        #imax = np.argmax(vals)

        left_diffs = np.diff(vals)
        
        tol = 1e-10 + 1e-6 * np.max(np.abs(vals))
        if np.max(vals) <= tol:
            return 0.0, ts, vals, grad_evals
        
        increasing = np.all(left_diffs >= -tol)
        decreasing = np.all(left_diffs <= tol)
        
        if increasing:
            lam_max = vals[-1]
            lam_bar = safety * lam_max
            return max(lam_bar, 0.0), ts, vals, grad_evals
        if decreasing:
            lam_max = vals[0]
            lam_bar = safety * lam_max
            return max(lam_bar, 0.0), ts, vals, grad_evals
        
        lam_max = float(np.max(vals))

        if refine and horizon > 1e-12:
            # indices of the top few grid values, sorted from largest to smaller
            candidate_idxs = np.argsort(vals)[-n_candidates:][::-1]

            for max_idx in candidate_idxs:
                left_idx = max(0, max_idx - 1)
                right_idx = min(len(ts) - 1, max_idx + 1)
                a = ts[left_idx]
                b = ts[right_idx]

                if b - a > 1e-12:
                    local_evals = 0

                    def obj(tau):
                        nonlocal local_evals
                        local_evals += 1
                        xt, vt = self.state_on_orbit(x, v, tau)
                        return -self.rate(xt, vt)

                    res = minimize_scalar(obj, bounds=(a, b), method="bounded")
                    grad_evals += local_evals
                    if res.success and np.isfinite(res.fun):
                        lam_max = max(lam_max, -float(res.fun))

        lam_bar = safety * lam_max
        return max(lam_bar, 0.0), ts, vals, grad_evals
    # ------------------------------------------------------------
    # Automatic local-bound sampler
    # ------------------------------------------------------------

    def sample_auto(
        self,
        T=10.0,
        t_max=0.5,
        n_grid=16,
        safety=1.10,
        refine=True,
        adapt_t_max=False,
        t_max_min=1e-3,
        t_max_max=5.0,
        bound_violation_tol=1e-10,
    ):
        """
        Automatic-Zig-Zag style Boomerang sampler.

        On each local interval [0, horizon], horizon <= t_max,
        numerically maximize the true rate along the deterministic
        Boomerang orbit and thin against the resulting constant bound.

        This needs only E (via autodiff).

        Parameters
        ----------
        t_max : float
            Local time horizon used to build the bound.
        n_grid : int
            Number of grid subintervals for numerical maximization.
        safety : float > 1
            Safety inflation factor on the numerically computed max.
        adapt_t_max : bool
            Mild heuristic adaptation of t_max based on what happens
            on recent intervals.
        """
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        if safety < 1.0:
            raise ValueError("safety should be >= 1.0.")

        x = self.x_ref.copy()
        v = self.Sigma_sqrt @ np.random.randn(self.dim)
        t = 0.0

        x_skeleton = [x.copy()]
        v_skeleton = [v.copy()]
        t_skeleton = [t]

        accepted_switches = 0
        rejected_switches = 0
        refreshes = 0
        total_gradient_evals = 0
        bound_violations = 0

        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
        
        t_max_arr = [t_max]
        windows = 0
        empty_windows = 0

        # rolling adaptation diagnostics
        adapt_window = 20
        recent_empty_windows = 0
        recent_windows = 0
        recent_proposals = 0
        recent_accepted_proposals = 0

        # thresholds
        empty_window_threshold = 0.6
        acceptance_threshold = 0.3
        
        
        horizon_history = []
        lam_bar_history = []
        window_times = []
        window_x_history = []
        window_v_history = []
        
        pbar = tqdm(total=T, desc="Sampling", leave=True)
        while t < T:
            windows += 1
            horizon = min(t_max, dt_refresh, T - t)
            
            window_times.append(t)
            window_x_history.append(x.copy())
            window_v_history.append(v.copy())

            # local constant bound on [0, horizon]
            lam_bar, _, _, gradient_evals = self._max_rate_on_interval(
                x=x,
                v=v,
                horizon=horizon,
                n_grid=n_grid,
                safety=safety,
                refine=refine,
            )
            
            horizon_history.append(horizon)
            lam_bar_history.append(lam_bar)
            
            total_gradient_evals += gradient_evals

            window_was_empty = False
            proposal_happened = False
            proposal_accepted = False

            # no possible event on this local interval
            if lam_bar <= 1e-14:
                x, v = self.state_on_orbit(x, v, horizon)
                t += horizon
                dt_refresh -= horizon
                window_was_empty = True

                if dt_refresh <= 1e-14 and t < T:
                    v = self.Sigma_sqrt @ np.random.randn(self.dim)
                    dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
                    refreshes += 1

                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                
                pbar.update(t - pbar.n)

            else:
                tau = np.random.exponential(1.0 / lam_bar)

                # no Poisson proposal before end of local window
                if tau >= horizon:
                    x, v = self.state_on_orbit(x, v, horizon)
                    t += horizon
                    dt_refresh -= horizon
                    window_was_empty = True

                    if dt_refresh <= 1e-14 and t < T:
                        v = self.Sigma_sqrt @ np.random.randn(self.dim)
                        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
                        refreshes += 1

                        x_skeleton.append(x.copy())
                        v_skeleton.append(v.copy())
                        t_skeleton.append(t)
                    pbar.update(t - pbar.n)

                else:
                    proposal_happened = True

                    # proposed bounce time inside the local window
                    x_prop, v_prop = self.state_on_orbit(x, v, tau)
                    true_gradU = self.gradU(x_prop)
                    true_rate = max(float(np.dot(v_prop, true_gradU)), 0.0)
                    total_gradient_evals += 1

                    # diagnostic: if this happens, the numerical bound failed
                    if true_rate > lam_bar * (1.0 + bound_violation_tol):
                        bound_violations += 1
                        if adapt_t_max:
                            t_max = max(0.5 * t_max, t_max_min)
                            t_max_arr.append(t_max)
                        raise RuntimeError(
                            f"Automatic local bound violated: true_rate={true_rate:.6e}, "
                            f"lam_bar={lam_bar:.6e}. Increase safety / grid, or reduce t_max."
                        )

                    # thinning
                    if np.random.rand() * lam_bar <= true_rate:
                        x = x_prop
                        v = self.reflect_velocity(v_prop, true_gradU)
                        t += tau
                        dt_refresh -= tau

                        accepted_switches += 1
                        proposal_accepted = True

                        x_skeleton.append(x.copy())
                        v_skeleton.append(v.copy())
                        t_skeleton.append(t)
                        pbar.update(t - pbar.n)

                    else:
                        x = x_prop
                        v = v_prop
                        t += tau
                        dt_refresh -= tau

                        rejected_switches += 1
                        
                        pbar.update(t - pbar.n)

                    # refresh if it happens exactly now due to numerical exhaustion
                    if dt_refresh <= 1e-14 and t < T:
                        v = self.Sigma_sqrt @ np.random.randn(self.dim)
                        dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
                        refreshes += 1

                        x_skeleton.append(x.copy())
                        v_skeleton.append(v.copy())
                        t_skeleton.append(t)

            # ------------------------------------------------------------
            # Adapt t_max using rolling window diagnostics
            # ------------------------------------------------------------
            if adapt_t_max:
                recent_windows += 1
                if window_was_empty:
                    recent_empty_windows += 1
                    empty_windows += 1
                if proposal_happened:
                    recent_proposals += 1
                if proposal_accepted:
                    recent_accepted_proposals += 1

                if recent_windows >= adapt_window:
                    empty_fraction = recent_empty_windows / recent_windows
                    acceptance_fraction = (
                        recent_accepted_proposals / recent_proposals
                        if recent_proposals > 0 else 1.0
                    )

                    updated = False

                    # too many empty windows -> t_max likely too small
                    if empty_fraction > empty_window_threshold:
                        new_t_max = min(1.1 * t_max, t_max_max)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True

                    # too many rejected proposals -> t_max likely too large
                    elif recent_proposals > 0 and acceptance_fraction < acceptance_threshold:
                        new_t_max = max(0.95 * t_max, t_max_min)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True

                    if updated:
                        t_max_arr.append(t_max)

                    # reset rolling counters
                    recent_empty_windows = 0
                    recent_windows = 0
                    recent_proposals = 0
                    recent_accepted_proposals = 0
                    
        pbar.close()

        return {
            "t": np.array(t_skeleton),
            "x": np.array(x_skeleton),
            "v": np.array(v_skeleton),
            "x_ref": self.x_ref.copy(),
            "accepted_switches": accepted_switches,
            "rejected_switches": rejected_switches,
            "refreshes": refreshes,
            "bound_violations": bound_violations,
            "method": "automatic_local_constant",
            "t_max": t_max_arr,
            "windows": windows,
            "empty_windows": empty_windows,
            "gradient_evals": total_gradient_evals,
            "horizon_history": np.array(horizon_history),
            "lam_bar_history": np.array(lam_bar_history),
            "window_times": np.array(window_times),
            "window_x_history": np.array(window_x_history),
            "window_v_history": np.array(window_v_history),
        }

 
class FactorizedAutomaticBoomerangSampler(AutomaticBoomerangSampler):
    """
    Automatic factorized Boomerang sampler.

    This extends AutomaticBoomerangSampler by replacing the single global
    switching rate
        lambda(x, v) = <v, gradU(x)>_+
    with coordinate-wise rates
        lambda_i(x, v) = (v_i * d_i U(x))_+.

    Relative to the standard automatic Boomerang sampler, this class:
      - uses a diagonal Gaussian reference covariance,
      - flips only one velocity coordinate at accepted events,
      - refreshes velocity coordinates independently,
      - constructs *local numerical constant bounds* for each coordinate rate
        on short deterministic orbit windows.

    The local bounds are computed from shared gradient evaluations along the
    orbit, so one grid sweep yields bounds for all coordinates.
    """

    def __init__(self, E, dim, refresh_rate=1.0, m1=None, gradE=None, partial_gradE=None):
        super().__init__(E=E, dim=dim, refresh_rate=refresh_rate, m1=m1, gradE=gradE)
        self._user_partial_gradE = partial_gradE

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------

    def preprocess(
        self,
        x0=None,
        method="diagonal",
        x_ref=None,
        Sigma=None,
        Sigma_inv=None,
        jitter=1e-8,
        estimate_m1=True,
        m1_samples=20,
        m1_scale=1.0,
    ):
        """
        Construct the diagonal Gaussian reference used by the factorized sampler.

        Because the factorized Boomerang construction assumes diagonal
        reference covariance, this method is stricter than the base-class
        version.

        Parameters
        ----------
        method : {"diagonal", "laplace", "manual"}
            - ``"diagonal"``:
              Find the mode and keep only the diagonal of the Hessian. This is
              the recommended default for the factorized sampler.
            - ``"laplace"``:
              Only allowed if the Hessian at the mode is already diagonal (up
              to numerical tolerance). This is mainly a safety check.
            - ``"manual"``:
              Provide a diagonal `Sigma` or `Sigma_inv` together with `x_ref`.

        Notes
        -----
        Suitability guide:

        - General use for the factorized sampler:
          use ``method="diagonal"``.
        - If you know the reference is truly diagonal already:
          ``method="laplace"`` is acceptable.
        - If you have target-specific structure or an externally chosen
          diagonal preconditioner:
          use ``method="manual"``.
        """
        if x0 is None:
            x0 = np.zeros(self.dim)

        method = method.lower()

        if method in {"diagonal", "laplace"}:
            res = minimize(self.E, x0, jac=lambda x: np.array(self.gradE(x)))
            x_mode = np.array(res.x, dtype=float)
            H_ref = np.array(self.hessE(x_mode), dtype=float)

            if method == "diagonal":
                diag_H = np.clip(np.diag(H_ref), jitter, None)
                precision = np.diag(diag_H)
            else:
                offdiag = H_ref - np.diag(np.diag(H_ref))
                if not np.allclose(offdiag, 0.0, atol=1e-10):
                    raise ValueError(
                        "FactorizedAutomaticBoomerangSampler requires a diagonal "
                        "reference covariance. Use method='diagonal' or pass a "
                        "diagonal manual reference."
                    )
                precision = H_ref

            self._set_reference_from_precision(x_mode, precision, jitter=jitter)
            sigma_diag = np.clip(np.diag(self.Sigma), jitter, None)
            self.Sigma = np.diag(sigma_diag)
            self.Sigma_inv = np.diag(1.0 / sigma_diag)
            self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))

        elif method == "manual":
            if x_ref is None:
                raise ValueError("For method='manual', x_ref must be provided.")
            if (Sigma is None) == (Sigma_inv is None):
                raise ValueError(
                    "For method='manual', provide exactly one of Sigma or Sigma_inv."
                )

            self.x_ref = np.array(x_ref, dtype=float)
            if Sigma_inv is not None:
                precision = np.array(Sigma_inv, dtype=float)
                if not np.allclose(precision, np.diag(np.diag(precision)), atol=1e-10):
                    raise ValueError("Manual Sigma_inv must be diagonal for factorized Boomerang.")
                diag_precision = np.clip(np.diag(precision), jitter, None)
                self.Sigma_inv = np.diag(diag_precision)
                sigma_diag = 1.0 / diag_precision
                self.Sigma = np.diag(sigma_diag)
                self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))
            else:
                Sigma = np.array(Sigma, dtype=float)
                if not np.allclose(Sigma, np.diag(np.diag(Sigma)), atol=1e-10):
                    raise ValueError("Manual Sigma must be diagonal for factorized Boomerang.")
                sigma_diag = np.clip(np.diag(Sigma), jitter, None)
                self.Sigma = np.diag(sigma_diag)
                self.Sigma_inv = np.diag(1.0 / sigma_diag)
                self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))
        else:
            raise ValueError(
                "Unknown preprocess method. Use 'diagonal', 'laplace', or 'manual'."
            )

        if estimate_m1 and self.m1 is None:
            self._estimate_m1(n_samples=m1_samples, proposal_scale=m1_scale)

    # def preprocess(self, x0=None, diagonalize=True):
    #     """
    #     Find x_ref and construct a diagonal Gaussian reference.

    #     The factorized construction in the paper assumes diagonal Sigma.
    #     Therefore, by default, we diagonalize the Hessian-based Gaussian
    #     approximation at the mode.
    #     """
    #     if x0 is None:
    #         x0 = np.zeros(self.dim)

    #     res = minimize(self.E, x0, jac=lambda x: np.array(self.gradE(x)))
    #     self.x_ref = np.array(res.x, dtype=float)

    #     H_ref = np.array(self.hessE(self.x_ref), dtype=float)

    #     if diagonalize:
    #         diag_H = np.clip(np.diag(H_ref), 1e-8, None)
    #         self.Sigma_inv = np.diag(diag_H)
    #     else:
    #         offdiag = H_ref - np.diag(np.diag(H_ref))
    #         if not np.allclose(offdiag, 0.0, atol=1e-10):
    #             raise ValueError(
    #                 "FactorizedAutomaticBoomerangSampler requires diagonal "
    #                 "reference covariance. Use diagonalize=True."
    #             )
    #         self.Sigma_inv = H_ref

    #     sigma_diag = 1.0 / np.diag(self.Sigma_inv)
    #     self.Sigma = np.diag(sigma_diag)
    #     self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))

    #     # keep compatibility with the parent analytic-affine sampler API
    #     if self.m1 is None:
    #         vals = []
    #         for _ in range(20):
    #             z = self.x_ref + np.random.normal(scale=1.0, size=self.dim)
    #             H = np.array(self.hessE(z), dtype=float)
    #             vals.append(norm(H, 2))
    #         self.m1 = max(vals)

    # ------------------------------------------------------------
    # Partial derivatives (currently uses full gradient, if simpler version exists, these will be overridden)
    # ------------------------------------------------------------
    # def partial_gradU(self, x, i):
    #     return self.gradU(x)[i]
    
    def partial_gradU(self, x, i):
        if self._user_partial_gradE is not None:
            dEi = float(self._user_partial_gradE(x, i))
        else:
            dEi = float(self.gradE(x)[i])
        return dEi - self.Sigma_inv[i, i] * (x[i] - self.x_ref[i])
    
    def partial_rate(self, x, v, i):
        return max(float(v[i] * self.partial_gradU(x, i)), 0.0)
    # ------------------------------------------------------------
    # Factorized helpers
    # ------------------------------------------------------------

    def sample_velocity(self):
        return np.diag(self.Sigma_sqrt) * np.random.randn(self.dim)

    def coordinate_rate(self, x, v):
        """
        Vector of coordinate-wise switching rates.
        """
        return np.maximum(v * self.gradU(x), 0.0)

    def coordinate_refresh_time(self):
        """
        Independent coordinate refreshes with rate refresh_rate per coordinate.

        Equivalently, sample the first event of the superposed Poisson process
        of total rate dim * refresh_rate and then choose the affected
        coordinate uniformly.
        """
        if self.refresh_rate <= 0:
            return np.inf, None
        i = np.random.randint(self.dim)
        dt = np.random.exponential(1.0 / (self.dim * self.refresh_rate))
        return dt, i

    def _coordinate_rates_on_interval(self, x, v, ts):
        """
        Evaluate all coordinate rates on a shared grid of orbit times.

        Returns
        -------
        vals : ndarray, shape (len(ts), dim)
            vals[k, i] = lambda_i(x_tk, v_tk)
        grad_evals : int
            Number of residual-gradient evaluations used.
        """
        vals = np.empty((len(ts), self.dim), dtype=float)
        for k, tk in enumerate(ts):
            xt, vt = self.state_on_orbit(x, v, tk)
            vals[k, :] = np.maximum(vt * self.gradU(xt), 0.0)
        return vals, len(ts)

    def _max_coordinate_rates_on_interval(
        self,
        x,
        v,
        horizon,
        n_grid,
        safety,
        refine,
        n_candidates=2,
    ):
        """
        Local constant bounds for all coordinate rates on [0, horizon].

        For each coordinate i, compute
            lam_bar_i >= sup_{t in [0, horizon]} lambda_i(x_t, v_t)
        using a shared orbit grid and optional local scalar refinements near
        the best grid points.
        """
        if horizon <= 0:
            ts = np.array([0.0])
            vals = np.zeros((1, self.dim), dtype=float)
            return np.zeros(self.dim, dtype=float), ts, vals, 0

        ts = np.linspace(0.0, horizon, n_grid + 1)
        vals, grad_evals = self._coordinate_rates_on_interval(x, v, ts)

        lam_max = np.max(vals, axis=0)
        tol = 1e-10 + 1e-6 * np.max(np.abs(vals))

        if refine and horizon > 1e-12 and np.max(lam_max) > tol:
            for i in range(self.dim):
                if lam_max[i] <= tol:
                    continue

                diffs = np.diff(vals[:, i])
                increasing = np.all(diffs >= -tol)
                decreasing = np.all(diffs <= tol)
                if increasing or decreasing:
                    continue

                candidate_idxs = np.argsort(vals[:, i])[-n_candidates:][::-1]
                for max_idx in candidate_idxs:
                    left_idx = max(0, max_idx - 1)
                    right_idx = min(len(ts) - 1, max_idx + 1)
                    a = ts[left_idx]
                    b = ts[right_idx]

                    if b - a <= 1e-12:
                        continue

                    local_evals = 0

                    def obj(tau):
                        nonlocal local_evals
                        local_evals += 1
                        xt, vt = self.state_on_orbit(x, v, tau)
                        #return -max(float(vt[i] * self.gradU(xt)[i]), 0.0)
                        gi = self.partial_gradU(xt, i)
                        return -max(float(vt[i] * gi), 0.0)

                    res = minimize_scalar(obj, bounds=(a, b), method="bounded")
                    grad_evals += local_evals
                    if res.success and np.isfinite(res.fun):
                        lam_max[i] = max(lam_max[i], -float(res.fun))

        lam_bar = safety * lam_max
        return np.maximum(lam_bar, 0.0), ts, vals, grad_evals

    def _sample_bound_event_time(self, lam_bar_vec):
        """
        First event time and event coordinate for the superposition of
        coordinate-wise constant-rate Poisson processes.
        """
        total_rate = float(np.sum(lam_bar_vec))
        if total_rate <= 1e-14:
            return np.inf, None

        tau = np.random.exponential(1.0 / total_rate)
        probs = lam_bar_vec / total_rate
        i = int(np.random.choice(self.dim, p=probs))
        return tau, i

    # ------------------------------------------------------------
    # Automatic local-bound sampler
    # ------------------------------------------------------------

    def sample_auto(
        self,
        T=10.0,
        t_max=0.5,
        n_grid=16,
        safety=1.10,
        refine=True,
        adapt_t_max=False,
        t_max_min=1e-3,
        t_max_max=5.0,
        bound_violation_tol=1e-10,
        return_velocities=True,
    ):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        if safety < 1.0:
            raise ValueError("safety should be >= 1.0.")

        x = self.x_ref.copy()
        v = self.sample_velocity()
        t = 0.0

        x_skeleton = [x.copy()]
        t_skeleton = [t]
        if return_velocities:
            v_skeleton = [v.copy()]

        accepted_switches = 0
        rejected_switches = 0
        refreshes = 0
        total_gradient_evals = 0
        bound_violations = 0

        dt_refresh = np.random.exponential(1.0 / (self.dim * self.refresh_rate)) if self.refresh_rate > 0 else np.inf
        i_refresh_pending = np.random.randint(self.dim) if self.refresh_rate > 0 else None

        t_max_arr = [t_max]
        windows = 0
        empty_windows = 0

        adapt_window = 20
        recent_empty_windows = 0
        recent_windows = 0
        recent_proposals = 0
        recent_accepted_proposals = 0

        empty_window_threshold = 0.6
        acceptance_threshold = 0.3

        horizon_history = []
        lam_bar_history = []
        window_times = []
        window_x_history = []
        window_v_history = []
        
        pbar = tqdm(total=T, desc="Sampling", leave=True)
        while t < T:
            windows += 1
            horizon = min(t_max, dt_refresh, T - t)

            window_times.append(t)
            window_x_history.append(x.copy())
            window_v_history.append(v.copy())

            lam_bar_vec, _, _, gradient_evals = self._max_coordinate_rates_on_interval(
                x=x,
                v=v,
                horizon=horizon,
                n_grid=n_grid,
                safety=safety,
                refine=refine,
            )
            total_gradient_evals += gradient_evals

            horizon_history.append(horizon)
            lam_bar_history.append(lam_bar_vec.copy())

            window_was_empty = False
            proposal_happened = False
            proposal_accepted = False

            tau, i_switch = self._sample_bound_event_time(lam_bar_vec)

            if tau >= horizon:
                x, v = self.state_on_orbit(x, v, horizon)
                t += horizon
                dt_refresh -= horizon
                window_was_empty = True
                
                pbar.update(t - pbar.n)

                if dt_refresh <= 1e-14 and t < T and self.refresh_rate > 0:
                    sigma_i = self.Sigma_sqrt[i_refresh_pending, i_refresh_pending]
                    v[i_refresh_pending] = sigma_i * np.random.randn()
                    refreshes += 1

                    dt_refresh, i_refresh_pending = self.coordinate_refresh_time()

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())

            else:
                proposal_happened = True

                x_prop, v_prop = self.state_on_orbit(x, v, tau)
                #true_gradU = self.gradU(x_prop)
                gi = self.partial_gradU(x_prop, i_switch)
                total_gradient_evals += 1
                #true_rate_i = max(float(v_prop[i_switch] * true_gradU[i_switch]), 0.0)
                true_rate_i = max(float(v_prop[i_switch] * gi), 0.0)

                if true_rate_i > lam_bar_vec[i_switch] * (1.0 + bound_violation_tol):
                    bound_violations += 1
                    if adapt_t_max:
                        t_max = max(0.5 * t_max, t_max_min)
                        t_max_arr.append(t_max)
                    raise RuntimeError(
                        f"Automatic factorized local bound violated in coordinate {i_switch}: "
                        f"true_rate={true_rate_i:.6e}, "
                        f"lam_bar={lam_bar_vec[i_switch]:.6e}. "
                        "Increase safety / grid, or reduce t_max."
                    )

                if np.random.rand() * lam_bar_vec[i_switch] <= true_rate_i:
                    x = x_prop
                    v = v_prop
                    v[i_switch] *= -1.0
                    t += tau
                    dt_refresh -= tau
                    
                    pbar.update(t - pbar.n)

                    accepted_switches += 1
                    proposal_accepted = True

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())
                else:
                    x = x_prop
                    v = v_prop
                    t += tau
                    dt_refresh -= tau
                    rejected_switches += 1
                    
                    pbar.update(t - pbar.n)

                if dt_refresh <= 1e-14 and t < T and self.refresh_rate > 0:
                    sigma_i = self.Sigma_sqrt[i_refresh_pending, i_refresh_pending]
                    v[i_refresh_pending] = sigma_i * np.random.randn()
                    refreshes += 1

                    dt_refresh, i_refresh_pending = self.coordinate_refresh_time()

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())

            if adapt_t_max:
                recent_windows += 1
                if window_was_empty:
                    recent_empty_windows += 1
                    empty_windows += 1
                if proposal_happened:
                    recent_proposals += 1
                if proposal_accepted:
                    recent_accepted_proposals += 1

                if recent_windows >= adapt_window:
                    empty_fraction = recent_empty_windows / recent_windows
                    acceptance_fraction = (
                        recent_accepted_proposals / recent_proposals
                        if recent_proposals > 0 else 1.0
                    )

                    updated = False
                    if empty_fraction > empty_window_threshold:
                        new_t_max = min(1.1 * t_max, t_max_max)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True
                    elif recent_proposals > 0 and acceptance_fraction < acceptance_threshold:
                        new_t_max = max(0.95 * t_max, t_max_min)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True

                    if updated:
                        t_max_arr.append(t_max)

                    recent_empty_windows = 0
                    recent_windows = 0
                    recent_proposals = 0
                    recent_accepted_proposals = 0

        pbar.close()
        out = {
            "t": np.array(t_skeleton),
            "x": np.array(x_skeleton),
            "accepted_switches": accepted_switches,
            "rejected_switches": rejected_switches,
            "refreshes": refreshes,
            "bound_violations": bound_violations,
            "method": "factorized_automatic_local_constant",
            "x_ref": self.x_ref.copy(),
            "t_max": t_max_arr,
            "windows": windows,
            "empty_windows": empty_windows,
            "gradient_evals": total_gradient_evals,
            "horizon_history": np.array(horizon_history),
            "lam_bar_history": np.array(lam_bar_history),
            "window_times": np.array(window_times),
            "window_x_history": np.array(window_x_history),
            "window_v_history": np.array(window_v_history),
        }

        if return_velocities:
            out["v"] = np.array(v_skeleton)

        return out


class StickyAutomaticBoomerangSampler(AutomaticBoomerangSampler):
    """
    Sticky automatic Boomerang sampler.

    This extends AutomaticBoomerangSampler by allowing coordinates to freeze
    at x_i = 0 and thaw later at rate kappa_i * |v_i|, in the spirit of the
    sticky PDMP construction of Bierkens et al. (2023).

    Design choices
    --------------
    - Active coordinates evolve with the usual Boomerang orbit.
    - Frozen coordinates are held fixed at x_i = 0 and v_i = 0.
    - When a coordinate hits 0 under the deterministic flow, it freezes and
      stores the incoming velocity. This stored velocity is used both for the
      thaw rate and for the velocity restored at thaw.
    - While some coordinates are frozen, reflections and refreshes are applied
      only on the active block.

    Parameters
    ----------
    kappa : float or array_like
        Sticky thaw-rate parameter(s). If scalar, it is broadcast to all
        coordinates.
    freeze_tol : float
        Numerical tolerance for identifying hits at x_i = 0.
    root_tol : float
        Minimum positive root used when computing hitting times.
    """

    def __init__(
        self,
        E,
        dim,
        kappa=1.0,
        refresh_rate=1.0,
        m1=None,
        gradE=None,
        freeze_tol=1e-12,
        root_tol=1e-10,
    ):
        super().__init__(E=E, dim=dim, refresh_rate=refresh_rate, m1=m1, gradE=gradE)

        kappa_arr = np.asarray(kappa, dtype=float)
        if kappa_arr.ndim == 0:
            kappa_arr = np.full(dim, float(kappa_arr))
        if kappa_arr.shape != (dim,):
            raise ValueError("kappa must be a scalar or an array of shape (dim,).")
        if np.any(kappa_arr <= 0):
            raise ValueError("All kappa values must be strictly positive.")

        self.kappa = kappa_arr
        self.freeze_tol = float(freeze_tol)
        self.root_tol = float(root_tol)

    # ------------------------------------------------------------
    # Sticky helpers
    # ------------------------------------------------------------

    def _normalize_frozen_mask(self, frozen_mask):
        frozen_mask = np.asarray(frozen_mask, dtype=bool)
        if frozen_mask.shape != (self.dim,):
            raise ValueError("frozen_mask must have shape (dim,).")
        return frozen_mask

    def _state_on_orbit_sticky(self, x, v, frozen_mask, dt):
        """
        Deterministic state after dt when frozen coordinates are held fixed.
        """
        frozen_mask = self._normalize_frozen_mask(frozen_mask)

        x_new = np.array(x, dtype=float, copy=True)
        v_new = np.array(v, dtype=float, copy=True)

        active = ~frozen_mask
        if np.any(active):
            y = x_new[active] - self.x_ref[active]
            c = np.cos(dt)
            s = np.sin(dt)
            y_new = c * y + s * v_new[active]
            v_active_new = -s * y + c * v_new[active]
            x_new[active] = y_new + self.x_ref[active]
            v_new[active] = v_active_new

        x_new[frozen_mask] = 0.0
        v_new[frozen_mask] = 0.0
        return x_new, v_new

    def _sample_active_velocity(self, frozen_mask):
        """
        Sample a Gaussian velocity on the active block only.
        Frozen coordinates remain at zero.
        """
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        v = np.zeros(self.dim, dtype=float)
        active = np.flatnonzero(~frozen_mask)
        if len(active) == 0:
            return v

        Sigma_active = self.Sigma[np.ix_(active, active)]
        jitter = 1e-12 * np.eye(len(active))
        chol = np.linalg.cholesky(Sigma_active + jitter)
        v[active] = chol @ np.random.randn(len(active))
        return v

    def _reflect_velocity_active(self, v, gradU, frozen_mask):
        """
        Boomerang reflection restricted to the active block.

        When no coordinates are frozen, this reduces to the parent reflection.
        """
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        if not np.any(frozen_mask):
            return self.reflect_velocity(v, gradU)

        active = np.flatnonzero(~frozen_mask)
        if len(active) == 0:
            return np.zeros_like(v)

        v_new = np.array(v, dtype=float, copy=True)
        grad_active = np.asarray(gradU, dtype=float)[active]
        v_active = v_new[active]
        rate = float(np.dot(v_active, grad_active))

        Sigma_active = self.Sigma[np.ix_(active, active)]
        denom = float(grad_active @ Sigma_active @ grad_active)
        if denom <= 1e-14:
            v_new[frozen_mask] = 0.0
            return v_new

        v_new[active] = v_active - 2.0 * rate / denom * (Sigma_active @ grad_active)
        v_new[frozen_mask] = 0.0
        return v_new

    def _coordinate_hitting_time(self, x_i, v_i, x_ref_i):
        """
        Smallest positive t such that the active Boomerang orbit hits x_i(t)=0.

        The active-coordinate orbit is
            x_i(t) = x_ref_i + (x_i - x_ref_i) cos(t) + v_i sin(t).
        """
        a = float(x_i - x_ref_i)
        b = float(v_i)
        c = float(-x_ref_i)

        R = np.hypot(a, b)
        if R <= self.freeze_tol:
            return np.inf
        if abs(c) > R + 1e-14:
            return np.inf

        ratio = np.clip(c / R, -1.0, 1.0)
        phi = np.arctan2(b, a)
        delta = np.arccos(ratio)

        candidates = []
        for base in (phi - delta, phi + delta):
            for k in range(-1, 3):
                t = base + 2.0 * np.pi * k
                if t > self.root_tol:
                    candidates.append(t)

        if not candidates:
            return np.inf

        t_hit = min(candidates)
        # guard against numerical false positives
        x_hit = x_ref_i + a * np.cos(t_hit) + b * np.sin(t_hit)
        if abs(x_hit) > 1e-7:
            return np.inf
        return t_hit

    def _next_hitting_event(self, x, v, frozen_mask):
        """
        First deterministic hitting time of x_i = 0 among active coordinates.
        """
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        t_hit = np.inf
        i_hit = None

        for i in np.flatnonzero(~frozen_mask):
            ti = self._coordinate_hitting_time(x[i], v[i], self.x_ref[i])
            if ti < t_hit:
                t_hit = ti
                i_hit = int(i)

        return t_hit, i_hit

    def _sample_thaw_event(self, frozen_mask, frozen_velocity):
        """
        First thaw event among currently frozen coordinates.
        """
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        frozen = np.flatnonzero(frozen_mask)
        if len(frozen) == 0:
            return np.inf, None

        thaw_rates = self.kappa[frozen] * np.abs(frozen_velocity[frozen])
        total_rate = float(np.sum(thaw_rates))
        if total_rate <= 1e-14:
            return np.inf, None

        tau = np.random.exponential(1.0 / total_rate)
        probs = thaw_rates / total_rate
        idx = int(np.random.choice(len(frozen), p=probs))
        return tau, int(frozen[idx])

    def _max_rate_on_interval_sticky(
        self,
        x,
        v,
        frozen_mask,
        horizon,
        n_grid,
        safety,
        refine,
        n_candidates=2,
    ):
        """
        Automatic local constant bound along the sticky Boomerang orbit,
        assuming the frozen set stays unchanged on [0, horizon].
        """
        if horizon <= 0:
            return 0.0, np.array([0.0]), np.array([0.0]), 0

        ts = np.linspace(0.0, horizon, n_grid + 1)
        vals = np.empty_like(ts)

        for k, tk in enumerate(ts):
            xt, vt = self._state_on_orbit_sticky(x, v, frozen_mask, tk)
            vals[k] = self.rate(xt, vt)
        grad_evals = len(vals)

        tol = 1e-10 + 1e-6 * np.max(np.abs(vals))
        if np.max(vals) <= tol:
            return 0.0, ts, vals, grad_evals

        left_diffs = np.diff(vals)
        increasing = np.all(left_diffs >= -tol)
        decreasing = np.all(left_diffs <= tol)

        if increasing:
            lam_max = vals[-1]
            return max(safety * lam_max, 0.0), ts, vals, grad_evals
        if decreasing:
            lam_max = vals[0]
            return max(safety * lam_max, 0.0), ts, vals, grad_evals

        lam_max = float(np.max(vals))

        if refine and horizon > 1e-12:
            candidate_idxs = np.argsort(vals)[-n_candidates:][::-1]
            for max_idx in candidate_idxs:
                left_idx = max(0, max_idx - 1)
                right_idx = min(len(ts) - 1, max_idx + 1)
                a = ts[left_idx]
                b = ts[right_idx]
                if b - a <= 1e-12:
                    continue

                local_evals = 0

                def obj(tau):
                    nonlocal local_evals
                    local_evals += 1
                    xt, vt = self._state_on_orbit_sticky(x, v, frozen_mask, tau)
                    return -self.rate(xt, vt)

                res = minimize_scalar(obj, bounds=(a, b), method="bounded")
                grad_evals += local_evals
                if res.success and np.isfinite(res.fun):
                    lam_max = max(lam_max, -float(res.fun))

        return max(safety * lam_max, 0.0), ts, vals, grad_evals

    def _initial_sticky_state(self, x_init=None, v_init=None, frozen_mask_init=None):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")

        x = self.x_ref.copy() if x_init is None else np.array(x_init, dtype=float, copy=True)
        if x.shape != (self.dim,):
            raise ValueError("x_init must have shape (dim,).")

        if frozen_mask_init is None:
            frozen_mask = np.zeros(self.dim, dtype=bool)
        else:
            frozen_mask = self._normalize_frozen_mask(frozen_mask_init)

        if v_init is None:
            v = self._sample_active_velocity(frozen_mask)
            frozen_velocity = v.copy()
            frozen_velocity[frozen_mask] = 0.0
            for i in np.flatnonzero(frozen_mask):
                sigma_ii = max(self.Sigma[i, i], 1e-14)
                frozen_velocity[i] = np.sqrt(sigma_ii) * np.random.randn()
        else:
            v = np.array(v_init, dtype=float, copy=True)
            if v.shape != (self.dim,):
                raise ValueError("v_init must have shape (dim,).")
            frozen_velocity = v.copy()

        x[frozen_mask] = 0.0
        v[frozen_mask] = 0.0
        return x, v, frozen_mask, frozen_velocity

    # ------------------------------------------------------------
    # Sticky automatic sampler
    # ------------------------------------------------------------

    def sample_auto(
        self,
        T=10.0,
        t_max=0.5,
        n_grid=16,
        safety=1.10,
        refine=True,
        adapt_t_max=False,
        t_max_min=1e-3,
        t_max_max=5.0,
        bound_violation_tol=1e-10,
        x_init=None,
        v_init=None,
        frozen_mask_init=None,
        return_frozen_history=True,
    ):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        if safety < 1.0:
            raise ValueError("safety should be >= 1.0.")

        x, v, frozen_mask, frozen_velocity = self._initial_sticky_state(
            x_init=x_init,
            v_init=v_init,
            frozen_mask_init=frozen_mask_init,
        )
        t = 0.0

        x_skeleton = [x.copy()]
        v_skeleton = [v.copy()]
        t_skeleton = [t]
        if return_frozen_history:
            frozen_skeleton = [frozen_mask.copy()]

        accepted_switches = 0
        rejected_switches = 0
        refreshes = 0
        freezes = 0
        thaws = 0
        total_gradient_evals = 0
        bound_violations = 0

        dt_refresh = np.random.exponential(1.0 / self.refresh_rate) if self.refresh_rate > 0 else np.inf

        t_max_arr = [t_max]
        windows = 0
        empty_windows = 0

        adapt_window = 20
        recent_empty_windows = 0
        recent_windows = 0
        recent_proposals = 0
        recent_accepted_proposals = 0

        empty_window_threshold = 0.6
        acceptance_threshold = 0.3

        horizon_history = []
        lam_bar_history = []
        window_times = []
        window_x_history = []
        window_v_history = []
        window_frozen_history = []

        pbar = tqdm(total=T, desc="Sampling", leave=True)
        while t < T:
            windows += 1

            dt_hit, i_hit = self._next_hitting_event(x, v, frozen_mask)
            dt_thaw, i_thaw = self._sample_thaw_event(frozen_mask, frozen_velocity)
            horizon = min(t_max, dt_refresh, dt_hit, dt_thaw, T - t)

            window_times.append(t)
            window_x_history.append(x.copy())
            window_v_history.append(v.copy())
            window_frozen_history.append(frozen_mask.copy())

            lam_bar, _, _, gradient_evals = self._max_rate_on_interval_sticky(
                x=x,
                v=v,
                frozen_mask=frozen_mask,
                horizon=horizon,
                n_grid=n_grid,
                safety=safety,
                refine=refine,
            )
            total_gradient_evals += gradient_evals

            horizon_history.append(horizon)
            lam_bar_history.append(lam_bar)

            window_was_empty = False
            proposal_happened = False
            proposal_accepted = False

            if lam_bar <= 1e-14:
                tau = np.inf
            else:
                tau = np.random.exponential(1.0 / lam_bar)

            event_is_bound = tau < horizon

            if event_is_bound:
                proposal_happened = True

                x_prop, v_prop = self._state_on_orbit_sticky(x, v, frozen_mask, tau)
                true_gradU = self.gradU(x_prop)
                true_rate = max(float(np.dot(v_prop, true_gradU)), 0.0)
                total_gradient_evals += 1

                if true_rate > lam_bar * (1.0 + bound_violation_tol):
                    bound_violations += 1
                    if adapt_t_max:
                        t_max = max(0.5 * t_max, t_max_min)
                        t_max_arr.append(t_max)
                    raise RuntimeError(
                        f"Sticky automatic local bound violated: true_rate={true_rate:.6e}, "
                        f"lam_bar={lam_bar:.6e}. Increase safety / grid, or reduce t_max."
                    )

                x = x_prop
                t += tau
                dt_refresh -= tau
                
                pbar.update(t - pbar.n)

                if np.random.rand() * lam_bar <= true_rate:
                    v = self._reflect_velocity_active(v_prop, true_gradU, frozen_mask)
                    accepted_switches += 1
                    proposal_accepted = True
                else:
                    v = v_prop
                    rejected_switches += 1

                x_skeleton.append(x.copy())
                v_skeleton.append(v.copy())
                t_skeleton.append(t)
                if return_frozen_history:
                    frozen_skeleton.append(frozen_mask.copy())

            else:
                cause = "window_end"
                tol_event = 1e-12
                if abs(horizon - dt_hit) <= tol_event and i_hit is not None:
                    cause = "hit"
                elif abs(horizon - dt_thaw) <= tol_event and i_thaw is not None:
                    cause = "thaw"
                elif abs(horizon - dt_refresh) <= tol_event and self.refresh_rate > 0:
                    cause = "refresh"
                elif abs(horizon - (T - t)) <= tol_event:
                    cause = "terminal"

                x, v = self._state_on_orbit_sticky(x, v, frozen_mask, horizon)
                t += horizon
                dt_refresh -= horizon
                window_was_empty = True
                
                pbar.update(t - pbar.n)

                if cause == "terminal":
                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())
                    break

                if cause == "hit":
                    frozen_mask[i_hit] = True
                    frozen_velocity[i_hit] = v[i_hit]
                    x[i_hit] = 0.0
                    v[i_hit] = 0.0
                    freezes += 1

                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

                elif cause == "thaw":
                    frozen_mask[i_thaw] = False
                    x[i_thaw] = 0.0
                    v[i_thaw] = frozen_velocity[i_thaw]
                    thaws += 1

                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

                elif cause == "refresh" and t < T and self.refresh_rate > 0:
                    v_refresh = self._sample_active_velocity(frozen_mask)
                    v[~frozen_mask] = v_refresh[~frozen_mask]
                    refreshes += 1
                    dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

                    x_skeleton.append(x.copy())
                    v_skeleton.append(v.copy())
                    t_skeleton.append(t)
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

            if dt_refresh <= 1e-14 and t < T and self.refresh_rate > 0 and event_is_bound:
                v_refresh = self._sample_active_velocity(frozen_mask)
                v[~frozen_mask] = v_refresh[~frozen_mask]
                refreshes += 1
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

                x_skeleton.append(x.copy())
                v_skeleton.append(v.copy())
                t_skeleton.append(t)
                if return_frozen_history:
                    frozen_skeleton.append(frozen_mask.copy())

            if adapt_t_max:
                recent_windows += 1
                if window_was_empty:
                    recent_empty_windows += 1
                    empty_windows += 1
                if proposal_happened:
                    recent_proposals += 1
                if proposal_accepted:
                    recent_accepted_proposals += 1

                if recent_windows >= adapt_window:
                    empty_fraction = recent_empty_windows / recent_windows
                    acceptance_fraction = (
                        recent_accepted_proposals / recent_proposals
                        if recent_proposals > 0 else 1.0
                    )

                    updated = False
                    if empty_fraction > empty_window_threshold:
                        new_t_max = min(1.1 * t_max, t_max_max)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True
                    elif recent_proposals > 0 and acceptance_fraction < acceptance_threshold:
                        new_t_max = max(0.95 * t_max, t_max_min)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True

                    if updated:
                        t_max_arr.append(t_max)

                    recent_empty_windows = 0
                    recent_windows = 0
                    recent_proposals = 0
                    recent_accepted_proposals = 0

            if dt_refresh <= 1e-14 and self.refresh_rate > 0:
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)
                
        pbar.close()

        out = {
            "t": np.array(t_skeleton),
            "x": np.array(x_skeleton),
            "v": np.array(v_skeleton),
            "x_ref": self.x_ref.copy(),
            "accepted_switches": accepted_switches,
            "rejected_switches": rejected_switches,
            "refreshes": refreshes,
            "freezes": freezes,
            "thaws": thaws,
            "bound_violations": bound_violations,
            "method": "sticky_automatic_local_constant",
            "kappa": self.kappa.copy(),
            "t_max": t_max_arr,
            "windows": windows,
            "empty_windows": empty_windows,
            "gradient_evals": total_gradient_evals,
            "horizon_history": np.array(horizon_history),
            "lam_bar_history": np.array(lam_bar_history),
            "window_times": np.array(window_times),
            "window_x_history": np.array(window_x_history),
            "window_v_history": np.array(window_v_history),
            "window_frozen_history": np.array(window_frozen_history),
        }
        if return_frozen_history:
            out["frozen_mask"] = np.array(frozen_skeleton)

        return out


class FactorizedStickyAutomaticBoomerangSampler(StickyAutomaticBoomerangSampler):
    """
    Sticky + factorized automatic Boomerang sampler.

    This combines the coordinate-wise event structure of the Factorized
    Boomerang Sampler with the freeze/thaw mechanism of the sticky sampler.

    Relative to StickyAutomaticBoomerangSampler, this class:
      - enforces a diagonal Gaussian reference covariance,
      - uses coordinate-wise switching rates
            lambda_i(x, v) = (v_i * partial_i U(x))_+,
      - flips only the triggering velocity coordinate at accepted events,
      - refreshes active velocity coordinates independently,
      - keeps the sticky freeze/thaw logic for coordinates that hit x_i = 0.
    """

    def __init__(
        self,
        E,
        dim,
        kappa=1.0,
        refresh_rate=1.0,
        m1=None,
        gradE=None,
        partial_gradE=None,
        freeze_tol=1e-12,
        root_tol=1e-10,
    ):
        super().__init__(
            E=E,
            dim=dim,
            kappa=kappa,
            refresh_rate=refresh_rate,
            m1=m1,
            gradE=gradE,
            freeze_tol=freeze_tol,
            root_tol=root_tol,
        )
        self._user_partial_gradE = partial_gradE

    # ------------------------------------------------------------
    # Diagonal preprocessing (same restriction as factorized Boomerang)
    # ------------------------------------------------------------
    def preprocess(
        self,
        x0=None,
        method="diagonal",
        x_ref=None,
        Sigma=None,
        Sigma_inv=None,
        jitter=1e-8,
        estimate_m1=True,
        m1_samples=20,
        m1_scale=1.0,
    ):
        if x0 is None:
            x0 = np.zeros(self.dim)

        method = method.lower()

        if method in {"diagonal", "laplace"}:
            res = minimize(self.E, x0, jac=lambda x: np.array(self.gradE(x)))
            x_mode = np.array(res.x, dtype=float)
            H_ref = np.array(self.hessE(x_mode), dtype=float)

            if method == "diagonal":
                diag_H = np.clip(np.diag(H_ref), jitter, None)
                precision = np.diag(diag_H)
            else:
                offdiag = H_ref - np.diag(np.diag(H_ref))
                if not np.allclose(offdiag, 0.0, atol=1e-10):
                    raise ValueError(
                        "FactorizedStickyAutomaticBoomerangSampler requires a diagonal "
                        "reference covariance. Use method='diagonal' or pass a diagonal "
                        "manual reference."
                    )
                precision = H_ref

            self._set_reference_from_precision(x_mode, precision, jitter=jitter)
            sigma_diag = np.clip(np.diag(self.Sigma), jitter, None)
            self.Sigma = np.diag(sigma_diag)
            self.Sigma_inv = np.diag(1.0 / sigma_diag)
            self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))

        elif method == "manual":
            if x_ref is None:
                raise ValueError("For method='manual', x_ref must be provided.")
            if (Sigma is None) == (Sigma_inv is None):
                raise ValueError(
                    "For method='manual', provide exactly one of Sigma or Sigma_inv."
                )

            self.x_ref = np.array(x_ref, dtype=float)
            if Sigma_inv is not None:
                precision = np.array(Sigma_inv, dtype=float)
                if not np.allclose(precision, np.diag(np.diag(precision)), atol=1e-10):
                    raise ValueError(
                        "Manual Sigma_inv must be diagonal for factorized sticky Boomerang."
                    )
                diag_precision = np.clip(np.diag(precision), jitter, None)
                self.Sigma_inv = np.diag(diag_precision)
                sigma_diag = 1.0 / diag_precision
                self.Sigma = np.diag(sigma_diag)
                self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))
            else:
                Sigma = np.array(Sigma, dtype=float)
                if not np.allclose(Sigma, np.diag(np.diag(Sigma)), atol=1e-10):
                    raise ValueError(
                        "Manual Sigma must be diagonal for factorized sticky Boomerang."
                    )
                sigma_diag = np.clip(np.diag(Sigma), jitter, None)
                self.Sigma = np.diag(sigma_diag)
                self.Sigma_inv = np.diag(1.0 / sigma_diag)
                self.Sigma_sqrt = np.diag(np.sqrt(sigma_diag))
        else:
            raise ValueError(
                "Unknown preprocess method. Use 'diagonal', 'laplace', or 'manual'."
            )

        if estimate_m1 and self.m1 is None:
            self._estimate_m1(n_samples=m1_samples, proposal_scale=m1_scale)

    # ------------------------------------------------------------
    # Factorized / sticky helpers
    # ------------------------------------------------------------
    # def partial_gradU(self, x, i):
    #     return self.gradU(x)[i]
    
    def partial_gradU(self, x, i):
        if self._user_partial_gradE is not None:
            dEi = float(self._user_partial_gradE(x, i))
        else:
            dEi = float(self.gradE(x)[i])
        return dEi - self.Sigma_inv[i, i] * (x[i] - self.x_ref[i])

    def partial_rate(self, x, v, i):
        return max(float(v[i] * self.partial_gradU(x, i)), 0.0)

    def sample_velocity(self):
        return np.diag(self.Sigma_sqrt) * np.random.randn(self.dim)

    def _sample_active_velocity(self, frozen_mask):
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        v = np.zeros(self.dim, dtype=float)
        active = np.flatnonzero(~frozen_mask)
        if len(active) == 0:
            return v
        sigma_diag = np.diag(self.Sigma_sqrt)
        v[active] = sigma_diag[active] * np.random.randn(len(active))
        return v

    def _initial_sticky_state(self, x_init=None, v_init=None, frozen_mask_init=None):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")

        x = self.x_ref.copy() if x_init is None else np.array(x_init, dtype=float, copy=True)
        if x.shape != (self.dim,):
            raise ValueError("x_init must have shape (dim,).")

        if frozen_mask_init is None:
            frozen_mask = np.zeros(self.dim, dtype=bool)
        else:
            frozen_mask = self._normalize_frozen_mask(frozen_mask_init)

        if v_init is None:
            v = self._sample_active_velocity(frozen_mask)
            frozen_velocity = np.zeros(self.dim, dtype=float)
            sigma_diag = np.diag(self.Sigma_sqrt)
            for i in np.flatnonzero(frozen_mask):
                frozen_velocity[i] = sigma_diag[i] * np.random.randn()
        else:
            v = np.array(v_init, dtype=float, copy=True)
            if v.shape != (self.dim,):
                raise ValueError("v_init must have shape (dim,).")
            frozen_velocity = v.copy()

        x[frozen_mask] = 0.0
        v[frozen_mask] = 0.0
        return x, v, frozen_mask, frozen_velocity

    def coordinate_refresh_time(self, frozen_mask):
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        active = np.flatnonzero(~frozen_mask)
        if self.refresh_rate <= 0 or len(active) == 0:
            return np.inf, None

        i = int(np.random.choice(active))
        dt = np.random.exponential(1.0 / (len(active) * self.refresh_rate))
        return dt, i

    def _coordinate_rates_on_interval_sticky(self, x, v, frozen_mask, ts):
        frozen_mask = self._normalize_frozen_mask(frozen_mask)
        vals = np.zeros((len(ts), self.dim), dtype=float)
        active = ~frozen_mask
        for k, tk in enumerate(ts):
            xt, vt = self._state_on_orbit_sticky(x, v, frozen_mask, tk)
            if np.any(active):
                grad_xt = self.gradU(xt)
                vals[k, active] = np.maximum(vt[active] * grad_xt[active], 0.0)
        return vals, len(ts)

    def _max_coordinate_rates_on_interval_sticky(
        self,
        x,
        v,
        frozen_mask,
        horizon,
        n_grid,
        safety,
        refine,
        n_candidates=2,
    ):
        frozen_mask = self._normalize_frozen_mask(frozen_mask)

        if horizon <= 0:
            ts = np.array([0.0])
            vals = np.zeros((1, self.dim), dtype=float)
            return np.zeros(self.dim, dtype=float), ts, vals, 0

        ts = np.linspace(0.0, horizon, n_grid + 1)
        vals, grad_evals = self._coordinate_rates_on_interval_sticky(x, v, frozen_mask, ts)

        lam_max = np.max(vals, axis=0)
        tol = 1e-10 + 1e-6 * np.max(np.abs(vals))

        active_idx = np.flatnonzero(~frozen_mask)
        if refine and horizon > 1e-12 and np.max(lam_max) > tol:
            for i in active_idx:
                if lam_max[i] <= tol:
                    continue

                diffs = np.diff(vals[:, i])
                increasing = np.all(diffs >= -tol)
                decreasing = np.all(diffs <= tol)
                if increasing or decreasing:
                    continue

                candidate_idxs = np.argsort(vals[:, i])[-n_candidates:][::-1]
                for max_idx in candidate_idxs:
                    left_idx = max(0, max_idx - 1)
                    right_idx = min(len(ts) - 1, max_idx + 1)
                    a = ts[left_idx]
                    b = ts[right_idx]

                    if b - a <= 1e-12:
                        continue

                    local_evals = 0

                    def obj(tau):
                        nonlocal local_evals
                        local_evals += 1
                        xt, vt = self._state_on_orbit_sticky(x, v, frozen_mask, tau)
                        gi = self.partial_gradU(xt, i)
                        return -max(float(vt[i] * gi), 0.0)

                    res = minimize_scalar(obj, bounds=(a, b), method="bounded")
                    grad_evals += local_evals
                    if res.success and np.isfinite(res.fun):
                        lam_max[i] = max(lam_max[i], -float(res.fun))

        lam_max[frozen_mask] = 0.0
        lam_bar = safety * lam_max
        return np.maximum(lam_bar, 0.0), ts, vals, grad_evals

    def _sample_bound_event_time(self, lam_bar_vec):
        total_rate = float(np.sum(lam_bar_vec))
        if total_rate <= 1e-14:
            return np.inf, None

        tau = np.random.exponential(1.0 / total_rate)
        probs = lam_bar_vec / total_rate
        i = int(np.random.choice(self.dim, p=probs))
        return tau, i

    # ------------------------------------------------------------
    # Sticky + factorized automatic sampler
    # ------------------------------------------------------------
    def sample_auto(
        self,
        T=10.0,
        t_max=0.5,
        n_grid=16,
        safety=1.10,
        refine=True,
        adapt_t_max=False,
        t_max_min=1e-3,
        t_max_max=5.0,
        bound_violation_tol=1e-10,
        x_init=None,
        v_init=None,
        frozen_mask_init=None,
        return_frozen_history=True,
        return_velocities=True,
    ):
        if self.x_ref is None:
            raise RuntimeError("Run preprocess() first.")
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        if safety < 1.0:
            raise ValueError("safety should be >= 1.0.")

        x, v, frozen_mask, frozen_velocity = self._initial_sticky_state(
            x_init=x_init,
            v_init=v_init,
            frozen_mask_init=frozen_mask_init,
        )
        t = 0.0

        x_skeleton = [x.copy()]
        t_skeleton = [t]
        if return_velocities:
            v_skeleton = [v.copy()]
        if return_frozen_history:
            frozen_skeleton = [frozen_mask.copy()]

        accepted_switches = 0
        rejected_switches = 0
        refreshes = 0
        freezes = 0
        thaws = 0
        total_gradient_evals = 0
        bound_violations = 0

        dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

        t_max_arr = [t_max]
        windows = 0
        empty_windows = 0

        adapt_window = 20
        recent_empty_windows = 0
        recent_windows = 0
        recent_proposals = 0
        recent_accepted_proposals = 0

        empty_window_threshold = 0.6
        acceptance_threshold = 0.3

        horizon_history = []
        lam_bar_history = []
        window_times = []
        window_x_history = []
        window_v_history = []
        window_frozen_history = []

        pbar = tqdm(total=T, desc="Sampling", leave=True)
        while t < T:
            windows += 1

            dt_hit, i_hit = self._next_hitting_event(x, v, frozen_mask)
            dt_thaw, i_thaw = self._sample_thaw_event(frozen_mask, frozen_velocity)
            horizon = min(t_max, dt_refresh, dt_hit, dt_thaw, T - t)

            window_times.append(t)
            window_x_history.append(x.copy())
            window_v_history.append(v.copy())
            window_frozen_history.append(frozen_mask.copy())

            lam_bar_vec, _, _, gradient_evals = self._max_coordinate_rates_on_interval_sticky(
                x=x,
                v=v,
                frozen_mask=frozen_mask,
                horizon=horizon,
                n_grid=n_grid,
                safety=safety,
                refine=refine,
            )
            total_gradient_evals += gradient_evals

            horizon_history.append(horizon)
            lam_bar_history.append(lam_bar_vec.copy())

            window_was_empty = False
            proposal_happened = False
            proposal_accepted = False

            tau, i_switch = self._sample_bound_event_time(lam_bar_vec)
            event_is_bound = tau < horizon

            if event_is_bound:
                proposal_happened = True

                x_prop, v_prop = self._state_on_orbit_sticky(x, v, frozen_mask, tau)
                gi = self.partial_gradU(x_prop, i_switch)
                true_rate_i = max(float(v_prop[i_switch] * gi), 0.0)
                total_gradient_evals += 1

                if true_rate_i > lam_bar_vec[i_switch] * (1.0 + bound_violation_tol):
                    bound_violations += 1
                    if adapt_t_max:
                        t_max = max(0.5 * t_max, t_max_min)
                        t_max_arr.append(t_max)
                    raise RuntimeError(
                        f"Factorized sticky local bound violated in coordinate {i_switch}: "
                        f"true_rate={true_rate_i:.6e}, "
                        f"lam_bar={lam_bar_vec[i_switch]:.6e}. "
                        "Increase safety / grid, or reduce t_max."
                    )

                x = x_prop
                v = v_prop
                t += tau
                dt_refresh -= tau

                pbar.update(t - pbar.n)

                if np.random.rand() * lam_bar_vec[i_switch] <= true_rate_i:
                    v[i_switch] *= -1.0
                    accepted_switches += 1
                    proposal_accepted = True
                else:
                    rejected_switches += 1

                x_skeleton.append(x.copy())
                t_skeleton.append(t)
                if return_velocities:
                    v_skeleton.append(v.copy())
                if return_frozen_history:
                    frozen_skeleton.append(frozen_mask.copy())

            else:
                cause = "window_end"
                tol_event = 1e-12
                if abs(horizon - dt_hit) <= tol_event and i_hit is not None:
                    cause = "hit"
                elif abs(horizon - dt_thaw) <= tol_event and i_thaw is not None:
                    cause = "thaw"
                elif abs(horizon - dt_refresh) <= tol_event and i_refresh_pending is not None:
                    cause = "refresh"
                elif abs(horizon - (T - t)) <= tol_event:
                    cause = "terminal"

                x, v = self._state_on_orbit_sticky(x, v, frozen_mask, horizon)
                t += horizon
                dt_refresh -= horizon
                window_was_empty = True

                pbar.update(t - pbar.n)

                if cause == "terminal":
                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())
                    break

                if cause == "hit":
                    frozen_mask[i_hit] = True
                    frozen_velocity[i_hit] = v[i_hit]
                    x[i_hit] = 0.0
                    v[i_hit] = 0.0
                    freezes += 1

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

                    dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

                elif cause == "thaw":
                    frozen_mask[i_thaw] = False
                    x[i_thaw] = 0.0
                    v[i_thaw] = frozen_velocity[i_thaw]
                    thaws += 1

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

                    dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

                elif cause == "refresh" and t < T and i_refresh_pending is not None:
                    sigma_i = self.Sigma_sqrt[i_refresh_pending, i_refresh_pending]
                    v[i_refresh_pending] = sigma_i * np.random.randn()
                    refreshes += 1
                    dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

                    x_skeleton.append(x.copy())
                    t_skeleton.append(t)
                    if return_velocities:
                        v_skeleton.append(v.copy())
                    if return_frozen_history:
                        frozen_skeleton.append(frozen_mask.copy())

            if dt_refresh <= 1e-14 and t < T and i_refresh_pending is not None and event_is_bound:
                sigma_i = self.Sigma_sqrt[i_refresh_pending, i_refresh_pending]
                v[i_refresh_pending] = sigma_i * np.random.randn()
                refreshes += 1
                dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

                x_skeleton.append(x.copy())
                t_skeleton.append(t)
                if return_velocities:
                    v_skeleton.append(v.copy())
                if return_frozen_history:
                    frozen_skeleton.append(frozen_mask.copy())

            if adapt_t_max:
                recent_windows += 1
                if window_was_empty:
                    recent_empty_windows += 1
                    empty_windows += 1
                if proposal_happened:
                    recent_proposals += 1
                if proposal_accepted:
                    recent_accepted_proposals += 1

                if recent_windows >= adapt_window:
                    empty_fraction = recent_empty_windows / recent_windows
                    acceptance_fraction = (
                        recent_accepted_proposals / recent_proposals
                        if recent_proposals > 0 else 1.0
                    )

                    updated = False
                    if empty_fraction > empty_window_threshold:
                        new_t_max = min(1.1 * t_max, t_max_max)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True
                    elif recent_proposals > 0 and acceptance_fraction < acceptance_threshold:
                        new_t_max = max(0.95 * t_max, t_max_min)
                        if new_t_max != t_max:
                            t_max = new_t_max
                            updated = True

                    if updated:
                        t_max_arr.append(t_max)

                    recent_empty_windows = 0
                    recent_windows = 0
                    recent_proposals = 0
                    recent_accepted_proposals = 0

            if dt_refresh <= 1e-14 and i_refresh_pending is None:
                dt_refresh, i_refresh_pending = self.coordinate_refresh_time(frozen_mask)

        pbar.close()

        out = {
            "t": np.array(t_skeleton),
            "x": np.array(x_skeleton),
            "x_ref": self.x_ref.copy(),
            "accepted_switches": accepted_switches,
            "rejected_switches": rejected_switches,
            "refreshes": refreshes,
            "freezes": freezes,
            "thaws": thaws,
            "bound_violations": bound_violations,
            "method": "factorized_sticky_automatic_local_constant",
            "kappa": self.kappa.copy(),
            "t_max": t_max_arr,
            "windows": windows,
            "empty_windows": empty_windows,
            "gradient_evals": total_gradient_evals,
            "horizon_history": np.array(horizon_history),
            "lam_bar_history": np.array(lam_bar_history),
            "window_times": np.array(window_times),
            "window_x_history": np.array(window_x_history),
            "window_v_history": np.array(window_v_history),
            "window_frozen_history": np.array(window_frozen_history),
        }
        if return_velocities:
            out["v"] = np.array(v_skeleton)
        if return_frozen_history:
            out["frozen_mask"] = np.array(frozen_skeleton)

        return out

