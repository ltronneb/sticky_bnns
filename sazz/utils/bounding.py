import numpy as np
import torch

def brent(f, a, b, tol=1e-5, rel_tol=1e-5,maxiter = 100, diagnostics=True):
    """
    Given a function f, and an interval [a,b], find arg min f(x) for x in [a,b]
    Parameters:
    - f: function of a single input, time
    - a: minimum of search interval
    - b: maximum of search interval
    - tol: absolute tolerance
    - rel_tol: relative tolerance
    - maxiter: maximum number of iterations to perform

    Returns:
    - x: point where minimum is attained
    """

    # Brent minimizer
    phi = (1 + 5 ** 0.5) / 2

    x = w = v = a + (b - a) / 2
    f_x = f_w = f_v = f(x)
    
    n_rate_evals = 1

    # Now for the algorithm itself
    iteration = 0
    while np.abs(b - a) > tol + rel_tol * abs(x):
        u = None
        if x != w and x != v and w != v:
            numerator = (x - w) ** 2 * (f_x - f_v) - (x - v) ** 2 * (f_x - f_w)
            denominator = 2 * ((x - w) * (f_x - f_v) - (x - v) * (f_x - f_w))
            if denominator != 0:
                u = x - numerator / denominator
        if u is not None and a < u < b and abs(u - x) >= tol:
            # Use this interpolated u further
            pass
        else:
            # Golden section
            if x < (a + b) / 2:
                u = x + (b - x) / phi
            else:
                u = x - (b - x) / phi
        # Evaluate f at u
        f_u = f(u)
        n_rate_evals += 1 

        # Update the interval
        if f_u < f_x:
            # Better min found
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            f_v, f_w, f_x = f_w, f_x, f_u
        else:
            # Interval reduction
            if u < x:
                a = u
            else:
                b = u
            if f_u <= f_w or w == x:
                v, w, = w, u
                f_v, f_w = f_w, f_u
            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u

        # Update iter
        iteration += 1
        if iteration > maxiter:
            break
    if diagnostics:
        stats = {
            'rate_evals': n_rate_evals,
        }
        return x, stats
    return x


def brent_monotone_aware(f, a, b, tol=1e-5, rel_tol=1e-5, maxiter=100,
                         eps=1e-6, diagnostics=True):
    """
    Brent minimizer with a pre-optimization monotonicity check (Corbella et al.).

    Before running the full algorithm, evaluates f at both endpoints and checks
    whether the function is monotonic in [a, b]. If so, returns the endpoint
    minimum immediately without further iterations.

    The monotonicity check works as follows:
      1. Evaluate f(a) and f(b).
      2. Run the first Brent iteration.
      3. If either bracket endpoint is unchanged, probe eps-inward from that
         endpoint to confirm the function approaches from below there.
      4. If monotone: return that endpoint value. Otherwise: continue as normal.
    """
    phi = (1 + 5 ** 0.5) / 2

    f_a = f(a)
    f_b = f(b)
    n_rate_evals = 2

    x = w = v = a + (b - a) / 2
    f_x = f_w = f_v = f(x)
    n_rate_evals += 1

    a0, b0 = a, b  # remember initial bracket

    iteration = 0
    monotone_checked = False

    while np.abs(b - a) > tol + rel_tol * abs(x):
        u = None
        if x != w and x != v and w != v:
            numerator = (x - w) ** 2 * (f_x - f_v) - (x - v) ** 2 * (f_x - f_w)
            denominator = 2 * ((x - w) * (f_x - f_v) - (x - v) * (f_x - f_w))
            if denominator != 0:
                u = x - numerator / denominator
        if u is not None and a < u < b and abs(u - x) >= tol:
            pass
        else:
            if x < (a + b) / 2:
                u = x + (b - x) / phi
            else:
                u = x - (b - x) / phi

        f_u = f(u)
        n_rate_evals += 1

        if f_u < f_x:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            f_v, f_w, f_x = f_w, f_x, f_u
        else:
            if u < x:
                a = u
            else:
                b = u
            if f_u <= f_w or w == x:
                v, w = w, u
                f_v, f_w = f_w, f_u
            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u

        iteration += 1

        # After first iteration: check for monotonicity
        if iteration == 1 and not monotone_checked:
            monotone_checked = True
            if a == a0:
                # Left endpoint unchanged — probe just inside a
                f_probe = f(a0 + eps)
                n_rate_evals += 1
                if f_probe >= f_a:
                    # f is non-decreasing from a: minimum is at a
                    if diagnostics:
                        return a0, {'rate_evals': n_rate_evals, 'monotone': True}
                    return a0
            elif b == b0:
                # Right endpoint unchanged — probe just inside b
                f_probe = f(b0 - eps)
                n_rate_evals += 1
                if f_probe >= f_b:
                    # f is non-decreasing toward b from below: minimum is at b
                    if diagnostics:
                        return b0, {'rate_evals': n_rate_evals, 'monotone': True}
                    return b0

        if iteration > maxiter:
            break

    if diagnostics:
        return x, {'rate_evals': n_rate_evals, 'monotone': False}
    return x


def grid_thinning(rate_batch, horizon, n_segments=20, base_rate=0.0, alpha=1.0,
                  max_iter=200, device=None, dtype=torch.float64,
                  diagnostics=True):
    """
    Thinning against a grid-based piecewise-constant bound (Andral & Kamatani 2024).

    Grid is fixed at window start; rejections do not re-anchor it. On segment
    [t_i, t_{i+1}] the bound is max(y_i, y_{i+1}, m_i), where m_i is the ordinate
    of the intersection of the endpoint tangents.

    rate_batch(t) takes a 1-D tensor of times and returns the SIGNED rate at each,
    differentiably. The knot derivatives come from autograd, so the bound costs
    one forward and one backward pass per window.
    """
    t = torch.linspace(0.0, horizon, n_segments + 1, device=device, dtype=dtype,
                       requires_grad=True)
    y = rate_batch(t)
    d, = torch.autograd.grad(y.sum(), t)          # y_i depends only on t_i

    with torch.no_grad():
        t0, t1, y0, y1, d0, d1 = t[:-1], t[1:], y[:-1], y[1:], d[:-1], d[1:]
        dd = d0 - d1
        ok = dd.abs() > 1e-14
        x = torch.where(ok, (y1 - y0 + d0 * t0 - d1 * t1) / torch.where(ok, dd, torch.ones_like(dd)), t0)
        x = torch.min(torch.max(x, t0), t1)
        m = torch.where(ok, d0 * x + y0 - d0 * t0, y0)

        bound = alpha * (torch.maximum(torch.maximum(y0, y1), m.clamp(min=0.0)) + base_rate)
        cum = torch.cat([bound.new_zeros(1), torch.cumsum(bound * (t1 - t0), 0)])

    # serial accept/reject: scalar and sequential, so run it on the host
    t_np = t.detach().cpu().numpy()
    bound_np = bound.cpu().numpy()
    cum_np = cum.cpu().numpy()

    e = 0.0
    n_proposals, n_rate_evals, max_ratio, bound_violations = 0, n_segments + 1, 0.0, 0
    result = np.inf

    for _ in range(max_iter):
        e += -np.log(1.0 - np.random.random())
        if e >= cum_np[-1]:
            break
        i = min(max(int(np.searchsorted(cum_np, e, side='right')) - 1, 0), n_segments - 1)
        if bound_np[i] <= 1e-14:
            break
        t_prop = t_np[i] + (e - cum_np[i]) / bound_np[i]
        if t_prop >= horizon:
            break

        n_proposals += 1
        n_rate_evals += 1
        with torch.no_grad():
            lam = rate_batch(torch.tensor([t_prop], device=device, dtype=dtype)).item()
        ratio = (max(lam, 0.0) + base_rate) / bound_np[i]
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0:
            bound_violations += 1
        if np.random.random() < ratio:
            result = t_prop
            break
    else:
        import warnings
        warnings.warn(
            f"grid_thinning: max_iter={max_iter} reached, horizon={horizon:.6f}.",
            RuntimeWarning,
        )

    if diagnostics:
        return result, {
            'accepted': result < np.inf,
            'rejected': result == np.inf,
            'proposals': n_proposals,
            'rate_evals': n_rate_evals,
            'max_ratio': max_ratio,
            'bound_violations': bound_violations,
            'tau': result,
        }
    return result


def piecewise_thinning_sinusoidal_second_order(rate_time, horizon, alpha=1.2,
                                   n_segments=8, R=2.0, max_iter=200,
                                   second_harmonic=False,
                                   diagnostics=True):
    """
    Piecewise-linear thinning with a sinusoidal-aware initial envelope.

    The Boomerang sampler's rate along its orbit is approximately
        r(t) = A + B1*sin(2t) + C1*cos(2t)
    with period pi. We fit A, B1, C1 from three rate evaluations at
    t = 0, pi/4, pi/2.

    If second_harmonic=True, we additionally fit
        r(t) += B2*sin(4t) + C2*cos(4t)
    using two more evaluations at t = pi/8 and t = 3pi/16, for a total
    of 5 gradient evaluations. This captures the dominant non-Gaussian
    correction to the rate.

    The fitted function is then sampled at n_segments points per period
    (no extra gradient evals) to build a tight piecewise-linear envelope.

    """
    rate_fn = lambda t: max(rate_time(t), 0.0)

    # --- Phase 1: Fit from gradient evaluations ---

    # First harmonic: 3 evaluations at t = 0, pi/4, pi/2
    # r(t) ~ A + B1*sin(2t) + C1*cos(2t)
    # r(0)    = A + C1
    # r(pi/4) = A + B1
    # r(pi/2) = A - C1
    r_at_0 = rate_fn(0.0)
    r_at_pi4 = rate_fn(np.pi / 4.0)
    r_at_pi2 = rate_fn(np.pi / 2.0)
    n_rate_evals = 3

    A = 0.5 * (r_at_0 + r_at_pi2)
    C1 = 0.5 * (r_at_0 - r_at_pi2)
    B1 = r_at_pi4 - A

    B2 = 0.0
    C2 = 0.0

    if second_harmonic:
        # Second harmonic: 2 more evaluations
        #
        # t = pi/8:  sin(4*pi/8) = 1,  cos(4*pi/8) = 0
        #   => residual = B2
        #
        # t = 3pi/16: sin(4*3pi/16) = sin(3pi/4) = sqrt(2)/2
        #             cos(4*3pi/16) = cos(3pi/4) = -sqrt(2)/2
        #   => residual = B2*sqrt(2)/2 - C2*sqrt(2)/2
        #
        t4 = np.pi / 8.0
        t5 = 3.0 * np.pi / 16.0

        r_at_t4 = rate_fn(t4)
        r_at_t5 = rate_fn(t5)
        n_rate_evals = 5

        # First harmonic prediction at these points
        pred_t4 = A + B1 * np.sin(2.0 * t4) + C1 * np.cos(2.0 * t4)
        pred_t5 = A + B1 * np.sin(2.0 * t5) + C1 * np.cos(2.0 * t5)

        # Residuals
        delta_t4 = r_at_t4 - pred_t4
        delta_t5 = r_at_t5 - pred_t5

        # Solve:  delta_t4 = B2 * sin(4*t4) + C2 * cos(4*t4)
        #         delta_t5 = B2 * sin(4*t5) + C2 * cos(4*t5)
        s4 = np.sin(4.0 * t4)  # sin(pi/2) = 1
        c4 = np.cos(4.0 * t4)  # cos(pi/2) = 0
        s5 = np.sin(4.0 * t5)  # sin(3pi/4) = sqrt(2)/2
        c5 = np.cos(4.0 * t5)  # cos(3pi/4) = -sqrt(2)/2

        det = s4 * c5 - s5 * c4
        if abs(det) > 1e-14:
            B2 = (delta_t4 * c5 - delta_t5 * c4) / det
            C2 = (s4 * delta_t5 - s5 * delta_t4) / det
        else:
            B2 = 0.0
            C2 = 0.0

    def fitted_rate(t):
        """Fitted approximation of the rate."""
        val = A + B1 * np.sin(2.0 * t) + C1 * np.cos(2.0 * t)
        if second_harmonic:
            val += B2 * np.sin(4.0 * t) + C2 * np.cos(4.0 * t)
        return val

    # --- Phase 2: Build piecewise-linear envelope ---
    # The first harmonic has period pi; the second has period pi/2.
    # Use pi as the tiling period (covers both).
    period = np.pi
    envelope_end = min(horizon, period)

    knot_times_1p = np.linspace(0.0, envelope_end, n_segments + 1)
    knot_values_1p = np.array([alpha * max(fitted_rate(t), 0.0)
                                for t in knot_times_1p])

    # Concavity fix: insert midpoint knots where chord underestimates
    refined_times = [knot_times_1p[0]]
    refined_values = [knot_values_1p[0]]
    for i in range(len(knot_times_1p) - 1):
        t0, t1 = knot_times_1p[i], knot_times_1p[i + 1]
        v0, v1 = knot_values_1p[i], knot_values_1p[i + 1]
        t_mid = 0.5 * (t0 + t1)
        v_mid_chord = 0.5 * (v0 + v1)
        v_mid_fit = alpha * max(fitted_rate(t_mid), 0.0)
        if v_mid_fit > v_mid_chord:
            refined_times.append(t_mid)
            refined_values.append(v_mid_fit)
        refined_times.append(t1)
        refined_values.append(v1)

    knot_times_1p = np.array(refined_times)
    knot_values_1p = np.array(refined_values)

    # Tile for horizons beyond one period
    if horizon > period:
        n_periods = int(np.ceil(horizon / period))
        all_times = list(knot_times_1p)
        all_values = list(knot_values_1p)
        for p in range(1, n_periods):
            offset = p * period
            for i in range(1, len(knot_times_1p)):
                t = offset + knot_times_1p[i]
                if t > horizon + 1e-14:
                    break
                all_times.append(t)
                all_values.append(knot_values_1p[i])
        knot_times = np.array(all_times)
        knot_values = np.array(all_values)
    else:
        knot_times = knot_times_1p
        knot_values = knot_values_1p

    # Build segments: (t_start, t_end, slope, intercept)
    segments_pending = []
    for i in range(len(knot_times) - 1):
        t0 = knot_times[i]
        t1 = knot_times[i + 1]
        v0 = knot_values[i]
        v1 = knot_values[i + 1]
        dt = t1 - t0
        if dt < 1e-14:
            continue
        slope = (v1 - v0) / dt
        segments_pending.append((t0, t1, slope, v0))

    # --- Phase 3: Thinning loop ---
    if not segments_pending:
        result = np.inf
        if diagnostics:
            return result, _make_stats(result, 0, n_rate_evals, 0.0, 0)
        return result

    t0_seg, t1_seg, a, b = segments_pending.pop(0)
    t_prev = t0_seg
    lam_prev = b

    n_proposals = 0
    max_ratio = 0.0
    bound_violations = 0
    result = np.inf

    for _ in range(max_iter):
        u = np.random.random()
        xi = -np.log(1.0 - u)

        # Sample from linear rate h(s) = b + a*s
        if abs(a) < 1e-14:
            if b < 1e-14:
                if segments_pending:
                    t0_seg, t1_seg, a, b = segments_pending.pop(0)
                    t_prev = t0_seg
                    lam_prev = b
                    continue
                result = np.inf
                break
            s = xi / b
        else:
            disc = b**2 + 2.0 * a * xi
            if disc < 0.0:
                if segments_pending:
                    t0_seg, t1_seg, a, b = segments_pending.pop(0)
                    t_prev = t0_seg
                    lam_prev = b
                    continue
                result = np.inf
                break
            s = (-b + np.sqrt(disc)) / a

        t_prop = t_prev + s

        if s < 1e-14:
            continue

        # Overshoot current segment — advance to next
        if segments_pending and t_prop > segments_pending[0][0]:
            t0_next, t1_next, a_next, b_next = segments_pending.pop(0)
            t_prev = t0_next
            lam_prev = b_next
            a, b = a_next, b_next
            continue

        if t_prop >= horizon:
            result = np.inf
            break

        # Evaluate true rate
        n_proposals += 1
        n_rate_evals += 1
        lam_true = rate_fn(t_prop)
        lam_knot = alpha * lam_true
        h_prop = b + a * s

        ratio = lam_true / h_prop if h_prop > 1e-14 else 0.0

        if ratio > 1.0:
            bound_violations += 1

        if ratio > R:
            # Subdivide with midpoint
            t_mid_ref = 0.5 * (t_prev + t_prop)
            lam_mid_ref = alpha * rate_fn(t_mid_ref)
            n_rate_evals += 1

            a = (lam_mid_ref - lam_prev) / (t_mid_ref - t_prev)
            b = lam_prev

            segments_pending.insert(0, (t_mid_ref, t_prop,
                (lam_knot - lam_mid_ref) / (t_prop - t_mid_ref), lam_mid_ref))
            continue

        if h_prop < 1e-14:
            if segments_pending:
                t0_seg, t1_seg, a, b = segments_pending.pop(0)
                t_prev = t0_seg
                lam_prev = b
                continue
            result = np.inf
            break

        max_ratio = max(max_ratio, ratio)

        if np.random.random() < ratio:
            result = t_prop
            break

        # Rejected: standard PLI update from here
        a = (lam_knot - lam_prev) / (t_prop - t_prev)
        b = lam_knot
        lam_prev = lam_knot
        t_prev = t_prop
        segments_pending = []

    if diagnostics:
        return result, _make_stats(result, n_proposals, n_rate_evals,
                                   max_ratio, bound_violations)
    return result


def piecewise_thinning(rate_time, horizon, alpha=1.2, t_init=None,
                       R=2.0, max_iter=200, diagnostics=True):
    if t_init is None:
        t_init = max(horizon / 100, 1e-6)

    rate_fn = lambda t: max(rate_time(t), 0.0)

    lam_prev = alpha * rate_fn(0.0)
    lam_init = alpha * rate_fn(t_init)

    a = (lam_init - lam_prev) / t_init
    b = lam_prev
    t_prev = 0.0

    # Pending segments: each is (t_start, t_end, slope, intercept_at_t_start)
    # Used when we subdivide on a bound violation — second half goes here.
    segments_pending = []

    n_proposals = 0
    n_rate_evals = 2
    max_ratio = 0.0
    bound_violations = 0
    result = np.inf

    for _ in range(max_iter):
        u = np.random.random()
        xi = -np.log(1.0 - u)

        if abs(a) < 1e-14:
            if b < 1e-14:
                # Flat zero rate on this segment — advance if we have a queued segment
                if segments_pending:
                    t0_seg, t1_seg, a, b = segments_pending.pop(0)
                    t_prev = t0_seg
                    lam_prev = b
                    continue
                result = np.inf
                break
            s = xi / b
        else:
            disc = b**2 + 2.0 * a * xi
            if disc < 0.0:
                if segments_pending:
                    t0_seg, t1_seg, a, b = segments_pending.pop(0)
                    t_prev = t0_seg
                    lam_prev = b
                    continue
                result = np.inf
                break
            s = (-b + np.sqrt(disc)) / a

        t_prop = t_prev + s

        # If we've crossed into a queued segment, switch to it
        if segments_pending and t_prop > segments_pending[0][0]:
            t0_next, t1_next, a_next, b_next = segments_pending.pop(0)
            t_prev = t0_next
            lam_prev = b_next
            a, b = a_next, b_next
            continue

        if t_prop >= horizon:
            result = np.inf
            break

        n_proposals += 1
        n_rate_evals += 1
        lam_true = rate_fn(t_prop)
        lam_knot = alpha * lam_true
        h_prop = b + a * s

        ratio = lam_true / h_prop if h_prop > 1e-14 else 0.0
        #max_ratio = max(max_ratio, ratio)
        if ratio > 1.0:
            bound_violations += 1

        if ratio > R:
            # SUBDIVIDE: evaluate at midpoint, queue second half, retry first half
            t_mid = 0.5 * (t_prev + t_prop)
            lam_mid = alpha * rate_fn(t_mid)
            n_rate_evals += 1

            # New first-half segment: t_prev -> t_mid
            a = (lam_mid - lam_prev) / (t_mid - t_prev)
            b = lam_prev

            # Queue second half: t_mid -> t_prop, with slope using lam_knot at t_prop
            slope_second = (lam_knot - lam_mid) / (t_prop - t_mid)
            segments_pending.insert(0, (t_mid, t_prop, slope_second, lam_mid))
            continue

        if h_prop < 1e-14:
            if segments_pending:
                t0_seg, t1_seg, a, b = segments_pending.pop(0)
                t_prev = t0_seg
                lam_prev = b
                continue
            result = np.inf
            break
        max_ratio = max(max_ratio, ratio)

        if np.random.random() < ratio:
            result = t_prop
            break

        # Rejected: standard PLI update, discard queued segments
        a = (lam_knot - lam_prev) / (t_prop - t_prev)
        b = lam_knot
        lam_prev = lam_knot
        t_prev = t_prop
        segments_pending = []
    else:
        import warnings
        warnings.warn(
            f"piecewise_thinning: max_iter={max_iter} reached. "
            f"Last t={t_prev:.6f}, horizon={horizon:.6f}.",
            RuntimeWarning,
        )
        result = np.inf

    if diagnostics:
        stats = {
            'accepted': result < np.inf,
            'rejected': result == np.inf,
            'proposals': n_proposals,
            'rate_evals': n_rate_evals,
            'max_ratio': max_ratio,
            'bound_violations': bound_violations,
            'tau': result,
        }
        return result, stats
    return result


def _make_stats(result, proposals, rate_evals, max_ratio, bound_violations):
    return {
        'accepted': result < np.inf,
        'rejected': result == np.inf,
        'proposals': proposals,
        'rate_evals': rate_evals,
        'max_ratio': max_ratio,
        'bound_violations': bound_violations,
        'tau': result,
    }
    
    
# def sample_trajectory_at_regular_intervals(Position, Velocity, EventTimes, dt):
#     """
#     Given event-based trajectory (Position, Velocity, EventTimes),
#     return the state at regular time grid with spacing dt.
#     """
#     # Create regular time grid
#     T_max = EventTimes[EventTimes > 0].max()
#     t_grid = np.arange(0, T_max + dt, dt)

#     # For each regular time, find the index of the last event time <= t
#     idx = np.searchsorted(EventTimes, t_grid, side='right') - 1
#     idx = np.clip(idx, 0, len(EventTimes) - 2)

#     # Compute time offset within each segment
#     tau = (t_grid - EventTimes[idx])[:, None]  # shape (M,1)

#     # Linear interpolation for position; velocity is piecewise constant
#     pos_grid = Position[idx] + Velocity[idx] * tau
#     vel_grid = Velocity[idx]

#     return t_grid, pos_grid, vel_grid
