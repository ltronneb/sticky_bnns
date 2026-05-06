import numpy as np

def piecewise_thinning(rate_time, horizon, alpha=1.2, t_init=None, 
                       R=2.0, max_iter=200, diagnostics=True):
    if t_init is None:
        t_init = horizon / 100.0

    rate_fn = lambda t: max(rate_time(t), 0.0)

    lam_prev = alpha * rate_fn(0.0)
    lam_init = alpha * rate_fn(t_init)

    a = (lam_init - lam_prev) / t_init
    b = lam_prev
    t_prev = 0.0

    n_proposals = 0
    n_rate_evals = 2  # the two initial evaluations
    max_ratio = 0.0
    bound_violations = 0

    for _ in range(max_iter):
        u = np.random.random()
        xi = -np.log(1.0 - u)

        if abs(a) < 1e-14:
            if b < 1e-14:
                result = np.inf
                break
            s = xi / b
        else:
            disc = b**2 + 2.0 * a * xi
            if disc < 0.0:
                result = np.inf
                break
            s = (-b + np.sqrt(disc)) / a

        t_prop = t_prev + s

        if t_prop >= horizon:
            result = np.inf
            break

        n_proposals += 1
        n_rate_evals += 1

        lam_true = rate_fn(t_prop)
        lam_knot = alpha * lam_true
        h_prop = b + a * s

        ratio = lam_true / h_prop if h_prop > 1e-14 else 0.0
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0:
            bound_violations += 1

        if ratio > R:
            a = (lam_knot - lam_prev) / (t_prop - t_prev)
            b = lam_prev
            #result = np.inf
            #break
            continue

        if h_prop < 1e-14:
            result = np.inf
            break

        if np.random.random() < ratio:
            result = t_prop
            break

        a = (lam_knot - lam_prev) / (t_prop - t_prev)
        b = lam_knot
        lam_prev = lam_knot
        t_prev = t_prop
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


def piecewise_thinning_midpoint(rate_time, horizon, alpha=1.5, t_init=None,
                                R=1.2, max_iter=200, diagnostics=True):
    if t_init is None:
        t_init = horizon / 500.0
 
    rate_fn = lambda t: max(rate_time(t), 0.0)
 
    # Evaluate endpoints and midpoint
    lam0 = alpha * rate_fn(0.0)
    t_mid = 0.5 * t_init
    lam_mid = alpha * rate_fn(t_mid)
    lam_init = alpha * rate_fn(t_init)
 
    n_rate_evals = 3  # three initial evaluations
 
    # Midpoint check: if the chord (0, lam0) -> (t_init, lam_init) underestimates
    # the true rate at the midpoint, split into two sub-chords.
    chord_at_mid = 0.5 * (lam0 + lam_init)
    if lam_mid > chord_at_mid:
        # use two-segment envelope; currently active segment is [0, t_mid]
        a = (lam_mid - lam0) / t_mid
        b = lam0
        segments_pending = [  # [(t0, t1, a, b), ...] for subsequent segments
            (t_mid, t_init, (lam_init - lam_mid) / (t_init - t_mid), lam_mid),
        ]
    else:
        a = (lam_init - lam0) / t_init
        b = lam0
        segments_pending = []
 
    lam_prev = lam0
    t_prev = 0.0
    n_proposals = 0
    max_ratio = 0.0
    bound_violations = 0
    result = np.inf
 
    for _ in range(max_iter):
        u = np.random.random()
        xi = -np.log(1.0 - u)
 
        if abs(a) < 1e-14:
            if b < 1e-14:
                result = np.inf
                break
            s = xi / b
        else:
            disc = b**2 + 2.0 * a * xi
            if disc < 0.0:
                result = np.inf
                break
            s = (-b + np.sqrt(disc)) / a
 
        t_prop = t_prev + s
 
        # If we overshoot the current segment and a pending segment exists,
        # advance to the next segment (standard piecewise-linear thinning).
        if segments_pending and t_prop > segments_pending[0][0]:
            # advance: consume the pending segment as the new active chord
            t0_next, t1_next, a_next, b_next = segments_pending.pop(0)
            # re-anchor time: reset t_prev to t0_next, update xi partially used
            t_prev = t0_next
            lam_prev = b_next
            a, b = a_next, b_next
            continue  # redraw u, attempt proposal in new segment
 
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
            # Subdivide: evaluate midpoint between t_prev and t_prop,
            # build two sub-segments for a tighter envelope.
            t_mid_ref = 0.5 * (t_prev + t_prop)
            lam_mid_ref = alpha * rate_fn(t_mid_ref)
            n_rate_evals += 1
 
            # First segment: (t_prev, lam_prev) -> (t_mid_ref, lam_mid_ref)
            a = (lam_mid_ref - lam_prev) / (t_mid_ref - t_prev)
            b = lam_prev
 
            # Queue second segment: (t_mid_ref, lam_mid_ref) -> (t_prop, lam_knot)
            segments_pending.insert(0, (t_mid_ref, t_prop,
                (lam_knot - lam_mid_ref) / (t_prop - t_mid_ref), lam_mid_ref))
            # t_prev and lam_prev stay the same — redraw from refined envelope
            continue
 
        if h_prop < 1e-14:
            result = np.inf
            break
        
        max_ratio = max(max_ratio, ratio)
        if np.random.random() < ratio:
            result = t_prop
            break
 
        # rejected: update chord between (t_prev, lam_prev) and (t_prop, lam_knot)
        a = (lam_knot - lam_prev) / (t_prop - t_prev)
        b = lam_knot
        lam_prev = lam_knot
        t_prev = t_prop
        segments_pending = []  # we're past the initial envelope now
 
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



def piecewise_thinning_sinusoidal(rate_time, horizon, alpha=1.2,
                                   n_segments=24, R=2.0, max_iter=200,
                                   diagnostics=True):
    """
    Piecewise-linear thinning with a sinusoidal-aware initial envelope.

    The Boomerang sampler's rate along its orbit is approximately
        r(t) = A + B*sin(2t) + C*cos(2t)
    with period pi.  We fit A, B, C from three rate evaluations at
    t = 0, pi/4, pi/2, then build a piecewise-linear upper envelope
    by evaluating the fitted sinusoid at many points (no extra gradient
    evals).

    To ensure the piecewise-linear segments don't underestimate the
    sinusoid between knots (concavity issue), we check the sinusoidal
    midpoint and insert an extra knot if the chord dips below it.

    After the initial envelope is consumed, the method falls back to
    standard PLI refinement with the ratio>R subdivision fix.

    Parameters
    ----------
    rate_time : callable
        rate_time(t) -> float, the signed rate along the trajectory.
    horizon : float
        Maximum time to search for an event.
    alpha : float
        Scaling factor for the approximate upper bound (>= 1).
    n_segments : int
        Number of linear segments per period for the envelope.
    R : float
        Ratio threshold; if ratio > R, subdivide.
    max_iter : int
        Maximum number of thinning iterations.
    diagnostics : bool
        Whether to return diagnostic statistics.
    """
    rate_fn = lambda t: max(rate_time(t), 0.0)

    # --- Phase 1: Fit sinusoid from 3 gradient evaluations ---
    # r(t) ~ A + B*sin(2t) + C*cos(2t)
    # r(0)    = A + C
    # r(pi/4) = A + B
    # r(pi/2) = A - C
    r0 = rate_fn(0.0)
    r1 = rate_fn(np.pi / 4.0)
    r2 = rate_fn(np.pi / 2.0)
    n_rate_evals = 3

    A_fit = 0.5 * (r0 + r2)
    C_fit = 0.5 * (r0 - r2)
    B_fit = r1 - A_fit

    def sinusoidal_rate(t):
        return A_fit + B_fit * np.sin(2.0 * t) + C_fit * np.cos(2.0 * t)

    # --- Phase 2: Build piecewise-linear envelope ---
    period = np.pi
    envelope_end = min(horizon, period)

    # Build knots for one period from the fitted sinusoid
    knot_times_1p = np.linspace(0.0, envelope_end, n_segments + 1)
    knot_values_1p = np.array([alpha * max(sinusoidal_rate(t), 0.0)
                                for t in knot_times_1p])

    # Concavity fix: for each segment, if the sinusoidal midpoint exceeds
    # the linear chord midpoint, insert an extra knot there.
    refined_times = [knot_times_1p[0]]
    refined_values = [knot_values_1p[0]]
    for i in range(len(knot_times_1p) - 1):
        t0, t1 = knot_times_1p[i], knot_times_1p[i + 1]
        v0, v1 = knot_values_1p[i], knot_values_1p[i + 1]
        t_mid = 0.5 * (t0 + t1)
        v_mid_chord = 0.5 * (v0 + v1)
        v_mid_sin = alpha * max(sinusoidal_rate(t_mid), 0.0)
        if v_mid_sin > v_mid_chord:
            refined_times.append(t_mid)
            refined_values.append(v_mid_sin)
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

    Parameters
    ----------
    rate_time : callable
        rate_time(t) -> float, the signed rate along the trajectory.
    horizon : float
        Maximum time to search for an event.
    alpha : float
        Scaling factor for the approximate upper bound (>= 1).
    n_segments : int
        Number of linear segments per period for the envelope.
    R : float
        Ratio threshold; if ratio > R, subdivide.
    max_iter : int
        Maximum number of thinning iterations.
    second_harmonic : bool
        If True, fit 5-parameter model (2 extra gradient evals).
    diagnostics : bool
        Whether to return diagnostic statistics.
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
    
    
