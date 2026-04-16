import numpy as np

def piecewise_thinning(rate_time, horizon, alpha=1.2, t_init=None, R=2.0, max_iter=200, diagnostics=True):
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
            result = np.inf
            break

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


def piecewise_thinning_midpoint(rate_time, horizon, alpha=1.2, t_init=None,
                                R=2.0, max_iter=200, diagnostics=True):
    if t_init is None:
        t_init = horizon / 100.0

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
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0:
            bound_violations += 1

        if ratio > R:
            result = np.inf
            break

        if h_prop < 1e-14:
            result = np.inf
            break

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
