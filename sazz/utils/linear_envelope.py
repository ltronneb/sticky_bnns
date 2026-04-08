import numpy as np
# def piecewise_thinning(rate_time, horizon, alpha=1.05, t_init=None, R=2.0, max_iter=200):

#     if t_init is None:
#         t_init = horizon / 5.0

#     rate_fn = lambda t: max(rate_time(t), 0.0)

#     lam_prev = alpha * rate_fn(0.0)
#     lam_init = alpha * rate_fn(t_init)

#     a = (lam_init - lam_prev) / t_init
#     b = lam_prev
#     t_prev = 0.0

#     for _ in range(max_iter):
#         u = np.random.random()
#         xi = -np.log(1.0 - u)

#         if abs(a) < 1e-14:
#             if b < 1e-14:
#                 return np.inf
#             s = xi / b
#         else:
#             disc = b**2 + 2.0 * a * xi
#             if disc < 0.0:
#                 return np.inf
#             s = (-b + np.sqrt(disc)) / a

#         t_prop = t_prev + s

#         if t_prop >= horizon:
#             return np.inf

#         lam_true = rate_fn(t_prop)
#         lam_knot = alpha * lam_true
#         h_prop = b + a * s
        
#         if lam_true/h_prop > R:
#             return np.inf

#         if h_prop < 1e-14:
#             return np.inf

#         if np.random.random() < lam_true / h_prop:
#             return t_prop

#         # Rejection: update segment from (t_prev, lam_prev) to (t_prop, lam_knot)
#         a = (lam_knot - lam_prev) / (t_prop - t_prev)
#         b = lam_knot
#         lam_prev = lam_knot
#         t_prev = t_prop

#     import warnings
#     warnings.warn(
#         f"piecewise_thinning: max_iter={max_iter} reached. "
#         f"Rate may be near zero or oscillating faster than envelope can track. "
#         f"Last t={t_prev:.6f}, horizon={horizon:.6f}. Returning np.inf.",
#         RuntimeWarning,
#     )
#     return np.inf


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
            'proposals': n_proposals,
            'rate_evals': n_rate_evals,
            'max_ratio': max_ratio,
            'bound_violations': bound_violations,
            'tau': result,
        }
        return result, stats
    return result