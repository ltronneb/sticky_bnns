import numpy as np

def brent(f, a, b, tol=1e-5, rel_tol=1e-5,maxiter = 100):
    """
    Given a function f, and an interval [a,b], find arg min f(x) for x \in [a,b]
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
    return x


def sample_trajectory_at_regular_intervals(Position, Velocity, EventTimes, dt):
    """
    Given event-based trajectory (Position, Velocity, EventTimes),
    return the state at regular time grid with spacing dt.

    Parameters:
    - Position: array of shape (N, D) of positions at each event
    - Velocity: array of shape (N, D) of velocities at each event
    - EventTimes: array of shape (N,) of event times (monotonic increasing)
    - dt: float, desired time grid spacing

    Returns:
    - t_grid: array of shape (M,) of regular times from 0 up to T_max
    - pos_grid: array of shape (M, D) positions at each time in t_grid
    - vel_grid: array of shape (M, D) velocities at each time in t_grid
    """
    # Create regular time grid
    T_max = EventTimes[EventTimes > 0].max()
    t_grid = np.arange(0, T_max + dt, dt)

    # For each regular time, find the index of the last event time <= t
    idx = np.searchsorted(EventTimes, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(EventTimes) - 2)

    # Compute time offset within each segment
    tau = (t_grid - EventTimes[idx])[:, None]  # shape (M,1)

    # Linear interpolation for position; velocity is piecewise constant
    pos_grid = Position[idx] + Velocity[idx] * tau
    vel_grid = Velocity[idx]

    return t_grid, pos_grid, vel_grid