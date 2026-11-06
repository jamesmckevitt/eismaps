"""Fits of every spectrum of an EIS spectral window at once.

EISPAC fits one spectrum at a time with MPFIT. The functions here fit the same
multi-Gaussian model to all the spectra of a window together with array
operations, which is much faster. They use the template's start values, limits,
ties and fixed parameters, and MPFIT's convergence tests. Errors on the fitted
parameters are computed as MPFIT computes them.
"""

import copy

import numpy as np
import scipy.linalg
from eispac.core.eisfitresult import EISFitResult
from eispac.instr import calc_velocity

# Levenberg-Marquardt damping at the start of a fit, and for a second attempt.
START_DAMPING = 1e-3
RESTART_DAMPING = 1.0

# Lower bound on the damping, so that a rejected step can still raise it.
MIN_DAMPING = 1e-100

# Fraction of its value a peak may fall to in one step of a second attempt.
MIN_PEAK_FRACTION = 0.1

# Peaks below this fraction of their error count as zero.
ZERO_PEAK_FRACTION = 1e-6

# Eigenvalue ratio below which a matrix is treated as singular.
SINGULAR_RATIO = 1e-20

# MPFIT's tolerance for dependent columns in its covariance matrix.
RANK_TOLERANCE = 1e-14


def _model_and_jacobian(params, wave, n_gauss, n_poly):
    """Return the model and its derivatives for many spectra.

    Parameters
    ----------
    params : np.ndarray
        Parameters, shape (n_spectra, n_params): peak, centroid and width of
        each Gaussian, then the background polynomial in increasing order.
    wave : np.ndarray
        Wavelengths, shape (n_spectra, n_wave).
    n_gauss, n_poly : int
        Number of Gaussians and of background terms.

    Returns
    -------
    tuple of np.ndarray
        Model, shape (n_spectra, n_wave), and derivatives, shape
        (n_spectra, n_wave, n_params).
    """
    n_spectra, n_wave = wave.shape
    model = np.zeros((n_spectra, n_wave))
    jacobian = np.empty((n_spectra, n_wave, 3 * n_gauss + n_poly))
    for n in range(n_gauss):
        peak = params[:, 3 * n, None]
        offset = wave - params[:, 3 * n + 1, None]
        width = params[:, 3 * n + 2, None]
        shape = np.exp(-0.5 * (offset / width) ** 2)
        gaussian = peak * shape
        model += gaussian
        jacobian[:, :, 3 * n] = shape
        jacobian[:, :, 3 * n + 1] = gaussian * offset / width ** 2
        jacobian[:, :, 3 * n + 2] = gaussian * offset ** 2 / width ** 3
    for k in range(n_poly):
        power = wave ** k
        model += params[:, 3 * n_gauss + k, None] * power
        jacobian[:, :, 3 * n_gauss + k] = power
    return model, jacobian


def _apply_ties(params, tied):
    """Set tied parameters in place from MPFIT tie expressions such as 'p[5]'."""
    transposed = params.T
    for i, expression in enumerate(tied):
        if expression:
            transposed[i] = eval(expression, {'__builtins__': {}}, {'p': transposed})
    return params


def _tie_derivatives(tied):
    """Return the derivative of each parameter after tying with respect to each before.

    Raises
    ------
    ValueError
        If a tie is not linear, as the fit assumes.
    """
    n_params = len(tied)
    start = np.random.default_rng(1).normal(size=n_params)
    tied_start = _apply_ties(start.copy(), tied)
    derivatives = np.zeros((n_params, n_params))
    for k in range(n_params):
        one, two = start.copy(), start.copy()
        one[k] += 1.0
        two[k] += 2.0
        derivatives[:, k] = _apply_ties(one, tied) - tied_start
        if not np.allclose(_apply_ties(two, tied) - tied_start, 2 * derivatives[:, k]):
            raise ValueError(f'Template tie is not linear: {tied}')
    return derivatives


def _start_values(wave, inten, good, guess, n_gauss, n_poly):
    """Return start parameters for each spectrum, as `eispac.core.scale_guess` gives them.

    The background is scaled to the three lowest points and each peak to the
    data at its centroid, using `good` points only. The background
    coefficients are evaluated with the highest order first, as EISPAC does.
    """
    n_spectra = wave.shape[0]
    params = np.tile(guess, (n_spectra, 1))
    background = np.zeros_like(wave)
    if n_poly > 0:
        data_floor = np.sort(np.where(good, inten, np.inf), axis=1)[:, :3].mean(axis=1)
        coefficients = guess[3 * n_gauss:]
        guess_curve = np.zeros_like(wave)
        for c in coefficients:
            guess_curve = guess_curve * wave + c
        guess_floor = np.median(np.sort(np.where(good, guess_curve, np.inf), axis=1)[:, :3], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = coefficients[None, :] * (data_floor / guess_floor)[:, None]
        params[:, 3 * n_gauss:] = scaled
        for k in range(n_poly):
            background = background * wave + scaled[:, k, None]
    rows = np.arange(n_spectra)
    for n in range(n_gauss):
        nearest = np.argmin(np.where(good, np.abs(wave - guess[3 * n + 1]), np.inf), axis=1)
        params[:, 3 * n] = np.maximum(inten[rows, nearest] - background[rows, nearest], 0.0)
    return params


def _solve(matrices, vectors):
    """Solve many small linear systems, with least squares for any that are singular."""
    try:
        solution = np.linalg.solve(matrices, vectors[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        solution = np.full_like(vectors, np.nan)
    bad = ~np.isfinite(solution).all(axis=1)
    if bad.any():
        solution[bad] = (np.linalg.pinv(matrices[bad]) @ vectors[bad][:, :, None])[:, :, 0]
    return solution


def _rank_test_variances(jacobian):
    """Return parameter variances from one Jacobian, as MPFIT computes them.

    Columns of a pivoted QR factorisation with a diagonal element below
    `RANK_TOLERANCE` of the largest count as dependent and get zero variance.
    """
    variances = np.zeros(jacobian.shape[1])
    r_matrix, pivots = scipy.linalg.qr(jacobian, mode='economic', pivoting=True)[1:]
    diagonal = np.abs(np.diag(r_matrix))
    if diagonal.size == 0 or diagonal[0] == 0:
        return variances
    rank = int((diagonal > RANK_TOLERANCE * diagonal[0]).sum())
    inverse = scipy.linalg.solve_triangular(r_matrix[:rank, :rank], np.eye(rank))
    variances[pivots[:rank]] = (inverse ** 2).sum(axis=1)
    return variances


def _variances(jacobian):
    """Return parameter variances of many spectra from their Jacobians."""
    n_free = jacobian.shape[2]
    hessian = np.einsum('pmj,pmk->pjk', jacobian, jacobian)
    zero = np.einsum('pjj->pj', hessian) == 0
    hessian += zero[:, :, None] * np.eye(n_free)

    # Scaling to a correlation matrix makes the test independent of the units
    # of each parameter, which differ by many orders of magnitude.
    scale = np.sqrt(np.einsum('pjj->pj', hessian))
    with np.errstate(all='ignore'):
        correlation = hessian / (scale[:, :, None] * scale[:, None, :])
    finite = np.isfinite(correlation).all(axis=(1, 2))
    eigenvalues = np.zeros((hessian.shape[0], n_free))
    eigenvectors = np.zeros_like(hessian)
    eigenvalues[finite], eigenvectors[finite] = np.linalg.eigh(correlation[finite])
    singular = ~(eigenvalues[:, 0] > SINGULAR_RATIO * eigenvalues[:, -1])
    safe = np.where(singular[:, None], 1.0, eigenvalues)
    with np.errstate(all='ignore'):
        variances = np.einsum('pjk,pk->pj', eigenvectors ** 2, 1.0 / safe) / scale ** 2 * ~zero
    singular |= ~np.isfinite(variances).all(axis=1)
    for i in np.nonzero(singular)[0]:
        variances[i] = _rank_test_variances(jacobian[i])
    return variances


class _Problem:
    """Spectra and template settings shared by the steps of a fit."""

    def __init__(self, wave, inten, errs, template, parinfo, min_points):
        self.n_gauss = int(template['n_gauss'])
        self.n_poly = int(template['n_poly'])
        self.n_params = 3 * self.n_gauss + self.n_poly
        self.min_points = max(min_points, self.n_params)
        self.guess = np.asarray(template['fit'], dtype=float)
        self.good = (errs > 0) & (wave >= template['wmin']) & (wave <= template['wmax']) & np.isfinite(inten)
        # Unused points have zero weight, and NaN would survive that.
        self.wave = np.where(self.good, wave, 0.0)
        self.inten = np.where(self.good, inten, 0.0)
        self.n_good = self.good.sum(axis=1)
        self.weight = self.good / np.where(self.good, errs, 1.0)

        self.tied = [str(p.get('tied', '') or '').strip() for p in parinfo]
        fixed = np.array([bool(p.get('fixed', 0)) for p in parinfo])
        self.free = np.nonzero(~fixed & ~np.array([bool(t) for t in self.tied]))[0]
        self.free_derivatives = _tie_derivatives(self.tied)[:, self.free]

        limits = np.array([p.get('limits', [0.0, 0.0]) for p in parinfo], dtype=float)[self.free]
        limited = np.array([p.get('limited', [0, 0]) for p in parinfo], dtype=bool)[self.free]
        self.has_lower, self.lower = limited[:, 0], limits[:, 0]
        self.has_upper, self.upper = limited[:, 1], limits[:, 1]

        is_peak = np.array([k < 3 * self.n_gauss and k % 3 == 0 for k in self.free])
        self.zero_limit_peaks = is_peak & self.has_lower & (self.lower == 0)

    def full_params(self, base, rows, free_values):
        """Return all parameters of `rows` with their free parameters set."""
        params = base[rows].copy()
        params[:, self.free] = free_values
        return _apply_ties(params, self.tied)

    def residuals(self, base, rows, free_values, jacobian=False):
        """Return weighted residuals and, if asked, their derivatives for the free parameters."""
        params = self.full_params(base, rows, free_values)
        model, model_jacobian = _model_and_jacobian(params, self.wave[rows], self.n_gauss, self.n_poly)
        weight = self.weight[rows]
        residuals = weight * (self.inten[rows] - model)
        if not jacobian:
            return residuals, None
        return residuals, -(weight[:, :, None] * (model_jacobian @ self.free_derivatives))

    def drop_pegged(self, jacobian, residuals, free_values):
        """Zero the derivatives of parameters on a limit that the fit would push further out."""
        pull = np.einsum('pm,pmj->pj', residuals, jacobian)
        pegged = ((self.has_lower & (free_values == self.lower) & (pull > 0))
                  | (self.has_upper & (free_values == self.upper) & (pull < 0)))
        return np.where(pegged[:, None, :], 0.0, jacobian)


def _damped_step(problem, hessian, gradient, damping, scale, values, stops_at_lower):
    """Return the Levenberg-Marquardt step, holding parameters a limit would stop."""
    n_free = gradient.shape[1]
    identity = np.eye(n_free)
    matrices = hessian + damping[:, None, None] * (np.where(scale > 0, scale, 1.0) ** 2)[:, :, None] * identity
    step = -_solve(matrices, gradient)
    held = np.zeros_like(step, dtype=bool)
    for _ in range(n_free):
        outward = ((stops_at_lower & (values == problem.lower) & (step < 0))
                   | (problem.has_upper & (values == problem.upper) & (step > 0))) & ~held
        if not outward.any():
            break
        held |= outward
        kept = ~held
        step = -_solve(matrices * (kept[:, :, None] & kept[:, None, :]) + held[:, :, None] * identity,
                       np.where(held, 0.0, gradient))
    return step


def _limit_step(problem, values, step, stops_at_lower, kept_off_zero):
    """Return the new parameters, stopping the whole step at the first limit reached."""
    with np.errstate(all='ignore'):
        to_lower = np.where(stops_at_lower & (step < 0), (problem.lower - values) / step, np.inf)
        to_upper = np.where(problem.has_upper & (step > 0), (problem.upper - values) / step, np.inf)
    fraction = np.clip(np.minimum(to_lower, to_upper).min(axis=1), 0.0, 1.0)
    new = values + fraction[:, None] * step
    short = (fraction < 1)[:, None]
    new = np.where(short & stops_at_lower & (step < 0) & (to_lower <= fraction[:, None]), problem.lower, new)
    new = np.where(short & problem.has_upper & (step > 0) & (to_upper <= fraction[:, None]), problem.upper, new)
    new = np.where(kept_off_zero, np.maximum(new, MIN_PEAK_FRACTION * values), new)
    new = np.where(problem.has_lower, np.maximum(new, problem.lower), new)
    return np.where(problem.has_upper, np.minimum(new, problem.upper), new)


def _iterate(problem, base, rows, values, damping0, maxiter, tolerance, keep_peaks_off_zero=False):
    """Run Levenberg-Marquardt on `rows` until each spectrum converges.

    Parameters
    ----------
    problem : _Problem
        Spectra and template settings.
    base : np.ndarray
        Start parameters of all spectra, shape (n_spectra, n_params).
    rows : np.ndarray
        Spectra to fit.
    values : np.ndarray
        Start values of their free parameters.
    damping0 : float
        Starting damping.
    maxiter : int
        Maximum number of Jacobians per spectrum.
    tolerance : float
        MPFIT's `ftol`, `xtol` and `gtol`.
    keep_peaks_off_zero : bool, default=False
        Let peaks approach zero without reaching it. A peak of exactly zero
        has zero derivatives for its centroid and width, so a fit cannot
        leave it.

    Returns
    -------
    tuple of np.ndarray
        Free parameters and status of `rows`. Status follows MPFIT: 1 for
        chi-squared, 2 for parameters, 3 for both, 4 for the gradient, 5 for
        reaching `maxiter`, and 0 if the fit failed.
    """
    n_free = problem.free.size
    kept_off_zero = problem.zero_limit_peaks if keep_peaks_off_zero else np.zeros(n_free, dtype=bool)
    stops_at_lower = problem.has_lower & ~kept_off_zero
    result = values.copy()
    status = np.full(rows.size, -99)
    iterations = np.zeros(rows.size, dtype=int)
    active = np.arange(rows.size)
    residuals = problem.residuals(base, rows, result)[0]
    chi2 = (residuals * residuals).sum(axis=1)
    damping = np.full(rows.size, damping0)
    growth = np.full(rows.size, 2.0)
    scale = np.zeros((rows.size, n_free))
    gradient = np.zeros((rows.size, n_free))
    hessian = np.zeros((rows.size, n_free, n_free))
    needs_jacobian = np.ones(rows.size, dtype=bool)

    while active.size:
        current = result[active]
        done = np.zeros(active.size, dtype=bool)

        if needs_jacobian.any():
            new = np.nonzero(needs_jacobian)[0]
            residuals, jacobian = problem.residuals(base, rows[active[new]], current[new], jacobian=True)
            jacobian = problem.drop_pegged(jacobian, residuals, current[new])
            gradient[new] = np.einsum('pmj,pm->pj', jacobian, residuals)
            hessian[new] = np.einsum('pmj,pmk->pjk', jacobian, jacobian)
            column_norm = np.sqrt(np.einsum('pmj,pmj->pj', jacobian, jacobian))
            scale[new] = np.maximum(scale[new], column_norm)
            iterations[active[new]] += 1

            # gtol: largest cosine between the residuals and a column of the Jacobian.
            with np.errstate(all='ignore'):
                cosine = np.where(column_norm > 0, np.abs(gradient[new]) / (np.sqrt(chi2[new])[:, None] * column_norm), 0.0)
            converged = (cosine.max(axis=1) <= tolerance) | (chi2[new] == 0)
            done[new[converged]] = True
            status[active[done]] = 4

        step = _damped_step(problem, hessian, gradient, damping, scale, current, stops_at_lower)
        trial = _limit_step(problem, current, step, stops_at_lower, kept_off_zero)
        change = trial - current

        new_chi2 = (problem.residuals(base, rows[active], trial)[0] ** 2).sum(axis=1)
        predicted = -(2 * np.einsum('pj,pj->p', change, gradient) + np.einsum('pj,pjk,pk->p', change, hessian, change))
        actual = chi2 - new_chi2
        with np.errstate(all='ignore'):
            ratio = np.where(predicted > 0, actual / predicted, -1.0)
            relative_actual = np.abs(actual) / chi2
            relative_predicted = predicted / chi2
        accept = (ratio > 1e-4) & np.isfinite(new_chi2) & ~done

        ftol_met = (relative_actual <= tolerance) & (relative_predicted <= tolerance) & (0.5 * ratio <= 1)
        xtol_met = (np.sqrt(((scale * change) ** 2).sum(axis=1))
                    <= tolerance * np.sqrt(((scale * trial) ** 2).sum(axis=1)))

        result[active[accept]] = trial[accept]
        chi2 = np.where(accept, new_chi2, chi2)
        damping = np.where(accept, damping * np.maximum(1 / 3, 1 - (2 * ratio - 1) ** 3), damping * growth)
        damping = np.maximum(damping, MIN_DAMPING)
        growth = np.where(accept, 2.0, growth * 2)
        needs_jacobian = accept.copy()

        code = np.where(ftol_met & xtol_met, 3, np.where(xtol_met, 2, np.where(ftol_met, 1, 0)))
        newly_done = (code > 0) & ~done
        status[active[newly_done]] = code[newly_done]
        out_of_iterations = ~done & ~newly_done & (iterations[active] >= maxiter) & accept
        status[active[out_of_iterations]] = 5
        stop = done | newly_done | out_of_iterations | ~np.isfinite(damping)
        status[active[stop & (status[active] == -99)]] = 0

        keep = ~stop
        active = active[keep]
        chi2, damping, growth = chi2[keep], damping[keep], growth[keep]
        scale, gradient, hessian = scale[keep], gradient[keep], hessian[keep]
        needs_jacobian = needs_jacobian[keep]

    return result, status


def _fit_block(wave, inten, errs, template, parinfo, min_points, maxiter, tolerance):
    """Fit one block of spectra with arrays of shape (n_spectra, n_wave)."""
    problem = _Problem(wave, inten, errs, template, parinfo, min_points)
    n_spectra = wave.shape[0]
    base = _start_values(problem.wave, problem.inten, problem.good, problem.guess, problem.n_gauss, problem.n_poly)
    free_start = base[:, problem.free]

    enough = problem.n_good >= problem.min_points
    outside = (((problem.has_lower & (free_start < problem.lower))
                | (problem.has_upper & (free_start > problem.upper))).any(axis=1)
               | ~np.isfinite(free_start).all(axis=1))
    status = np.where(enough, 0, -1)
    rows = np.nonzero(enough & ~outside)[0]

    params = np.zeros((n_spectra, problem.n_params))
    perror = np.zeros((n_spectra, problem.n_params))
    chi2 = np.zeros(n_spectra)
    if rows.size == 0:
        return params, perror, chi2, status

    values, row_status = _iterate(problem, base, rows, free_start[rows], START_DAMPING, maxiter, tolerance)
    row_chi2 = (problem.residuals(base, rows, values)[0] ** 2).sum(axis=1)

    # A fit can reach a peak of zero early and stop there although a line is
    # present. These are repeated from a gentler start and the better fit kept.
    peaks = problem.full_params(base, rows, values)[:, 0:3 * problem.n_gauss:3]
    empty = np.nonzero((peaks <= 0).any(axis=1))[0]
    if empty.size:
        second, second_status = _iterate(problem, base, rows[empty], free_start[rows[empty]],
                                         RESTART_DAMPING, maxiter, tolerance, keep_peaks_off_zero=True)
        second_chi2 = (problem.residuals(base, rows[empty], second)[0] ** 2).sum(axis=1)
        better = (second_status > 0) & (second_chi2 < row_chi2[empty])
        values[empty[better]] = second[better]
        row_status[empty[better]] = second_status[better]

    converged = row_status > 0
    status[rows] = row_status
    rows, values = rows[converged], values[converged]
    if rows.size == 0:
        return params, perror, chi2, status

    residuals, jacobian = problem.residuals(base, rows, values, jacobian=True)
    jacobian = problem.drop_pegged(jacobian, residuals, values)
    errors = np.zeros((rows.size, problem.n_params))
    errors[:, problem.free] = np.sqrt(np.maximum(_variances(jacobian), 0.0))
    fitted = problem.full_params(base, rows, values)

    # A peak that has fallen far below its error is a fit with no line in it.
    for n in range(problem.n_gauss):
        no_line = fitted[:, 3 * n] < ZERO_PEAK_FRACTION * errors[:, 3 * n]
        fitted[no_line, 3 * n] = 0.0
        errors[no_line, 3 * n] = 0.0

    params[rows] = fitted
    perror[rows] = errors
    with np.errstate(all='ignore'):
        chi2[rows] = (residuals * residuals).sum(axis=1) / (problem.n_good[rows] - problem.free.size)
    return params, perror, chi2, status


def fit_spectra(wave, inten, errs, template, parinfo, min_points=7, maxiter=2000,
                tolerance=1e-8, block=50000):
    """Fit many spectra with one template.

    Parameters
    ----------
    wave, inten, errs : np.ndarray
        Wavelength, intensity and error of each spectrum, shape
        (n_spectra, n_wave). Points with `errs` <= 0 are not used.
    template : dict
        EISPAC template, with `fit`, `n_gauss`, `n_poly`, `wmin` and `wmax`.
    parinfo : list[dict]
        MPFIT parameter settings from the template.
    min_points : int, default=7
        Fewest good points to fit a spectrum, and at least the number of
        parameters.
    maxiter : int, default=2000
        Maximum number of iterations per spectrum.
    tolerance : float, default=1e-8
        MPFIT's `ftol`, `xtol` and `gtol`. EISPAC uses 1e-6. The fits here
        stop slightly earlier than MPFIT at the same value.
    block : int, default=50000
        Spectra fitted together, which limits the memory used.

    Returns
    -------
    dict
        `params` and `perror`, shape (n_spectra, n_params), and `chi2` and
        `status`, shape (n_spectra,). Spectra with too few points have status
        -1, and failed fits status 0, with zero parameters.
    """
    wave = np.asarray(wave, dtype=float)
    inten = np.asarray(inten, dtype=float)
    errs = np.asarray(errs, dtype=float)
    parts = [_fit_block(wave[i:i + block], inten[i:i + block], errs[i:i + block], template, parinfo,
                        min_points, maxiter, tolerance)
             for i in range(0, wave.shape[0], block)]
    params, perror, chi2, status = (np.concatenate(p) for p in zip(*parts))
    return {'params': params, 'perror': perror, 'chi2': chi2, 'status': status}


def cube_arrays(cube):
    """Return wavelength, intensity and error arrays of an EISCube, with bad points zeroed."""
    wave = np.array(cube.wavelength, dtype=float)
    inten = np.array(cube.data, dtype=float)
    errs = np.array(cube.uncertainty.array, dtype=float)
    bad = np.asarray(cube.mask, dtype=bool)
    inten[bad] = 0.0
    errs[bad] = 0.0
    return wave, inten, errs


def fit_cube(cube, template, parinfo, component=0, **kwargs):
    """Fit a window, returning an EISPAC fit result.

    Parameters
    ----------
    cube : eispac.EISCube
        Spectral window.
    template : dict
        EISPAC template.
    parinfo : list[dict]
        MPFIT parameter settings from the template.
    component : int, default=0
        Gaussian stored as the main component of the result.
    **kwargs
        Passed to :func:`fit_spectra`.

    Returns
    -------
    eispac.EISFitResult
        Fit result usable with the functions in :mod:`eismaps.maps`.
    """
    wave, inten, errs = cube_arrays(cube)
    n_pxls, n_steps, n_wave = inten.shape
    fitted = fit_spectra(wave.reshape(-1, n_wave), inten.reshape(-1, n_wave), errs.reshape(-1, n_wave),
                         template, parinfo, **kwargs)

    result = EISFitResult(wave, template, parinfo, func_name='multigaussian',
                          data_units=cube.unit.to_string(), radcal=copy.deepcopy(cube.radcal))
    result.meta = copy.deepcopy(cube.meta)
    result.fit_module = 'eismaps.batched_fit'
    result.fit_method = 'LevMarLSQ'

    n_gauss = int(template['n_gauss'])
    fit = result.fit
    fit['main_component'] = component
    fit['status'] = fitted['status'].reshape(n_pxls, n_steps)
    fit['chi2'] = fitted['chi2'].reshape(n_pxls, n_steps)
    fit['wavelength'] = wave
    fit['params'] = fitted['params'].reshape(n_pxls, n_steps, -1)
    fit['perror'] = fitted['perror'].reshape(n_pxls, n_steps, -1)
    fit['mask'] = ~((errs > 0) & (wave >= template['wmin']) & (wave <= template['wmax']))
    for n in range(n_gauss):
        peak, width = fit['params'][:, :, 3 * n], fit['params'][:, :, 3 * n + 2]
        peak_err, width_err = fit['perror'][:, :, 3 * n], fit['perror'][:, :, 3 * n + 2]
        centroid, centroid_err = fit['params'][:, :, 3 * n + 1], fit['perror'][:, :, 3 * n + 1]
        fit['int'][:, :, n] = np.sqrt(2 * np.pi) * peak * width
        fit['width'][:, :, n] = width
        fit['err_width'][:, :, n] = width_err
        with np.errstate(all='ignore'):
            relative = np.sqrt((peak_err / peak) ** 2 + (width_err / width) ** 2)
        fit['err_int'][:, :, n] = np.where(peak > 0, fit['int'][:, :, n] * relative, 0.0)
        rest = float(parinfo[3 * n + 1]['value'])
        fit['vel'][:, :, n] = calc_velocity(centroid, rest)
        with np.errstate(divide='ignore', invalid='ignore'):
            fit['err_vel'][:, :, n] = np.where(centroid > 0, centroid_err / centroid, 0.0) * fit['vel'][:, :, n]
    return result
