import numpy as np
import scipy.stats as stats

def fit_gpd(data, tail_fraction=0.1):
    threshold = np.quantile(data, 1 - tail_fraction)
    exceedances = data[data > threshold] - threshold

    if len(exceedances) < 10:
        return threshold, 0, np.mean(exceedances) if len(exceedances) > 0 else 1.0

    mean_excess = np.mean(exceedances)
    var_excess = np.var(exceedances)

    if var_excess < 1e-6:
        return threshold, 0, mean_excess

    shape = 0.5 * (mean_excess**2 / var_excess - 1)
    scale = 0.5 * mean_excess * (mean_excess**2 / var_excess + 1)

    return threshold, shape, max(scale, 1e-6)

def calculate_threshold(S_m, S_nm, p_u):
    u_m, zeta_m, mu_m = fit_gpd(S_m, tail_fraction=0.1)

    S_nm_inv = -S_nm
    u_nm, zeta_nm, mu_nm = fit_gpd(S_nm_inv, tail_fraction=0.1)
    u_nm = -u_nm

    tau_min = max(S_m.min(), S_nm.min())
    tau_max = min(S_m.max(), S_nm.max())
    tau_range = np.linspace(tau_min, tau_max, 1000)

    min_error = float('inf')
    optimal_tau = (tau_min + tau_max) / 2

    for tau in tau_range:
        if tau > u_m:
            p_tail = 1 - stats.genpareto.cdf(tau - u_m, zeta_m, scale=mu_m)
            n_exceed = np.sum(S_m > u_m)
            p_m = (n_exceed / len(S_m)) * p_tail
        else:
            p_m = np.mean(S_m > tau)

        if tau < u_nm:
            p_tail = 1 - stats.genpareto.cdf(-(tau - u_nm), zeta_nm, scale=mu_nm)
            n_below = np.sum(S_nm < u_nm)
            p_nm = (n_below / len(S_nm)) * p_tail
        else:
            p_nm = np.mean(S_nm < tau)

        error_prob = (1 - p_u) * p_m + p_u * p_nm

        if error_prob < min_error:
            min_error = error_prob
            optimal_tau = tau

    return optimal_tau