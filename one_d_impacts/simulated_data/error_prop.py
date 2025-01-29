import numpy as np

def propagate_fit_errors(x, popt, perr):
    a, b, c, d = popt
    sigma_a, sigma_b, sigma_c, sigma_d = perr

    # Compute partial derivatives
    exp_term = np.exp(-((x - b) ** 2) / (2 * c ** 2))
    df_da = -exp_term
    df_db = a * exp_term * ((x - b) / (c ** 2))
    df_dc = a * exp_term * ((x - b) ** 2) / (c ** 3)
    df_dd = 1  # d contributes directly

    # Propagated error formula
    sigma_y = np.sqrt((df_da * sigma_a) ** 2 +
                      (df_db * sigma_b) ** 2 +
                      (df_dc * sigma_c) ** 2 +
                      (df_dd * sigma_d) ** 2)

    return sigma_y