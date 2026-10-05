import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx


def densite_continue_stable(y, x, t, theta):
    inv_sqrt_2pi_t = 1.0 / np.sqrt(2 * np.pi * t)

    term1 = inv_sqrt_2pi_t * np.exp(-((x - y) ** 2) / (2 * t))

    S = np.abs(x) + np.abs(y)
    exp_mirror = np.exp(-(S ** 2) / (2 * t))
    term2 = inv_sqrt_2pi_t * exp_mirror

    U = theta * np.sqrt(2 * t) + S / np.sqrt(2 * t)
    term3 = theta * exp_mirror * erfcx(U)

    return term1 - term2 + term3


def envelope_gaussienne(y, x, t, M):
    return M * (1 / np.sqrt(2 * np.pi * t)) * np.exp(-((y - x) ** 2) / (2 * t))


# --- Paramètres ---
theta = 1.0
t = 0.5
M_initial = 2.0
M_optimal = 1.0

x_values = [0.0, 1.0, 20.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, x_start in enumerate(x_values):
    ax = axes[i]

    sigma = np.sqrt(t)
    y_grid = np.linspace(x_start - 5 * sigma, x_start + 5 * sigma, 10_000)

    h_vals = densite_continue_stable(y_grid, x_start, t, theta)
    env_vals_2 = envelope_gaussienne(y_grid, x_start, t, M_initial)
    env_vals_1 = envelope_gaussienne(y_grid, x_start, t, M_optimal)

    ax.plot(y_grid, h_vals, color='blue', lw=2.5, label=r'$h(y)$')
    ax.plot(y_grid, env_vals_2, color='red', ls='--', alpha=0.6, label=r'Enveloppe $M=2$')
    ax.plot(y_grid, env_vals_1, color='green', ls=':', lw=2, label=r'Enveloppe $M=1$')

    ax.fill_between(y_grid, h_vals, env_vals_2, color='red', alpha=0.08)

    ax.set_title(rf"$x_0 = {x_start}$", fontweight='bold')
    ax.set_xlabel(r"$y$")
    if i == 0:
        ax.set_ylabel("Densité")

    ax.grid(alpha=0.2)

# Légende unique, propre
axes[1].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False
)

plt.suptitle(
    rf"$t={t}, \theta={theta}$",
    fontsize=14
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()
