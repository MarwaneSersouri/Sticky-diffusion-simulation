import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx
from scipy.stats import norm

# Paramètres
t = 1.0
x_start = 0.0
thetas = [0.5, 1.0, 5.0]
colors = ['blue', 'orange', 'green']
y_grid = np.linspace(-3, 3, 1000)

# Fonction pour la CDF (Fonction de répartition)
def cdf_sticky(y, t, theta):
    # Formule théorique de la CDF pour X_0 = 0
    res = np.zeros_like(y)
    for i, val in enumerate(y):
        if val < 0:
            res[i] = norm.cdf(val/np.sqrt(t)) - np.exp(2*theta*np.abs(val) + 2*theta**2*t) * norm.cdf(-np.abs(val)/np.sqrt(t) - 2*theta*np.sqrt(t))
        else:
            res[i] = norm.cdf(val/np.sqrt(t)) + np.exp(2*theta*np.abs(val) + 2*theta**2*t) * norm.cdf(-np.abs(val)/np.sqrt(t) - 2*theta*np.sqrt(t))
    return res

# --- Tracé de la CDF ---
fig, ax = plt.subplots(figsize=(8, 6))

for th, col in zip(thetas, colors):
    cdf_vals = cdf_sticky(y_grid, t, th)
    ax.plot(y_grid, cdf_vals, color=col, lw=2, label=rf'$\theta={th}$')

ax.set_title('Fonction de répartition (CDF)')
ax.set_xlabel('Position $y$')
ax.set_ylabel(r'Probabilité $P(X_t \leq y)$')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cdf_theta.pdf', bbox_inches='tight')
plt.show()