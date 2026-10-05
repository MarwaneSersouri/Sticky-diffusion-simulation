import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx

#  1. fonctions de base 

def densite_continue_stable(y, x, t, theta):
    """Calcule h(y) pour la densité théorique."""
    inv_sqrt_2pi_t = 1.0 / np.sqrt(2 * np.pi * t)
    term1 = inv_sqrt_2pi_t * np.exp(-((x - y) ** 2) / (2 * t))
    S = np.abs(x) + np.abs(y)
    exp_mirror = np.exp(-(S ** 2) / (2 * t))
    term2 = inv_sqrt_2pi_t * exp_mirror
    U = theta * np.sqrt(2 * t) + S / np.sqrt(2 * t)
    term3 = theta * exp_mirror * erfcx(U)
    return term1 - term2 + term3

def masse_dirac_stable(x, t, theta):
    """Calcule P(Stick)."""
    U = theta * np.sqrt(2 * t) + np.abs(x) / np.sqrt(2 * t)
    p = np.exp(-(x**2) / (2 * t)) * erfcx(U)
    return np.clip(p, 0.0, 1.0)

def un_pas_de_simulation(x_prev, dt, theta):
    """Fait avancer une particule de dt."""
    # 1. Dirac
    p_stick = masse_dirac_stable(x_prev, dt, theta)
    if np.random.rand() < p_stick:
        return 0.0
    
    # 2. Rejet (M=1)
    sqrt_dt = np.sqrt(dt)
    inv_sqrt_2pi_dt = 1.0 / np.sqrt(2 * np.pi * dt)
    
    while True:
        y = np.random.normal(x_prev, sqrt_dt)
        h_y = densite_continue_stable(y, x_prev, dt, theta)
        g_y = inv_sqrt_2pi_dt * np.exp(-((y - x_prev)**2) / (2 * dt))
        denom = (1.0 - p_stick) * g_y
        
        if denom > 0 and np.random.rand() * denom <= h_y:
            return y

# "Composition" 

def experience_composition(T, n_steps, theta, n_simulations=5000):
    print(f"Lancement de l'expérience : T={T}, Steps={n_steps}, Theta={theta}")
    
    # CAS A : grand saut (1 seul pas de taille T)
    # On simule n_simulations particules d'un coup
    resultats_grand = np.zeros(n_simulations)
    for i in range(n_simulations):
        resultats_grand[i] = un_pas_de_simulation(0.0, T, theta) # Départ de 0
        
    # CAS B : Les  (Composition de n étapes de taille T/n)
    dt = T / n_steps
    resultats_petits_pas = np.zeros(n_simulations)

    
    for i in range(n_simulations):
        x = 0.0 # Départ de 0
        for _ in range(n_steps):
            x = un_pas_de_simulation(x, dt, theta)
        resultats_petits_pas[i] = x
        
    return resultats_grand, resultats_petits_pas

# Paramètres
T_final = 1.0
N_steps_composition = 100 # On découpe T en 50 morceaux
Theta_test = 1.0
Nb_particules = 5000

# Exécution
res_A, res_B = experience_composition(T_final, N_steps_composition, Theta_test, Nb_particules)

# ... (Garder tout le code de simulation précédent) ...

# 3. Analyse Visuelle (Séparation Dirac / Continu)

# On sépare les trajectoires qui ont fini en 0 de celles qui sont dans la diffusion
non_zeros_A = res_A[np.abs(res_A) > 1e-9] # Partie continue A
non_zeros_B = res_B[np.abs(res_B) > 1e-9] # Partie continue B

prop_0_A = 1.0 - len(non_zeros_A) / len(res_A)
prop_0_B = 1.0 - len(non_zeros_B) / len(res_B)

# Calcul théorique de la proba de coller
p_stick_theorique = masse_dirac_stable(0, T_final, Theta_test)

# B. Tracé
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})

# --- GRAPHE 1 : La Densité Conditionnelle (Sachant qu'on n'est pas en 0) ---
# Note importante : Pour comparer avec la théorie, il faut renormaliser h(y)
# Densité conditionnelle = h(y) / (1 - P_stick)

# Histogrammes (sur les données filtrées uniquement)
ax1.hist(non_zeros_A, bins=60, density=True, alpha=0.4, color='blue', label=f"1 Saut (Continu)")
ax1.hist(non_zeros_B, bins=60, density=True, alpha=0.4, color='orange', label=f"{N_steps_composition} Pas (Continu)")


y_grid = np.linspace(min(non_zeros_A.min(), non_zeros_B.min()), max(non_zeros_A.max(), non_zeros_B.max()), 1000)

y_grid = y_grid[np.abs(y_grid) > 1e-4] 

h_vals = densite_continue_stable(y_grid, 0.0, T_final, Theta_test)

densite_conditionnelle = h_vals / (1 - p_stick_theorique)

ax1.plot(y_grid, densite_conditionnelle, 'k--', lw=2, label="Théorie Renormalisée $h(y)/(1-p)$")

ax1.set_title("Comparaison de la PARTIE CONTINUE ($X_T \\neq 0$)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Position")
ax1.set_ylabel("Densité de probabilité")
ax1.legend()
ax1.grid(alpha=0.3)


# GRAPHE 2 : La Masse de Dirac (Comparaison des Zéros)
labels = ['1 Saut', '50 Pas', 'Théorie']
values = [prop_0_A, prop_0_B, p_stick_theorique]
colors = ['blue', 'orange', 'black']

ax2.bar(labels, values, color=colors, alpha=0.6)
ax2.set_title("Probabilité d'absorption en 0 ($P(X_T=0)$)", fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(values)*1.2) # Un peu de marge
ax2.grid(axis='y', alpha=0.3)

# Affichage des valeurs sur les barres
for i, v in enumerate(values):
    ax2.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('validation_densites.pdf', bbox_inches='tight')
plt.show()