import numpy as np
from scipy.special import erfcx
import matplotlib.pyplot as plt

def densite_continue_non_normalisee(y, x, t, theta):
    # Terme 1: Gaussienne classique
    term1 = (1 / np.sqrt(2 * np.pi * t)) * np.exp(-((x - y)**2) / (2 * t))
    
    # Terme 2: Gaussienne "miroir" (négative)
    term2 = (1 / np.sqrt(2 * np.pi * t)) * np.exp(-((abs(x) + abs(y))**2) / (2 * t))
    
    # Terme 3: Terme correctif
    arg_erfc = theta * np.sqrt(2 * t) + (abs(x) + abs(y)) / np.sqrt(2 * t)
    term3 = theta * np.exp(2 * theta * (abs(x) + abs(y))) * \
            np.exp(2 * (theta**2) * t) * erfcx(arg_erfc) * np.exp(-1* arg_erfc**2)
            
    return term1 - term2 + term3

def masse_dirac(x, t, theta):
    arg_erfc = theta * np.sqrt(2 * t) + abs(x) / np.sqrt(2 * t)
    proba_0 = np.exp(2 * theta * abs(x)) * \
              np.exp(2 * (theta**2) * t) * erfcx(arg_erfc)* np.exp(-1* arg_erfc**2)
    return np.clip(proba_0, 0, 1)

def simulation_pas_de_temps(x_prev, dt, theta, proposal_sampler_g, proposal_pdf_g, M):
    """
    Retourne (y, nombre_de_rejets)
    """
    
    # 1. Calcul de la probabilité de coller à 0
    p_stick = masse_dirac(x_prev, dt, theta)
    
    # 2. Test de Bernoulli
    u_stick = np.random.uniform(0, 1)
    
    if u_stick <= p_stick:
        return 0.0, 0 # <--- MODIF : On retourne 0 rejet car on n'est pas entré dans la boucle
    else:
        # 3. Si on ne colle pas, REJET
        compteur_rejets = 0 # <--- MODIF : Initialisation du compteur
        
        while True:
            # a. Proposition
            y = proposal_sampler_g() 
            
            # b. Uniforme pour le rejet
            u_reject = np.random.uniform(0, 1)
            
            # c. Calcul des densités
            h_y = densite_continue_non_normalisee(y, x_prev, dt, theta)
            g_y = proposal_pdf_g(y)
            
            # d. Ratio d'acceptation
            denom = M * g_y * (1 - p_stick)
            
            if denom > 0 and u_reject <= h_y / denom:
                return y, compteur_rejets # <--- MODIF : On retourne la valeur et le nombre d'échecs avant succès
            
            # Si on est ici, c'est qu'on a rejeté
            compteur_rejets += 1 # <--- MODIF : Incrémentation

def simuler_trajectoire_sticky(t_max, n_steps, theta, x0=0.0):
    dt = t_max / n_steps
    traj = np.zeros(n_steps + 1)
    traj[0] = x0
    
    # Tableau pour stocker l'historique des rejets
    rejets_history = np.zeros(n_steps) # <--- MODIF
    
    M_constante = 1.0
    
    for i in range(n_steps):
        x_prev = traj[i]
        
        sampler = lambda: np.random.normal(x_prev, np.sqrt(dt))
        dens = lambda y: (1 / np.sqrt(2 * np.pi * dt)) * np.exp(-((y - x_prev)**2) / (2 * dt))
        
        # Récupération des deux valeurs retournées
        valeur, n_rejets = simulation_pas_de_temps( # <-- MODIF
            x_prev, 
            dt, 
            theta, 
            sampler, 
            dens, 
            M_constante
        )
        
        traj[i+1] = valeur
        rejets_history[i] = n_rejets # <--- MODIF : On stocke le nombre de rejets
                    
    return np.linspace(0, t_max, n_steps + 1), traj, rejets_history # <--- MODIF : On retourne l'historique

# ambessa R irl 

T = 3
N = 500
theta_param = 1

# On récupère aussi 'rejets' (à utiliser que si instabilités de calculs réglées)
t_axis, X_t, rejets = simuler_trajectoire_sticky(T, N, theta_param)


total_rejets = np.sum(rejets)
avg_rejets = np.mean(rejets)
print(f"--- Statistiques de Rejet (M=2.0) ---")
print(f"Nombre total de points simulés : {N}")
print(f"Nombre total de rejets : {int(total_rejets)}")

fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))

# --- Trajectoire
ax1.plot(t_axis, X_t, lw=0.8, label="Trajectoire simulée")
ax1.axhline(0, color='red', alpha=0.3, ls='--')
ax1.set_title(f"Sticky BM\nTheta={theta_param}, N={N}")
ax1.set_xlabel("Temps")
ax1.set_ylabel("Position")
ax1.legend()

plt.tight_layout()
plt.savefig('trajectoire_rejets.pdf', bbox_inches='tight')
plt.show()