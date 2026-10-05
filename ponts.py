import numpy as np
from scipy.special import erfcx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def q(t: float, x: float, theta: float) -> float:
    """
    Masse atomique en 0 : P_x(X_t = 0).
    """
    if t <= 0:
        return 1.0 if x == 0.0 else 0.0
    U = theta * np.sqrt(2 * t) + abs(x) / np.sqrt(2 * t)

    return float(np.clip(np.exp(-(x**2) / (2 * t)) * erfcx(U), 0.0, 1.0))

def p(t: float, x: float, y: float, theta: float) -> float:
    """
    Densité continue de transition x → y (y ≠ 0).
    """
    if t <= 0:
        return 0.0
    inv = 1.0 / np.sqrt(2 * np.pi * t)
    S   = abs(x) + abs(y)
    U   = theta * np.sqrt(2 * t) + S / np.sqrt(2 * t)

    term1 = inv * np.exp(-((x - y)**2) / (2 * t))          
    em    = np.exp(-(S**2) / (2 * t))                       
    term2 = inv * em                                         
    term3 = theta * em * erfcx(U)                           
    return float(term1 - term2 + term3)

# DENSITÉS DU PONT BROWNIEN COLLANT

def _bridge_normalisation(x: float, T: float, b: float, theta: float) -> float:
    """Constante de normalisation du pont : p(T,x,b) ou q(T,x)."""
    if abs(b) > 1e-10:
        return p(T, x, b, theta)
    else:
        return q(T, x, theta)

def bridge_atom(t: float, x: float, T: float, b: float, theta: float) -> float:
    """
    Masse atomique en 0 de la loi du pont P(X_t = 0 | X_T = b).
    Correspond exactement au terme devant δ_0 des formules (1) et (2).
    """
    norm = _bridge_normalisation(x, T, b, theta)
    if norm <= 0:
        return 0.0
    qt   = q(t, x, theta)
    if abs(b) > 1e-10:                      # formule (1)
        return qt * p(T - t, 0.0, b, theta) / norm
    else:                                   # formule (2)
        return qt * q(T - t, 0.0, theta)   / norm

def bridge_density(t: float, x: float, z: float,
                   T: float, b: float, theta: float) -> float:
    """
    Densité continue en z de la loi du pont P(X_t ∈ dz | X_T = b).
    (Le 'if abs(z) < 1e-10: return 0.0' a été supprimé pour respecter 
    la continuité de la composante dz dans tes formules).
    """
    norm = _bridge_normalisation(x, T, b, theta)
    if norm <= 0:
        return 0.0
    if abs(b) > 1e-10:                      # formule (1)
        return p(t, x, z, theta) * p(T - t, z, b, theta) / norm
    else:                                   # formule (2)
        return p(t, x, z, theta) * q(T - t, z, theta)    / norm

#girsanov

def b_drift(x: float) -> float:
    return np.sin(x)

def A_primitive(x: float) -> float:
    return np.cos(x)

def phi(x: float) -> float:
    return 0.5 * (np.sin(x)**2 + np.cos(x))

K_CONST = -0.5     
M_PPP   = 1.125    

def psi_terminal(y: float, x0: float, T: float) -> float:
    return np.exp(A_primitive(y) - A_primitive(x0) - K_CONST * T)

def simuler_ppp(T: float, M: float):
    N = np.random.poisson(lam=M * T)
    if N == 0:
        return np.array([]), np.array([])
    t_pts = np.sort(np.random.uniform(0, T, size=N))
    y_pts = np.random.uniform(0, M, size=N)
    return t_pts, y_pts

# ÉCHANTILLONNAGE AVEC LES BORNES EXACTES

def sample_bridge_position(t: float, x: float, T: float,
                           b: float, theta: float,
                           rng: np.random.Generator) -> float:
    if t <= 0: return x
    if t >= T: return b

    atom = bridge_atom(t, x, T, b, theta)
    if rng.random() < atom:
        return 0.0

    mu     = x + (t / T) * (b - x)
    sigma2 = t * (T - t) / T
    sigma  = np.sqrt(sigma2)
    if sigma < 1e-12: return mu

    M_exact = 1.0 + theta * np.sqrt(2 * np.pi * sigma2)

    # On passe à 5000 essais pour ne pas déclencher le fallback "ligne droite" !
    for _ in range(5000):
        z = rng.normal(mu, sigma)
        f_z = bridge_density(t, x, z, T, b, theta)
        g_z = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((z - mu)**2) / (2 * sigma2))

        if g_z <= 0: continue
        
        if rng.random() < f_z / (M_exact * g_z):
            return z

    return mu

def sample_XT_sbm(x0: float, T: float, theta: float,
                  rng: np.random.Generator) -> float:
    # Masse atomique en 0
    q_val = q(T, x0, theta)
    if rng.random() < q_val:
        return 0.0

    sigma = np.sqrt(T)
    
    # CORRECTION 2 : Borne exacte pour la loi marginale du SBM
    M_sbm = 1.0 + theta * np.sqrt(2 * np.pi * T)

    for _ in range(1000):
        y = rng.normal(x0, sigma)
        # (Suppression du if abs(y) < 1e-10)
        
        f = p(T, x0, y, theta)
        g = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * \
            np.exp(-((y - x0)**2) / (2 * T))
            
        if g <= 0: continue
        
        # Rejet avec la borne exacte
        if rng.random() < f / (M_sbm * g):
            return y
            
    return x0

def trajectory(x0: float, T: float, theta: float,
                               n_grid: int = 300,
                               seed: int = None) -> dict:
    rng = np.random.default_rng(seed)
    n_rejections = 0
    grid_times = np.linspace(0, T, n_grid + 1)

    while True:
        # Étape 1 : X_T
        xT = None
        psi_max = np.exp(1.0 - np.cos(x0) + T / 2.0) 
        
        for _ in range(500):
            xT_cand = sample_XT_sbm(x0, T, theta, rng)
            w = psi_terminal(xT_cand, x0, T)
            if rng.random() < w / psi_max:
                xT = xT_cand
                break
        if xT is None: xT = sample_XT_sbm(x0, T, theta, rng)

        # Étape 2 : PPP
        t_ppp, y_ppp = simuler_ppp(T, M_PPP)

        # ── LA MAGIE SÉQUENTIELLE ──
        # On regroupe les points PPP et la grille de visualisation
        events = [(t, 'grid', 0.0) for t in grid_times]
        events.extend([(t_ppp[i], 'ppp', y_ppp[i]) for i in range(len(t_ppp))])
        events.sort(key=lambda e: e[0]) # Tri par ordre chronologique

        accept = True
        x_prev = x0
        t_prev = 0.0
        grid_values = []

        # On avance dans le temps, point par point, en partant toujours de x_prev
        for t_curr, ev_type, y_val in events:
            if t_curr == 0.0:
                if ev_type == 'grid': grid_values.append(x0)
                continue

            dt = t_curr - t_prev
            t_remain = T - t_prev

            # Tirage conditionnel partant de la position précédente
            if dt > 1e-12:
                x_curr = sample_bridge_position(dt, x_prev, t_remain, xT, theta, rng)
            else:
                x_curr = x_prev

            # Si c'est un point PPP, on juge la trajectoire
            if ev_type == 'ppp':
                if y_val < phi(x_curr) - K_CONST:
                    accept = False
                    break # On rejette tout et on recommence
            # Si c'est un point de grille, on le garde pour le dessin
            elif ev_type == 'grid':
                grid_values.append(x_curr)

            # Le point actuel devient le point précédent pour la prochaine étape
            x_prev = x_curr
            t_prev = t_curr

        if accept:
            break
        n_rejections += 1

    return {
        "times":        grid_times,
        "values":       np.array(grid_values),
        "xT":           xT,
        "n_rejections": n_rejections,
        "ppp_times":    t_ppp,
        "ppp_heights":  y_ppp,
    }

# VISUALISATION

def plot_trajectories(results: list, T: float, theta: float, x0: float):
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[2.5, 1],
                            hspace=0.35, wspace=0.3)

    colors = ["#38bdf8", "#f472b6", "#fbbf24",
              "#34d399", "#a78bfa", "#fb923c"]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.axhline(0, color="#334155", lw=0.8, ls="--")

    for i, res in enumerate(results):
        c = colors[i % len(colors)]
        ax1.plot(res["times"], res["values"],
            color=c, lw=1.2, alpha=0.85)
        
        mask = np.abs(res["values"]) < 0.05
        ax1.scatter(res["times"][mask], res["values"][mask],
                    color=c, s=6, alpha=0.4, zorder=5)
        ax1.scatter([T], [res["xT"]], color=c, s=40, zorder=6)

    ax1.set_title(
        f"Dérives\n",
        color="#ffffff", fontsize=11, pad=10
    )
    ax1.set_xlabel("t", color="#94a3b8")
    ax1.set_ylabel("X_t", color="#94a3b8")
    ax1.tick_params(colors="#64748b")
    for spine in ax1.spines.values(): spine.set_edgecolor("#1f2d45")
    ax1.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")

    plt.savefig('trajectoires_ea.pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    X0    = 0
    T     = 10.0    
    THETA = 2.0   
    N_TRAJ = 8
    N_GRID = 300   

    print(f"Beskos-Roberts — SBM(θ={THETA}) avec les 3 bornes analytiques exactes")
    print(f"x0={X0}, T={T}, N_traj={N_TRAJ}, n_grid={N_GRID}")
    print("─" * 60)

    results = []
    for i in range(N_TRAJ):
        res = trajectory(x0=X0, T=T, theta=THETA, n_grid=N_GRID, seed=None)
        results.append(res)
        print(f"  Traj #{i+1:2d} | X_T = {res['xT']:+.4f} | rejets = {res['n_rejections']:3d} | PPP : {len(res['ppp_times'])} pts")

    total_rej = sum(r["n_rejections"] for r in results)
    print("─" * 60)
    print(f"Total rejets : {total_rej}  |  Moyenne : {total_rej/N_TRAJ:.1f}")
    plot_trajectories(results, T=T, theta=THETA, x0=X0)