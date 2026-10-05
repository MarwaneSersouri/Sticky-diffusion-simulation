import numpy as np
from scipy.special import erfcx
from scipy import integrate
import matplotlib.pyplot as plt

# Densités de base 

def q(t, x, theta):
    if t <= 0:
        return 1.0 if x == 0.0 else 0.0
    U = theta * np.sqrt(2 * t) + abs(x) / np.sqrt(2 * t)
    return float(np.clip(np.exp(-(x**2) / (2 * t)) * erfcx(U), 0.0, 1.0))

def p(t, x, y, theta):
    if t <= 0:
        return 0.0
    inv = 1.0 / np.sqrt(2 * np.pi * t)
    S   = abs(x) + abs(y)
    U   = theta * np.sqrt(2 * t) + S / np.sqrt(2 * t)
    em  = np.exp(-(S**2) / (2 * t))
    return float(inv * np.exp(-((x - y)**2) / (2 * t)) - inv * em + theta * em * erfcx(U))

def bridge_atom(t, x, T, b, theta):
    norm = p(T, x, b, theta) if abs(b) > 1e-10 else q(T, x, theta)
    if norm <= 0:
        return 0.0
    qt = q(t, x, theta)
    if abs(b) > 1e-10:
        return qt * p(T - t, 0.0, b, theta) / norm
    else:
        return qt * q(T - t, 0.0, theta) / norm

def bridge_density(t, x, z, T, b, theta):
    if abs(z) < 1e-10:
        return 0.0
    norm = p(T, x, b, theta) if abs(b) > 1e-10 else q(T, x, theta)
    if norm <= 0:
        return 0.0
    if abs(b) > 1e-10:
        return p(t, x, z, theta) * p(T - t, z, b, theta) / norm
    else:
        return p(t, x, z, theta) * q(T - t, z, theta) / norm

def phi(x):
    return 0.5 * (np.sin(x)**2 + np.cos(x))

def A_primitive(x):
    return np.cos(x)

K_CONST = -0.5

def psi_terminal(y, x0, T):
    return np.exp(A_primitive(y) - A_primitive(x0) - K_CONST * T)



# Paramètres globaux des plots


THETA = 1.0
T     = 1.0
X0    = 1.0
t_mid = T / 2      # instant intermédiaire pour les ponts
ys    = np.linspace(-4, 4, 1000)
ys_nz = ys[np.abs(ys) > 0.05]  # exclure z=0 pour la partie continue

colors = {
    "sbm":    "#38bdf8",
    "bridge": "#f472b6",
    "psi":    "#fbbf24",
    "phi":    "#34d399",
    "bm":     "#a78bfa",
}



# FIGURE 1 : Densités de transition du SBM  p(t, x0, y, theta)
#            et masse atomique q(t, x0, theta)


fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
fig.suptitle("Densités de transition du SBM — p(t, x₀, y, θ) et q(t, x₀, θ)",
             color="#e2e8f0", fontsize=12)

# Gauche : p(t, x0, y) pour différents t
ax = axes[0]
ax.set_facecolor("#0d1117")
ax.axvline(0, color="#334155", lw=0.8, ls="--")
t_vals = [0.1, 0.3, 0.5, 1.0, 2.0]
palette = ["#38bdf8", "#f472b6", "#fbbf24", "#34d399", "#a78bfa"]
for t_val, col in zip(t_vals, palette):
    density = [p(t_val, X0, y, THETA) for y in ys_nz]
    q_val   = q(t_val, X0, THETA)
    ax.plot(ys_nz, density, color=col, lw=1.5, label=f"t={t_val}  (q={q_val:.3f})")
    ax.axvline(0, color=col, lw=3, alpha=q_val, ymin=0, ymax=0.05)  # masse atomique

ax.set_title(f"p(t, x₀={X0}, y, θ={THETA})", color="#e2e8f0", fontsize=10)
ax.set_xlabel("y", color="#94a3b8")
ax.set_ylabel("densité", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

# Droite : q(t, x0) en fonction du temps
ax = axes[1]
ax.set_facecolor("#0d1117")
t_range = np.linspace(0.01, 3, 400)
for x_val, col in zip([0.0, 0.5, 1.0, 2.0], palette):
    q_vals = [q(t, x_val, THETA) for t in t_range]
    ax.plot(t_range, q_vals, color=col, lw=1.5, label=f"x₀={x_val}")
ax.set_title(f"q(t, x₀, θ={THETA}) en fonction de t", color="#e2e8f0", fontsize=10)
ax.set_xlabel("t", color="#94a3b8")
ax.set_ylabel("P(X_t = 0)", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

plt.tight_layout()
plt.savefig('sbm_densites_transition.pdf', bbox_inches='tight')
plt.close()



# FIGURE 2 : Les 4 cas du pont brownien collant  (section 7.7)
#
#   Cas 1 : b != 0, z != 0  → densité continue formule (1)
#   Cas 2 : b != 0, z = 0   → masse atomique formule (1)
#   Cas 3 : b = 0,  z != 0  → densité continue formule (2)
#   Cas 4 : b = 0,  z = 0   → masse atomique formule (2)


fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#0d1117")
fig.suptitle(
    f"Les 4 cas du pont brownien collant — x₀={X0}, t={t_mid}, T={T}, θ={THETA}\n"
    f"P(X_t ∈ dz | X_0=x₀, X_T=b)",
    color="#e2e8f0", fontsize=12
)

cas_params = [
    # (b, titre, formule)
    ( 1.5, "Cas 1 & 2 : b = 1.5 ≠ 0  — formule (1)", "#38bdf8"),
    ( 0.5, "Cas 1 & 2 : b = 0.5 ≠ 0  — formule (1)", "#f472b6"),
    ( 0.0, "Cas 3 & 4 : b = 0         — formule (2)", "#fbbf24"),
    (-1.0, "Cas 1 & 2 : b = -1.0 ≠ 0 — formule (1)", "#34d399"),
]

for ax, (b_val, titre, col) in zip(axes.flat, cas_params):
    ax.set_facecolor("#0d1117")
    ax.axvline(0, color="#334155", lw=0.8, ls="--")

    # Partie continue
    density = [bridge_density(t_mid, X0, z, T, b_val, THETA) for z in ys_nz]
    ax.plot(ys_nz, density, color=col, lw=2, label="densité continue")

    # Masse atomique en 0 (représentée comme une flèche verticale)
    atom = bridge_atom(t_mid, X0, T, b_val, THETA)
    ax.annotate("", xy=(0, atom), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.text(0.05, atom, f"atome={atom:.3f}", color=col, fontsize=8)

    # Vérification normalisation
    mass_cont, _ = integrate.quad(
        lambda z: bridge_density(t_mid, X0, z, T, b_val, THETA), -10, 10
    )
    total = mass_cont + atom
    ax.set_title(f"{titre}\n(normalisation : {total:.4f})", color="#e2e8f0", fontsize=9)
    ax.set_xlabel("z", color="#94a3b8")
    ax.set_ylabel("densité", color="#94a3b8")
    ax.tick_params(colors="#64748b")
    ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
    for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

plt.tight_layout()
plt.savefig('pont_brownien_4cas.pdf', bbox_inches='tight')
plt.close()



# FIGURE 3 : Poids de Girsanov
#
#   psi(y) = exp(A(y) - A(x0) - k*T)  — poids terminal
#   phi(x) - k                          — fonction du PPP
#   loi SBM biaisée par psi             — distribution de X_T sous P*


fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
fig.suptitle("Poids de Girsanov et loi de X_T sous P*", color="#e2e8f0", fontsize=12)

# Gauche : psi(y) = exp(cos(y) - cos(x0) - k*T)
ax = axes[0]
ax.set_facecolor("#0d1117")
psi_vals = [psi_terminal(y, X0, T) for y in ys]
ax.plot(ys, psi_vals, color=colors["psi"], lw=2)
ax.axvline(0, color="#334155", lw=0.8, ls="--", label=f"max en y=0 : ψ={psi_terminal(0,X0,T):.3f}")
ax.set_title(f"ψ(y) = exp(cos(y) − cos({X0}) − k·{T})", color="#e2e8f0", fontsize=9)
ax.set_xlabel("y", color="#94a3b8")
ax.set_ylabel("ψ(y)", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

# Milieu : phi(x) - k
ax = axes[1]
ax.set_facecolor("#0d1117")
xs = np.linspace(-np.pi, np.pi, 400)
phi_vals = np.array([phi(x) - K_CONST for x in xs])
ax.plot(xs, phi_vals, color=colors["phi"], lw=2, label="φ(x) − k")
ax.axhline(1.125, color="#f472b6", lw=1, ls="--", label="M_PPP = 1.125")
ax.fill_between(xs, 0, phi_vals, alpha=0.15, color=colors["phi"])
ax.set_title("φ(x) − k  (hauteur du PPP doit être AU-DESSUS)", color="#e2e8f0", fontsize=9)
ax.set_xlabel("x", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

# Droite : loi SBM de X_T vs loi biaisée par psi (P*)
ax = axes[2]
ax.set_facecolor("#0d1117")
sbm_density  = np.array([p(T, X0, y, THETA) for y in ys_nz])
psi_w        = np.array([psi_terminal(y, X0, T) for y in ys_nz])
biased       = sbm_density * psi_w
# Normaliser pour affichage
norm_sbm,  _ = integrate.quad(lambda y: p(T, X0, y, THETA), -10, 10)
norm_bias, _ = integrate.quad(
    lambda y: p(T, X0, y, THETA) * psi_terminal(y, X0, T), -10, 10
)
ax.plot(ys_nz, sbm_density / max(norm_sbm, 1e-10),
        color=colors["sbm"],    lw=1.5, ls="--", label="SBM  p(T,x₀,y)")
ax.plot(ys_nz, biased / max(norm_bias, 1e-10),
        color=colors["psi"],    lw=2,           label="P* = p × ψ / Z")
ax.set_title("Loi de X_T : SBM vs biaisée par ψ", color="#e2e8f0", fontsize=9)
ax.set_xlabel("y", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

plt.tight_layout()
plt.savefig('girsanov_poids_loi.pdf', bbox_inches='tight')
plt.close()

# FIGURE 4 : Illustration du test PPP
#
#   On trace phi(X_t) - k le long d'une trajectoire simulée
#   avec les points PPP (T_i, U_i) superposés
#   Verts = acceptés (U_i >= phi(X_{T_i}) - k)
#   Rouges = rejetés (U_i < phi(X_{T_i}) - k)


from ponts import trajectory  # import local

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
fig.suptitle("Illustration du test PPP sur une trajectoire acceptée",
             color="#e2e8f0", fontsize=12)

res = trajectory(x0=X0, T=T, theta=THETA, n_grid=300, seed=42)

# Gauche : trajectoire simulée
ax = axes[0]
ax.set_facecolor("#0d1117")
ax.axhline(0, color="#334155", lw=0.8, ls="--")
ax.plot(res["times"], res["values"], color=colors["sbm"], lw=1.5)
ax.scatter(res["ppp_times"],
           [bridge_density(ti, X0, 0, T, res["xT"], THETA) for ti in res["ppp_times"]],
           color="#fbbf24", s=20, zorder=5, label="instants PPP T_i")
ax.set_title(f"Trajectoire acceptée  (X_T={res['xT']:.3f})", color="#e2e8f0", fontsize=9)
ax.set_xlabel("t", color="#94a3b8")
ax.set_ylabel("X_t", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

# Droite : phi(X_{T_i}) - k vs U_i pour chaque point PPP
ax = axes[1]
ax.set_facecolor("#0d1117")
if len(res["ppp_times"]) > 0:
    # Interpoler les valeurs de X aux instants PPP
    x_at_ppp = np.interp(res["ppp_times"], res["times"], res["values"])
    phi_vals_ppp = np.array([phi(x) - K_CONST for x in x_at_ppp])
    u_vals = res["ppp_heights"]

    accepted_mask = u_vals >= phi_vals_ppp
    rejected_mask = ~accepted_mask

    ax.scatter(res["ppp_times"][accepted_mask], u_vals[accepted_mask],
               color="#34d399", s=30, zorder=5, label="U_i accepté")
    ax.scatter(res["ppp_times"][rejected_mask], u_vals[rejected_mask],
               color="#f87171", s=30, zorder=5, label="U_i rejeté")
    ax.plot(res["ppp_times"], phi_vals_ppp,
            color=colors["phi"], lw=1.5, ls="--", label="φ(X_{T_i}) − k")

ax.set_title("Test PPP : U_i vs φ(X_{T_i}) − k", color="#e2e8f0", fontsize=9)
ax.set_xlabel("T_i", color="#94a3b8")
ax.tick_params(colors="#64748b")
ax.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0", edgecolor="#1f2d45")
for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")

plt.tight_layout()
plt.savefig('test_ppp_trajectoire.pdf', bbox_inches='tight')
plt.close()
