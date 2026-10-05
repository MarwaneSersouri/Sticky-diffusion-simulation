# Sticky Brownian Motion — Exact Simulation

Projet de recherche pour la simulation exacte (sans biais de discrétisation) du **Mouvement Brownien Collant** (*Sticky Brownian Motion*). 

---

### Description
Le mouvement brownien collant (SBM) ralentit en $0$ avec une intensité $\theta > 0$. Les méthodes usuelles (Euler-Maruyama) sont biaisées à cause de la singularité. Ce projet implémente :
1. **Simulation pas-à-pas sans biais** : rejet avec enveloppe gaussienne optimale ($M = 1$).
2. **Simulation exacte de trajectoires avec dérive** : adaptation de l'algorithme rétrospectif de Beskos-Roberts (changement de mesure de Girsanov, pont brownien collant et processus ponctuel de Poisson).

### Fichiers
- `ponts.py` : Algorithme exact de Beskos-Roberts (SBM avec dérive $b(x)=\sin(x)$).
- `simulation.py` : Simulation pas-à-pas par acceptation-rejet ($M=1$) et suivi des rejets.
- `compos_n_fois.py` : Validation de l'absence de biais (1 saut vs composition de $N$ pas).
- `densites_enveloppe.py` & `visu_densites.py` : Tracé des densités, de l'atome en 0 et du test PPP.
- `cdf.py` : Fonction de répartition (saut en $0$).
- `memoire.pdf` : Mémoire de recherche complet (théorie et preuves).


