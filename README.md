# Numerical Companion to Paper: Wake, Saturation, Vacuum Residue, and the Double-Slit Assembly in Finite-Validation Models
 
*United Field Initiative (UFI). Companion to: "Physics from Finite Validation: Vacuum Energy, Quanta as Constituents of the Vacuum, and the Double-Slit Assembly" [DOI to be assigned]. Code: GitHub repository [link].*
 
## Purpose and Status Discipline
 
This companion contains every numerical run supporting Paper 2, with full code, parameters, seeds, and per-result status labels. The discipline follows the series: **confirmed in-model** (analytic and/or multi-seed numerical support), **indication** (suggestive, statistics insufficient), **rejected** (tested and failed), **artefact** (a result traced to the setup, documented as a lesson). Runs are exploratory instruments, not measurements of nature: "confirmed" always means confirmed *within the stated model*.
 
Two model regimes are used: a Kuramoto-type phase network (wake, saturation, vacuum residue) and a 2D sine-Gordon field (interference, assembly). Their continuum bridge is an open problem (Paper 2, §7.1); here they are treated as two instruments applied to one principle.
 
All scripts are standalone Python 3 (numpy only), each reproducible with the seeds given.
 
---
 
## Block A — The Wake (Kuramoto network)
 
**A.1 Setup.** Amorphous 3D network: N nodes uniform in a box, each linked to its deg nearest neighbours (spatial-hash kNN); Kuramoto phase dynamics with neighbour coupling K; a moving disturbance = a node of elevated intrinsic frequency (drive) relocated along a straight path. Measurement: phase/frequency deviation **relative to a control network** (identical initialisation, no disturbance) — this subtraction is what isolates the wake (background off-path: 0.0000).
 
**A.2 The wake exists and is tubular [confirmed in-model].**
- `wake_kuramoto.py` — 2D, global-mean metric: wake invisible (metric swamped). Rejected as setup, retained as a lesson.
- `wake_kuramoto_v2.py` — 2D, control-difference metric: wake present along the path; one-dimensional furrow; no velocity threshold detected.
- `wake_kuramoto_3d.py` — 3D regular lattice: a cone widening away from the body (apparent Kelvin-like wake).
- `wake_3d_robust.py` — the decisive control: the cone is **threshold-stable on the regular lattice but disappears on the amorphous network** of equal connectivity. Verdict: the cone is an **artefact of lattice axes**; on the isotropic network the wake is **tubular, of constant width**. This artefact/no-artefact pair is itself the numerical content of Paper 2, §5 (isotropy).
**A.3 Saturation [confirmed in-model, 8 seeds].**
- `tube_length_vs_energy.py` — tube length does not encode energy (length ≈ path; **rejected**: "length codes energy").
- `tube_energy_clean.py` — frequency-deviation metric (no π-ceiling): linear charge growth at small E, saturation above E≈2.
- `tube_saturation_vs_K.py` — plateau weakly dependent on K (not ∝ K).
- `stat_saturation.py` — statistics, 8 seeds × 5 energies, N=12000: linear start Q(1.0)/Q(0.5)=1.88≈2; plateau **3.53±1.06** (CV~30%); growth E=2→E=8 equal to 0.90 (no growth under a fourfold energy increase). Claimable: *saturation at a stable level with ±30% spread*. Not claimable: a precise "quantum of disturbance."
**A.4 Tube width [indication].**
- `tube_width.py` — width is not fundamental: it scales with connectivity (deg) and density; in the dense limit at fixed deg it narrows toward ~1–2 network steps (a "thread" at the discreteness scale). Two-point trend; statistics insufficient.
**A.5 Negative results [rejected; motivation for Block C].**
- The disturbance does not self-propagate: it dies without the body (screen signal 0.000 in `double_slit_v1.py`).
- Naive phase inertia (second order) produces global instability, not a travelling wave (`selfprop_test.py`).
These failures motivate the change of instrument to a field with stable travelling solutions (sine-Gordon).
---
 
## Block B — Vacuum Residue (Kuramoto mean-field)
 
**B.1 Sync-mean vs naive sum [confirmed in-model].**
- `vacuum_sync_mechanism.py` — 300 normal nodes (σ=0.15) + 300 "quantum" modes up to a cutoff raised 10⁶→10¹⁸: the naive sum grows 2.8×10⁷→9.2×10¹⁸ (extensive; the catastrophe in miniature — demonstrating one property only: cutoff-dependence); the sync-mean stays 1.0032, flat (intensive). Lock check: normal ~100%, quanta 0% at all cutoffs. Status caution (Paper 2, §3): within the model's chosen g(ω), the exclusion restates Paper 1's no-quantum-links postulate dynamically; an independent g(ω) is debt 7.3.
- `vacuum_sync_diagnostic.py` — detector diagnostics (cluster-median lock criterion).
**B.2 The residue law ε = σ²/(2K²) [derived + confirmed].**
- `vacuum_residual_law.py` — derivation: stationary capture, sin φ = (ω−Ω)/KR, narrow band → ε ≡ 1−R = σ²/2K². Numerics: σ-scan (K=1) ratio-to-theory 0.97/0.99/1.01/1.04/1.19 (departure only at σ=0.3, the boundary of the expansion); K-scan (σ=0.15) 1.11→0.98; global log-log fit: exponents 2.10 (σ) and 2.10 (K) vs theoretical 2/2, prefactor 0.624 vs 0.5, **R²=0.9992**. The strongest technical result of the companion.
**B.3 The Λ estimate [two-sided, stated with the minus first].**
- `lambda_estimate.py` — observational input (sources verified): ρ_Λ≈6×10⁻¹⁰ J/m³, ρ_P=4.63×10¹¹³ J/m³. Required residue ε≈1.3×10⁻¹²³ → σ/K≈5×10⁻⁶². "Reasonable" spreads overshoot by 111–121 orders. Minus: the value of Λ is not derived; the riddle is translated (magnitude → homogeneity). Plus: the translated form has a motivating analogy (quantum identity of particles of one kind) — with the disanalogy stated (exact/discrete vs approximate/continuous); and the relation σ/K=√(2ρ_Λ/ρ_P) reads Λ as the metre of node homogeneity.
- `waves_lattice_dispersion.py` — long-wavelength linearity (ω=ck), no directional corrections: the dispersion check cited in Paper 2 §4–5.
---
 
## Block C — The Double Slit (2D sine-Gordon)
 
**C.1 Instrument validation [confirmed].**
- `sine_gordon_soliton.py` — 1D kink: launched at v=0.4c, measured 0.395c; shape error 0.0016 after a path of 80 units. Stable particle-like travelling solutions exist in the instrument.
- Setup lessons [artefacts, documented]: `sg_soliton_slits.py`, `sg_soliton_slits_v2.py` — a topological kink cannot be boxed (θ=2π fills the half-plane; "energy beyond the wall" was there at t=0). Lesson: the "particle" must be a **non-topological localised packet**.
- `sg_packet_slits.py` — wall validation: solid wall holds 200:1 (E_right 1.9 vs 399 with no wall).
**C.2 Interference and control [confirmed, then statistics].**
- `sg_double_slit.py` — front through two slits: 8 symmetric peaks (pre-control run).
- `sg_slit_control.py` — two slits vs one, absorbing boundaries: two slits give troughs opposite the slits (y80=0.601, y120=0.615) and a central maximum (y100=1.506), mirror-symmetric peak pairs; one slit gives an asymmetric diffraction picture. 
- `sg_packet_slits.py` — the packet ("particle") through two slits: central maximum y=100, side peaks 61/139, troughs opposite the slits 0.026 vs 0.137 (single run); energy beyond the barrier 27.9 (two) vs 14.6 (one) — passage through both corridors, ratio ≈ 2.
**C.3 Robustness and decoherence threshold [confirmed, 5 seeds; measured].**
- `stat_slits.py` — a deliberately harsh test: field noise 0.02 across the whole domain kills the pattern (contrast 1.1±0.1). Lesson: interference drowns in background of comparable energy.
- `stat_slits_v2.py` — realistic variations (noise 0.003, launch jitter ±2, amplitude ±10%; 5 seeds): central peak 99.6±0.5, contrast 3.7±0.4 — robust. Noise scan: contrast 4.8 (0) → 4.1 (0.002) → 2.9 (0.005) → 1.8 (0.01) → 1.2 (0.02): the pattern dies at ~1% background — a measured decoherence threshold.
**C.4 The assembly fork [confirmed in-model; the discriminator].**
- `sigma_candidates.py` — symmetric slits, energy scan A=0.4..2.0: all candidates (argmax|θ|, argmax|θ_t|, centroid) coincide at y≈100 — symmetry masks the fork. The tick at the assembly point grows with packet energy (0.0795→0.2458).
- `sigma_asymmetric.py` — slits of widths 2 vs 6: candidates separate — max|θ| y=134, max|θ_t| y=99, centroid y=114.6. The fork is real.
- `sigma_energy.py` — the fourth candidate, energy density E=θ_t²/2+c²|∇θ|²/2+(1−cosθ): accumulated and peak argmax both at y=99 — energy lands on the tick (θ_t² dominates). Energy = tick, as one point.
- `stat_fork.py` — 5 seeds with realistic variations: gap tick↔centroid 15.3±2.3 (all seeds >12); accumulated energy 99.4±1.0 (the most stable metric); instantaneous |θ_t| is noisy (one outlier y=136 while energy stayed at 98) — the robust carrier is the **accumulated** energy.
- Consequence (Paper 2, §6.5): "address" assembly (strictly at the maximum) is rejected by the observed pattern; weight ∝ accumulated energy density reproduces the Born profile, with θ_t² as the structural carrier of the square. The weight is postulated, not derived (debt 7.2). Asymmetric setups are the discriminator between assembly hypotheses.
---
 
## Reproducibility
 
Python 3.10+, numpy (matplotlib for figure scripts). Every script runs standalone; seeds are written in the code. Per-seed outputs for the statistics blocks are printed by the scripts themselves. Runtime: seconds to ~minutes per script on a laptop; the 8-seed saturation statistics is the longest (~10 min).
 
## File → Paper 2 map
 
| Script | Paper 2 section | Status |
|---|---|---|
| wake_kuramoto.py | §5 (lesson) | rejected setup |
| wake_kuramoto_v2.py | §5 | confirmed (furrow) |
| wake_kuramoto_3d.py | §5 | superseded (cone) |
| wake_3d_robust.py | §5 | confirmed (artefact vs tube) |
| tube_length_vs_energy.py | §1 | rejected (length≠energy) |
| tube_energy_clean.py | §1 | confirmed (saturation) |
| tube_saturation_vs_K.py | §1 | confirmed (K-stability) |
| stat_saturation.py | §1 | confirmed, 8 seeds |
| tube_width.py | §1 | indication |
| selfprop_test.py | §6.3 (motivation) | rejected (instability) |
| double_slit_v1.py | §6.3 (motivation) | rejected (no self-propagation) |
| vacuum_sync_mechanism.py | §3 | confirmed (sync-mean flat) |
| vacuum_sync_diagnostic.py | §3 | diagnostics |
| vacuum_residual_law.py | §3.2 | derived + confirmed |
| lambda_estimate.py | §3.4 | two-sided estimate |
| waves_lattice_dispersion.py | §4–5 | confirmed (ω=ck) |
| sine_gordon_soliton.py | §6.3 | instrument validation |
| sg_soliton_slits.py / _v2.py | §6.3 (lessons) | artefacts, documented |
| sg_double_slit.py | §6.4 | pre-control run |
| sg_slit_control.py | §6.4 | confirmed (control) |
| sg_packet_slits.py | §6.4 | confirmed (packet) |
| stat_slits.py | §6.4 | lesson (harsh noise) |
| stat_slits_v2.py | §6.4 | confirmed, 5 seeds + threshold |
| sigma_candidates.py | §6.5 | confirmed (symmetric) |
| sigma_asymmetric.py | §6.5 | confirmed (fork) |
| sigma_energy.py | §6.5 | confirmed (energy=tick) |
| stat_fork.py | §6.5 | confirmed, 5 seeds |
 
## License
Code MIT; text and figures CC BY 4.0. © United Field Initiative.
