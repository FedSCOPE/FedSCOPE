# Experiment protocol

Primary focal: `DE_KN_industrial3`.

Primary data-quality rule: source-marked interpolation excluded at either cumulative-meter endpoint.

Primary horizon: 24 h (96 quarter-hours).

Primary uncertainty target: 95%, with mandatory 90% and 99% sensitivities.

Primary market analysis: all causal expanding-window episodes, with strategic metrics explicitly restricted to exact game-active states. No season is selected after observing performance.

Forecast baselines: weekly seasonal naive, local OLS, fixed-effects pooled OLS, federated empirical-Bayes random effects.

Primary signaling baselines: truthful priority and minimum-service priority. Local, pooled, uncertainty, and oracle variants are retained as diagnostic ablations in generated outputs.

Mandatory sensitivity suite: 6 h, 12 h and 24 h; 90/95/99%; strict vs source interpolation; every configured client with measured essential proxy as focal.

Hard invalidation conditions in `fedscope.audit_run`: server raw-row exposure; zero genuine game-active states; constant realized integrated-distress target on game-active states; strategic-federated losing service relative to minimum-service beyond machine-level tolerance.
