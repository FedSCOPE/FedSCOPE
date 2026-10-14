# Primary sources informing the implementation

1. Open Power System Data, Household Data / CoSSMic, version 2020-04-15. Data page: https://data.open-power-system-data.org/household_data/
2. Open Power System Data, Time Series, version 2018-03-13. Data page: https://data.open-power-system-data.org/time_series/2018-03-13/
3. Smith, V., Chiang, C.-K., Sanjabi, M., & Talwalkar, A. (2017). Federated Multi-Task Learning. NeurIPS 30. https://papers.nips.cc/paper_files/paper/2017/hash/6211080fa89981f66b1a0c9d55c61d0f-Abstract.html
4. Thapa, M. S., & Li, R. (2025). Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference. ICML 2025 / PMLR 267. https://proceedings.mlr.press/v267/thapa25a.html
5. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. Annals of Statistics. (Distance-correlation definition; implementation is the standard biased sample statistic.)

The empirical-Bayes linear random-effects estimator in this package is implemented directly from the Gaussian hierarchical model specified in `docs/METHOD.md`; it is not presented as a reproduction of any one cited personalized-FL algorithm.
6. Kelly, F. P., Maulloo, A. K., & Tan, D. K. H. (1998). Rate control in communication networks: shadow prices, proportional fairness and stability. Journal of the Operational Research Society 49, 237–252. https://www.statslab.cam.ac.uk/~fpk1/rate.html
