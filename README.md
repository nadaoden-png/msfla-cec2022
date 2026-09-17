# MSFLA: An α–β Controlled Modified Shuffled Frog-Leaping Algorithm

This repository contains the full source code, raw experimental data, and analysis
scripts supporting the manuscript:

> "An α–β Controlled Modified Shuffled Frog-Leaping Algorithm"
> Submitted to *Engineering Optimization* (Manuscript ID: 269544619)

## Overview

MSFLA extends the classical Shuffled Frog-Leaping Algorithm (SFLA) with an
α–β controlled rotation mechanism: the local-search update direction is rotated
by a fixed angle θ within a randomly sampled two-dimensional subspace spanned by
the original direction vector and a randomly generated orthogonal complement
(via Gram–Schmidt orthogonalization), with the step length drawn from a
truncated Gaussian distribution.

## Repository structure

```
├── src/
│   ├── algorithms.py              # SFLA, MSFLA, PSO, GA, DE implementations
│   ├── benchmark_functions.py     # Classical functions used in the scalability study
│   ├── cec2022_bridge.py          # Python wrapper around the official CEC2022 C code
│   ├── CEC2022/                   # Official CEC2022 benchmark code (patched for speed)
│   ├── run_batch.py                # CEC2022 experiment runner (D=10, D=20)
│   ├── run_batch_scale.py          # Scalability experiment runner (D=10,30,50,100)
│   ├── results_store.py            # Result persistence for CEC2022 runs
│   └── results_store_scale.py      # Result persistence for scalability runs
├── data/
│   ├── results_db.json             # Raw CEC2022 results (D=10 + D=20, 5 algorithms, 30 runs each)
│   └── results_scale.json          # Raw scalability results (4 dimensions, 5 functions, 5 algorithms, 30 runs each)
├── results/                        # Statistical analysis summaries (Shapiro-Wilk, Friedman, Wilcoxon+Holm)
├── figures/                        # Convergence plots for every function/dimension combination
├── requirements.txt
└── LICENSE
```

## Algorithm parameters

See `results/parameter_table.md` for the full parameter settings used for every
algorithm (SFLA, MSFLA, PSO, GA, DE), matching Table X in the manuscript.

## Reproducing the experiments

```bash
pip install -r requirements.txt

# CEC2022 benchmark (D=10 or D=20), all 12 functions, all 5 algorithms, 30 runs each
python src/run_batch.py --dim 10 --funcs 1-12 --algos SFLA,MSFLA,PSO,GA,DE --time_budget 3600

# Scalability study (classical functions), D in {10,30,50,100}
python src/run_batch_scale.py --dim 30 --funcs sphere,ackley,rastrigin,rosenbrock,schwefel \
    --algos SFLA,MSFLA,PSO,GA,DE --time_budget 3600
```

Both scripts resume automatically from `data/results_db.json` /
`data/results_scale.json` if interrupted, and stop once every combination
reaches the target of 30 independent runs.

## Function-evaluation budget

Following the CEC2022 competition protocol and the statistical-testing
guidance of Osaba et al. (2021, *Swarm and Evolutionary Computation*, 64:100888),
all algorithms are evaluated under an identical FE budget of 10,000 × D per run,
with 30 independent trials per algorithm–function–dimension combination.

## Statistical methodology

1. **Shapiro–Wilk** test for normality of each (function, algorithm) sample.
2. **Friedman** test to detect an overall significant difference among the
   five algorithms.
3. **Wilcoxon signed-rank** test with **Holm's step-down correction** for
   pairwise comparisons against MSFLA.

## Comparison with the CEC2022 competition winner (EA4Eig)

Raw EA4Eig results were obtained directly from the official competition
repository (P-N-Suganthan/2022-SO-BO) rather than re-implemented, to avoid
any risk of implementation drift from the published results.

## Citation

If you use this code, please cite the manuscript once published, and the
original CEC2022 technical report:

> Kumar, A., Price, K.V., Mohamed, A.W., Hadi, A.A., & Suganthan, P.N. (2022).
> Problem definitions and evaluation criteria for the CEC 2022 special session
> and competition on single objective bound constrained numerical optimization.

## License

MIT License — see `LICENSE`.

## Contact

Dr. Kanchana Daoden, Uttaradit Rajabhat University
kanchana.dao@uru.ac.th
