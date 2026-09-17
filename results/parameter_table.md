# Table X. Parameter settings for all algorithms used in the experimental comparison

## SFLA (baseline) and MSFLA (proposed)

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Total number of frogs | F | 400 | Population size |
| Number of memeplexes | m | 20 | Number of sub-populations |
| Frogs per memeplex | n | 20 | F = m × n |
| Local search iterations | N | 15 | Iterations per memeplex before shuffling |
| Step size bound | s_max | 1.0 × (ub − lb) | Maximum allowed step size (fraction of search range) |

### MSFLA-specific (α–β controlled mechanism)

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Rotation step-length control | α | 1.0 | Scales the magnitude of the rotated update step |
| Rotation angle | θ (beta) | 30° | Fixed angle used to rotate the direction vector within the sampled 2D subspace |
| Truncated Gaussian std. dev. | σ | 0.7 | Standard deviation of the truncated Gaussian used to sample the step-length ratio |
| Truncated Gaussian mean | μ | 0.5 | Mean of the truncated Gaussian, bounded to [0, 1] |

## PSO (baseline)

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Swarm size | — | 40 | Number of particles |
| Inertia weight | w | 0.7298 | Standard constriction-based PSO setting |
| Cognitive coefficient | c1 | 1.49618 | Personal-best attraction coefficient |
| Social coefficient | c2 | 1.49618 | Global-best attraction coefficient |

## GA (baseline)

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Population size | — | 100 | Number of individuals |
| Crossover operator | — | BLX-α | Blend crossover |
| Crossover blend factor | α | 0.5 | BLX-α expansion factor |
| Crossover probability | p_c | 0.9 | Probability of applying crossover |
| Mutation probability | p_m | 1/D | Per-gene mutation probability (D = problem dimension) |
| Elitism | — | 2 | Number of best individuals carried over unchanged |

## DE (baseline)

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Population size | — | 50 | Number of individuals |
| Variant | — | DE/rand/1/bin | Mutation and crossover strategy |
| Scaling factor | F_scale | 0.5 | Differential weight |
| Crossover rate | CR | 0.9 | Binomial crossover probability |

## Function-evaluation budget (identical across all algorithms, per Osaba et al. 2021 recommendation)

| Dimension | Max FEs (CEC2022, F1–F12) | Max FEs (Scalability functions) |
|---|---|---|
| D = 10 | 200,000 | 100,000 |
| D = 20 | 1,000,000 | 200,000 |
| D = 30 | — | 300,000 |
| D = 50 | — | 500,000 |
| D = 100 | — | 1,000,000 |

*Note: For the CEC2022 benchmark, the FE budget follows the official competition protocol (10,000 × D, per the CEC2022 technical report). For the scalability study, the same 10,000 × D rule is applied uniformly across D = 10, 30, 50, 100 using five classical benchmark functions (Sphere, Ackley, Rastrigin, Rosenbrock, Schwefel).*

## Independent runs

All algorithm–function–dimension combinations were run for **30 independent trials** with different random seeds, in accordance with the statistical-testing requirement (Osaba et al., 2021, Swarm and Evolutionary Computation, 64:100888).
