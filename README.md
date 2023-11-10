This repository is a fork from Julien Ferry's repo for probabilistic datasets reconstruction from interpretable models. I reuse his Greedy Rule Lists model and propose several Differentially Private versions of this model. We use a rarely used framework for Differential Privacy : Smoothed sensitivity.

Folder `pycorels` contains a modified version of the `CORELS` algorithm and its Python wrapper (original code available on [GitHub](https://github.com/corels/pycorels)).
Details of the changes are listed within the directory's README.

Folder `experiments` contains all the DP models and a benchmark file.

Pre-existing Files
* `HeuristicRL.py` contains the implementation of a Greedy Rule List Classifier by Julien Ferry (`GreedyRLClassifier` object, extending the classifier class proposed in CORELS for the sake of compatibility and efficiency)
* `utils_greedy.py` contains useful methods that are called by the Greedy Rule List classifier contained in `HeuristicRL.py`

DP models
* `HeuristicRL_DP_noise.py` is a DP Greedy Rule List Classifier based on the Noisy Max mechanism using Global sensitivity (either Laplace noise for pure DP or Gaussian noise for approximate DP)
* `HeuristicRL_DP.py` is a DP Greedy Rule List Classifier based on Exponential mechanism using Global sensitivity 
* `HeuristicRL_DP_smooth.py` is a DP Greedy Rule List Classifier based on the Noisy Max mechanism using Smoothed sensitivity (either Cauchy noise for pure DP or Laplace noise for approximate DP)

Benchmark and tests
* `sensitivity.py` is a srcipt to visualize the difference between noises using Smoothed Sensitivity against Global Sensitivity
* `benchmark.py` is a local benchmarking script testing various hyperparameters (max cardinality, max length, min support, confidence, DP-budget: epsilon, delta)
* `main_greedy_cauchy.py` runs a number of instances of **DpSmoothGreedyRLClassifier** using Cauchy noise, records the results in the folder "DP_results"
* `main_greedy_laplace.py`runs a number of instances of **DpSmoothGreedyRLClassifier** using Laplace noise, records the results in the folder "DP_results"
* `main_greedy_exp.py`runs a number of instances of **DPGreedyRLClassifier** (Exponential mechanism), records the results in the folder "DP_results"
* `main.py` can parse arguments inputted through the command line to run a model with the desired hyperparameters

Folder `data` contains the datasets used in the experiments (and many others!)
Folder `pfcalcul` contains the code to run benchmarks on the distributed computation unit.

[//]: #Folder `light-dp-greedy` is a minimal version of the repository to be run on the distributed computation unit.
