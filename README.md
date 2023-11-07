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
* `benchmark.py`


**TODO**: 



* `tentative_greedy_RL_class.py` provides an example to learn Rule Lists with tunable depth, width, and minimum rules support, for both our custom version of CORELS and our Greedy Rule List classifier implementation


Folder `data` contains the datasets used in the experiments (and many others!)
