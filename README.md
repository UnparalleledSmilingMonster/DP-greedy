This repository is a fork from [repository](https://github.com/ferryjul/ProbabilisticDatasetsReconstruction) for the [paper](https://arxiv.org/pdf/2308.15099.pdf). We reuse his Greedy Rule Lists model and propose several Differentially Private versions of this model. We work with a rarely used framework for Differential Privacy : Smooth sensitivity.

Folder `pycorels` contains a modified version of the `CORELS` algorithm and its Python wrapper (original code available on [GitHub](https://github.com/corels/pycorels)).
Details of the changes are listed within the directory's README.

Folder `experiments` contains all the DP models and a benchmark file.

Pre-existing Files
* `HeuristicRL.py` contains the implementation of a Greedy Rule List Classifier (`GreedyRLClassifier` object, extending the classifier class proposed in CORELS for the sake of compatibility and efficiency)
* `utils_greedy.py` contains useful methods that are called by the Greedy Rule List classifier contained in `HeuristicRL.py`

DP models
* `HeuristicRL_DP_noise.py` is a DP Greedy Rule List Classifier based on the Noisy Max mechanism using Global sensitivity on the counts (either Laplace noise for pure DP or Gaussian noise for approximate DP)(deprecated because less performant than next model)
* `DP_global_old.py` is a DP Greedy Rule List Classifier based on the Noisy Max mechanism using Global sensitivity of the Gini Impurity (either Laplace noise for pure DP or Gaussian noise for approximate DP)
* `HeuristicRL_DP.py` is a DP Greedy Rule List Classifier based on Exponential mechanism using Global sensitivity 
* `HeuristicRL_DP_smooth.py` is a DP Greedy Rule List Classifier based on the Noisy Max mechanism using Smoothed sensitivity (either Cauchy noise for pure DP or Laplace noise for approximate DP)
* `DP.py` contains all utility functions with respect to Differentially Privacy mechanisms (especially the implementation of smooth sensitivity for the Gini Impurity)

Benchmark and tests
* `main.py` can parse arguments inputted through the command line to run a model with the desired hyperparameters (Preferably use that one)
* `sensitivity.py` is a srcipt to visualize the difference between noises using Smoothed Sensitivity against Global Sensitivity
* `benchmark.py` is a local benchmarking script testing various hyperparameters (max cardinality, max length, min support, confidence, DP-budget: epsilon, delta)
* `main_greedy_cauchy.py` runs a number of instances of **DpSmoothGreedyRLClassifier** using Cauchy noise, records the results in the folder "DP_results"
* `main_greedy_laplace.py`runs a number of instances of **DpSmoothGreedyRLClassifier** using Laplace noise, records the results in the folder "DP_results"
* `main_greedy_exp.py`runs a number of instances of **DPGreedyRLClassifier** (Exponential mechanism), records the results in the folder "DP_results"
* `report_VS_noisy_counts.py` compares the two DP models using either sensitivity of the counts or sensitivity of the Gini Impurity
* `expes.py` is another local benchmarking script calling the file `main.py`repeteadly to get well formattted results for statistics

Folder `pfcalcul` contains the code to parse experiments and visualize results : 
* `parse.py` Parses the results (computes the average accuracy over the seeds and variance) and writes in a table of results in a latex file. The results are stored in a **.nfo** file.
* `visualize.py` plots the results on a given dataset.

Work on model resilience to inference attacks & model interpretability
* `feature_interest.py` computes the top-k features of the DP models and compares them to the vanilla greedy model (we assess the interpretability of the DP models)
* `dist_overfit.py` computes the distributional overfit and overall vulnerability of our DP models
* `label_only_attack.py` This inference attack does not give good results on DP models because the exploration of the latent space is truncated since the datasets are binarized. It would be interesting to explore the latent space of the original features and compute them back to binary features.
* `attacks.py` Generates a Black Box Membership Inference Attacks on our DP models. 

Folder `data` contains the datasets used in the experiments (and many others!)

Some other files exist on the repository, they are either not complete / not meant for usage (mostly debugging). 


[//]: #Folder `light-dp-greedy` is a minimal version of the repository to be run on the distributed computation unit. --->

