This repository contains the code of our experiments regarding probabilistic datasets reconstruction from interpretable models.

Folder `pycorels` contains a modified version of the `CORELS` algorithm and its Python wrapper (original code available on [GitHub](https://github.com/corels/pycorels)).
Details of the changes are listed within the directory's README.

Folder `experiments` contains all the scripts and data to reproduce our experiments and Figures.

* `toy_example_paper.py` contains the code to generate the decision tree presented in our paper (Figure TODO)
* `expes_utils.py` contains useful methods that are called in our experiments scripts
* `HeuristicRL.py` contains our implementation of a Greedy Rule List Classifier (extending the classifier class proposed in CORELS for the sake of compatibility and efficiency)
* `utils_greedy.py` contains useful methods that are called by our Greedy Rule List classifier contained in `HeuristicRL.py`
* `tentative_greedy_RL_class.py` provides an example to learn Rule Lists with tunable depth, width, and minimum rules support, for both our custom version of CORELS and our Greedy Rule List classifier implementation

For Decision Trees:
* `learn_compute_entropy_binary_rl_dt_trees.py` can be used to run the experiments comparing optimal (learnt with DL8.5) and non-optimal (learnt with sklearn implementation of CART) Decision Trees
* `learn_compute_entropy_binary_rl_dt_trees_batch.sh` can be used to launch the previously mentioned experiments on a computing platform
* `analyze_results_heuristic_vs_optimal.py` can be used to generate the Figures comparing optimal (learnt with DL8.5) and non-optimal (learnt with sklearn implementation of CART) Decision Trees

Folder `data` contains the datasets used in our experiments (and many others!)