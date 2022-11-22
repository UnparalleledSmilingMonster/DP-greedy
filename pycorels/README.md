Modified version of [pycorels](https://github.com/corels/pycorels) for experiments on probabilistic dataset reconstruction from interpretable models.
The main performed modifications are summarized hereafter.

### Search status and search stopping criteria

`get_status` indicates the seach status for the associated CorelsClassifier.
Returns either:

* "not_fitted": the search was not initialized yet (.fit() was not called yet)
* "exploration_not_started": search was initialized but is not running yet
* "exploration running": the prefix tree is being explored
* "time_out": search was stopped because CPU time (parameter `time_limit`) was reached
* "memory_out": search was stopped because maximum memory use (parameter `memory_limit`) was reached
* "max_nodes_reached": search was stopped because maximum number of nodes in the prefix tree (parameter `n_iter`) was reached
* "opt": the prefix tree was entirely explored/pruned and optimality is proved
* "unknown": should not happen


### Bounding the rule list size and support

* `max_length` parameter indicates maximum number of rules in the built rule list.

* `min_support` parameter is originally used in `CORELS` to mine only rules that capture at least a proportion of `min_support` of the training data.
In this new version, we also ensure that each rule captures at least a proportion of `min_support` of the training data IN THE CONTEXT OF THE RULE LIST.

### Models representation

Rule list models now include the per-class support for each rule (as is done in sklearn decision trees in each node/leaf).

