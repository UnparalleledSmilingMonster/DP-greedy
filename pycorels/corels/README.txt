Modified version of pycorels for experiments of probabilistic dataset reconstruction from interpretable models.

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


### Maximum rule list depth bounds 

`max_length` parameter indicates maximum number of rules in the built rule list.

