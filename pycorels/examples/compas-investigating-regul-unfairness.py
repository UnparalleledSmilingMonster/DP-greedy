from corels import *
import numpy as np

# Train split proportion
train_proportion = 0.8

X, y, features, prediction = load_from_csv("data/compas.csv")

accuracy_list = dict()
groups_size = dict()
indicator_list = ["Race=African-American", "Race=Caucasian", "Race=Native-American", "Race=Asian", "Race=Other"]

regul_list_1 = np.arange(0.1, 1.0, 0.1)
regul_list_2 = np.arange(0.01, 0.1, 0.01)
regul_list_3 = np.arange(0.001, 0.01, 0.001)
regul_list_4 = np.arange(0.0001, 0.001, 0.0001)

regul_list = np.concatenate([regul_list_1, regul_list_2, regul_list_3, regul_list_4])

for regul_param in regul_list:
    print("LAMBDA = ", regul_param)
    # A maximum cardinality of 3 makes CORELS search all rule antecedents
    # with up to three features combined together
    c = CorelsClassifier(max_card=3, n_iter=10**5, policy='bfs', verbosity=["progress","rulelist"], c=regul_param)

    # Generate train and test sets
    train_split = int(train_proportion * X.shape[0])

    X_train = X[:train_split]
    y_train = y[:train_split]

    X_test = X[train_split:]
    y_test = y[train_split:]

    # Fit the model. Features is a list of the feature names
    c.fit(X_train, y_train, features=features, prediction_name=prediction)

    # Score the model on the test set
    a = c.score(X_test, y_test)

    print("Test Accuracy: " + str(a))

    # Print the rulelist
    print(c.rl())

    # Fairness audit
    def compute_subgroup_accuracy(subgroup_indicator, X_all, y_all, predictions_all):
        subgroup_indicator_id = features.index(subgroup_indicator)
        if subgroup_indicator_id < 0:
            print("Indicator %s not in data!" %subgroup_indicator)
            return 0
        group_indices = np.where(X_train[:,subgroup_indicator_id] == 1)[0]
        print("There are %d examples in group %s." %(group_indices.size, subgroup_indicator))
        X_subgroup = X_all[group_indices,:]
        y_subgroup = y_all[group_indices]
        preds_subgroup = predictions_all[group_indices]

        from sklearn.metrics import accuracy_score
        subgroup_accuracy = accuracy_score(y_true=y_subgroup, y_pred=preds_subgroup)
        print("Accuracy for group %s is: %.3f" %(subgroup_indicator, subgroup_accuracy))
        return group_indices.size, subgroup_accuracy
    train_predictions = c.predict(X_train)

    for an_indicator in indicator_list:
        size, acc = compute_subgroup_accuracy(an_indicator, X_train, y_train, train_predictions)
        if an_indicator not in accuracy_list:
            accuracy_list[an_indicator] = [acc]
            groups_size[an_indicator] = size
        else:
            accuracy_list[an_indicator].append(acc)

# Fairness (subgroup accuracy parity) plot
import matplotlib.pyplot as plt

for an_indicator in indicator_list:
    plt.plot(regul_list, accuracy_list[an_indicator], label=an_indicator, marker='x')

plt.legend(loc='best')
plt.show()