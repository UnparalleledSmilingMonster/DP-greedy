from sklearn.tree import DecisionTreeClassifier

plot_extension = 'pdf'

X = [[12, 0, 3],
    [14, 1, 2],
    [11, 1, 2],
    [14, 0, 1]]
y = [0, 0, 1, 1]
features = ["$A_1$", "$A_2$", "$A_3$"]
clf = DecisionTreeClassifier(random_state=42, max_depth=2, min_samples_leaf=1)
clf.fit(X, y) 
print("DT accuracy = ", clf.score(X, y))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import matplotlib
import re

fig, ax = plt.subplots(figsize=(8,5))

plot_tree(clf, ax=ax, feature_names = features)

def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txt = txt.split("\n")
        newtxt = ""
        first = True
        for line in txt:
            if not ("gini" in line):
                if first:
                    newtxt = newtxt + line
                    first = False
                else:
                    newtxt = newtxt + "\n" + line
        obj.set_text(newtxt)
    return obj
    
ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]


fig.savefig("./tree_toy_example.%s" %(plot_extension), bbox_inches='tight')
plt.clf()