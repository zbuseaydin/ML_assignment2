from sklearn import tree
import matplotlib.pyplot as plt

def prepare_data(file_path):
    data_file = open(file_path)
    data = data_file.read().split("\n")[:-1]

    class_labels = []
    x_values = []
    diagnosis = {"M":0, "B": 1}

    for entry in data:
        entry = entry.split(",")
        class_labels.append(diagnosis[entry[1]])
        x_values.append(entry[2:])
    
    return x_values, class_labels


def performance(tree_classifier, x, real_y):
    misclassified = 0
    decision = tree_classifier.predict(x)
    for i in range(len(real_y)):
        if decision[i] != real_y[i]:
            misclassified += 1
    return 1 - (misclassified/len(real_y))


def train(x, y, max_depth):
    tree_classifier = tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier = tree_classifier.fit(x, y)
    return tree_classifier


def tune_max_depth(max_depth, x_train, y_train, x_test, y_test):
    test_perfs, train_perfs = [], []
    depths = range(1, max_depth)
    for depth in range(1, max_depth):
        cur_tree = train(x_train, y_train, depth)
        test_perf = performance(cur_tree, x_test, y_test)
        train_perf = performance(cur_tree, x_train, y_train)
        test_perfs.append(test_perf)
        train_perfs.append(train_perf)
        
    return test_perfs, train_perfs, depths


if __name__ == "__main__":
    X, Y = prepare_data("./breast+cancer+wisconsin+diagnostic/wdbc.data")
    training_size = int(len(X) * 0.8)
    training_X, training_Y = X[ :training_size], Y[ :training_size]
    test_X, test_Y = X[training_size+1: ], Y[training_size+1: ]
    test_perf, train_perf, depths = tune_max_depth(20, training_X, training_Y, test_X, test_Y)
    

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(depths, test_perf)
    ax1.set_ylabel("Test Performance")
    ax2.plot(depths, train_perf, color='red')
    ax2.set_ylabel("Train Performance")
    plt.show()

    # max_depth = 5 is decided after tuning
    decision_tree = train(training_X, training_Y, 5)
    tree.plot_tree(decision_tree)
    plt.show()