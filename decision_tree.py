from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy
import pandas as pd

feature_names = ['id', 'diagnosis',
        'mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
        'radius SE','texture SE','perimeter SE','area SE','smoothness SE','compactness SE','concavity SE','concave points SE','symmetry SE','fractal dimension SE',
        'worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension',
    ]

def prepare_data(file_path):
    df = pd.read_csv(file_path, header=None, names=feature_names)
    X = df.drop(columns=['id', 'diagnosis'])
    Y = df['diagnosis']
    return train_test_split(X, Y, test_size=0.2)


# plot the test, train and 5-cross validation average accuracies
# for the decision tree classifiers with max depth [1, 30)
def tune_max_depth(max_depth, train_X, train_Y, test_X, test_Y):
    val_accuracies, test_accuracies, train_accuracies = [], [], []
    depths = range(1, max_depth)
    for depth in depths:
        tree_classifier = tree.DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(tree_classifier, train_X, train_Y, cv=5)
        val_accuracies.append(numpy.sum(scores)/5)
        tree_classifier.fit(train_X, train_Y)
        train_accuracies.append(tree_classifier.score(train_X, train_Y))
        test_accuracies.append(tree_classifier.score(test_X, test_Y))
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.plot(depths, val_accuracies, label='5-fold Validation Average Accuracies')
    plt.plot(depths, test_accuracies, label='Test Accuracies')
    plt.plot(depths, train_accuracies, label='Train Accuracies')
    plt.legend()
    plt.show()


def logistic_regression(train_X, train_Y, test_X, test_Y, feature_importance_df):
    accuracies = []
    num_selected_features = [5, 10, 15, 20]
    for n in (num_selected_features):
        # select the 5,10,15,20 most important features
        selected = feature_importance_df.head(n)['Feature']
        trainX_selected = train_X[selected]
        testX_selected = test_X[selected]

        # need to scale x
        scaler = preprocessing.StandardScaler()
        trainX_scaled = scaler.fit_transform(trainX_selected)
        testX_scaled = scaler.transform(testX_selected)

        # train and test the logistic regression classifier
        model = LogisticRegression()
        model.fit(trainX_scaled, train_Y)
        accuracies.append(model.score(testX_scaled, test_Y))

    plt.xlabel('Number of Selected Features')
    plt.ylabel('Accuracy')
    plt.plot(num_selected_features, accuracies)
    plt.show()


def evaluate_decision_tree(max_depth, train_X, train_Y, test_X, test_Y):
    # train and test the decision tree
    decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
    decision_tree = decision_tree.fit(train_X, train_Y)
    train_accuracy = decision_tree.score(train_X, train_Y)
    test_accuracy = decision_tree.score(test_X, test_Y)
    print(f'DECISION TREE\nTraining Accuracy: {train_accuracy:.4f}\nTest Accuracy: {test_accuracy:.4f}')

    # plot the decision tree
    class_names = ["M", "B"]
    plt.figure(figsize=(12,12))
    tree.plot_tree(decision_tree, class_names=class_names, feature_names=feature_names[2:], filled=True, fontsize=6)
    plt.show()
    
    # get the calculated importances of the features
    feature_importance_df = pd.DataFrame({'Feature': feature_names[2:], 'Importance': decision_tree.feature_importances_})
    return feature_importance_df.sort_values(by='Importance', ascending=False)


def evaluate_random_forests(train_X, train_Y, test_X, test_Y):
    # train and test a random forest without specifying number of trees
    random_forest = RandomForestClassifier()
    random_forest.fit(train_X, train_Y)
    test_acc = random_forest.score(test_X, test_Y)
    print("Random forest test accuracy:", test_acc)

    # train and test random forests having "num_trees" trees
    train_accuracies, test_accuracies = [], []
    num_trees = [5, 10, 25, 50, 100, 150]
    for num_tree in num_trees:
        random_forest = RandomForestClassifier(n_estimators=num_tree)
        random_forest.fit(train_X, train_Y)
        train_accuracies.append(random_forest.score(train_X, train_Y))
        test_accuracies.append(random_forest.score(test_X, test_Y))
    
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.plot(num_trees, train_accuracies, linestyle='--', marker='o', label='Training')
    plt.plot(num_trees, test_accuracies, linestyle='--', marker='o', label='Test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    numpy.random.seed(42)   # seed is given to obtain the same result in each run
    train_X, test_X, train_Y, test_Y = prepare_data("./breast+cancer+wisconsin+diagnostic/wdbc.data")
    tune_max_depth(30, train_X, train_Y, test_X, test_Y)    # 30 is the number of total features

    max_depth = 4   # 4 is decided after analyzing the accuracies
    feature_importance_df = evaluate_decision_tree(max_depth, train_X, train_Y, test_X, test_Y)
#    print(feature_importance_df)

    logistic_regression(train_X, train_Y, test_X, test_Y, feature_importance_df)
    evaluate_random_forests(train_X, train_Y, test_X, test_Y)