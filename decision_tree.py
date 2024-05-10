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


def tune_max_depth(max_depth, x_train, y_train):
    accuracies = []
    depths = range(1, max_depth)
    for depth in depths:
        tree_classifier = tree.DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(tree_classifier, x_train, y_train, cv=5)
        accuracies.append(numpy.sum(scores)/5)
    plt.xlabel('Max Depth')
    plt.ylabel('Average Accuracy (5-fold)')
    plt.plot(depths, accuracies)
    plt.show()
    return numpy.argmax(accuracies)+1


def logistic_regression(train_X, train_Y, test_X, test_Y, feature_importance_df):
    accuracies = []
    num_selected_features = [5, 10, 15, 20]
    for n in (num_selected_features):
        selected = feature_importance_df.head(n)['Feature']
        trainX_selected = train_X[selected]
        testX_selected = test_X[selected]

        scaler = preprocessing.StandardScaler()
        trainX_scaled = scaler.fit_transform(trainX_selected)
        testX_scaled = scaler.transform(testX_selected)

        model = LogisticRegression()
        model.fit(trainX_scaled, train_Y)
        accuracies.append(model.score(testX_scaled, test_Y))

    plt.xlabel('Number of Selected Features')
    plt.ylabel('Accuracy')
    plt.plot(num_selected_features, accuracies)
    plt.show()


def evaluate_decision_tree(max_depth, train_X, train_Y, test_X, test_Y):
    decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
    decision_tree = decision_tree.fit(train_X, train_Y)
    train_accuracy = decision_tree.score(train_X, train_Y)
    test_accuracy = decision_tree.score(test_X, test_Y)
    print(f'Training Accuracy: {train_accuracy:.4f}\nTest Accuracy: {test_accuracy:.4f}')

    class_names = ["M", "B"]
    plt.figure(figsize=(12,12))
    tree.plot_tree(decision_tree, class_names=class_names, feature_names=feature_names[2:], filled=True, fontsize=6)
    plt.show()
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names[2:], 'Importance': decision_tree.feature_importances_})
    return feature_importance_df.sort_values(by='Importance', ascending=False)


def evaluate_random_forests(train_X, train_Y, test_X, test_Y):
    train_accuracies, test_accuracies = [], []
    num_trees = [5, 10, 20, 50, 100, 200]
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
    numpy.random.seed(42)
    train_X, test_X, train_Y, test_Y = prepare_data("./breast+cancer+wisconsin+diagnostic/wdbc.data")
    max_depth = tune_max_depth(30, train_X, train_Y)
    print("Tuned max depth:", max_depth)

    feature_importance_df = evaluate_decision_tree(max_depth, train_X, train_Y, test_X, test_Y)
    print(feature_importance_df)

    logistic_regression(train_X, train_Y, test_X, test_Y, feature_importance_df)
    evaluate_random_forests(train_X, train_Y, test_X, test_Y)