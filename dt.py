from random import seed
from random import randrange
from csv import reader
import math


METHOD_GINI = 1
METHOD_ENTROPY = 2

# Load CSV & deleting first column
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader((line.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                              for line in file))
    dataset = list(lines)
    for idx_row in range(len(dataset)):
        row = dataset[idx_row]
        del (row[0])
        for idx_col in range(len(row)):
            try:
                row[idx_col] = float(row[idx_col].strip())
            except:
                pass
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    x = True
    for fold in folds:
        #if !x: continue
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        x = False
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate gini split
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini_split = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini_split += (1.0 - score) * (size / n_instances)
    return gini_split


# Calculate gain split
def entropy(groups, classes, base):
    n_instances = float(sum([len(group) for group in groups]))
    entropy_parent = 0.0
    classes_size = dict()
    for group in groups:
        for class_val in classes:
            size = [row[-1] for row in group].count(class_val)
            if class_val in classes_size:
                classes_size[class_val] += size
            else:
                classes_size[class_val] = size
    for class_val, size in classes_size.items():
        if size != 0:
            entropy_parent -= math.log(size/n_instances, base) * size/n_instances

    gain_split = entropy_parent
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            class_size = [row[-1] for row in group].count(class_val)
            if class_size == 0:
                score = 0
                break
            e = class_size / size
            score -= math.log(e, base) * e
        gain_split -= score * (size / n_instances)
    return gain_split


# Select the best split
def get_split(dataset, method = METHOD_GINI, base = 2):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999 if method == METHOD_GINI else -1, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            if method == METHOD_ENTROPY:
                gain_split = entropy(groups, class_values, base)
                if gain_split > b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gain_split, groups
            else:
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create leaf
def set_leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child/leaf
def split(node, method, base, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = set_leaf(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = set_leaf(left), set_leaf(right)
        return
    # left child
    if len(left) <= min_size:
        node['left'] = set_leaf(left)
    else:
        node['left'] = get_split(left, method, base)
        split(node['left'], method, base, max_depth, min_size, depth + 1)
    # right child
    if len(right) <= min_size:
        node['right'] = set_leaf(right)
    else:
        node['right'] = get_split(right, method, base)
        split(node['right'], method, base, max_depth, min_size, depth + 1)


def build_tree(train, method, base, max_depth, min_size):
    root = get_split(train, method, base)
    split(root, method, base, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, method, base, max_depth, min_size):
    tree = build_tree(train, method, base, max_depth, min_size)
    print(tree)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


if __name__ == '__main__':
    seed(1)
    filename = 'yeast.data'
    dataset = load_csv(filename)
    # evaluate algorithm
    n_folds = 5
    max_depth = 8
    min_size = 10
    # METHOD_ENTROPY or METHOD_GINI
    method = METHOD_ENTROPY
    # log base for entropy method
    base = 2
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, method, base, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
