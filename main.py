import time
import math

class NearestNeighbor:
    def __init__(self):
        self.training = [] # memorization

    def train(self, training_instances): # store training data
        self.training = training_instances # each training instance: [class_label, feature1, feature2, ..., feature n]

    def test(self, test_instance): # prediction
        best_distance = float("inf")
        best_class = None

        for train in self.training: # what we do in each training
            # train[0] = class, train[1:] = features
            d = euclidean_distance(train[1:], test_instance[1:])
            if d < best_distance: # update distance
                best_distance = d
                best_class = train[0]

        return best_class
    
def euclidean_distance(point1, point2):
    squared_differences = []
    for i in range(len(point1)):
        difference = point1[i] - point2[i]
        squared_difference = difference ** 2
        squared_differences.append(squared_difference)
    sum_of_squares = sum(squared_differences)
    distance = math.sqrt(sum_of_squares)
    return distance

"""
Remove it from the dataset (this is the test instance)

Train NN on all remaining instances.

Filter both the training and test instances so they contain only the selected features.

Classify the test instance.

Count how many predictions are correct.
"""

class CrossValidation:
    def __init__(self, classifier, X, y, feature_subset):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.feature_subset = feature_subset

    def leave_one_out_validation(self):
        correct = 0
        n = len(self.X)

        for i in range(n):

            #start = time.time()

            # leave-one-out split
            train_X = [row[:] for j, row in enumerate(self.X) if j != i]
            train_y = [label for j, label in enumerate(self.y) if j != i]

            # build test row with selected features
            test_x = [self.y[i]] + [self.X[i][f] for f in self.feature_subset]

            # filter training rows to selected features
            filtered_train_X = [
                [train_y[k]] + [row[f] for f in self.feature_subset]
                for k, row in enumerate(train_X)
            ]

            # train the classifier
            self.classifier.train(filtered_train_X)

            # predict
            predicted = self.classifier.test(test_x)

            if predicted == self.y[i]:
                correct += 1

            #end = time.time()

            #print(f"Step {i}: true={self.y[i]}, predicted={predicted}, "
                  #f"time={(end - start)*1000:.3f} ms")

        return (correct / n) * 100

def evaluate_subset(feature_subset):

    # split dataset into X and y
    y = [row[0] for row in global_dataset]
    X = [row[1:] for row in global_dataset]

    classifier = NearestNeighbor()
    cv = CrossValidation(classifier, X, y, feature_subset)
    acc = cv.leave_one_out_validation()

    print(f"Accuracy using features {[f+1 for f in feature_subset]}: {acc:.4f}")   
    return acc

def forward_selection(total_features):
    print()
    print("Welcome to Franklyn and Mani's Feature Selection Algorithm")
    total_start = time.time()
    starting_accuracy = evaluate_subset([])
    print(f"Using no features, I get an accuracy of {starting_accuracy:.1f}%")
    print("Beginning search.")
    print()
    
    current_features = []
    best_overall_accuracy = starting_accuracy
    best_overall_set = []

    for level in range(total_features):
        level_start = time.time()
        feature_to_add_at_this_level = None
        best_accuracy_so_far = -1.0
        print(f"--> Level {level + 1} of search")

        for f in range(total_features):
            if f not in current_features:
                step_start = time.time()
                temp_features = current_features + [f] # setup feature ID to print (note: feature # starts at 0)
                accuracy = evaluate_subset(temp_features)
                step_time = time.time() - step_start
                print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")
                if accuracy > best_accuracy_so_far: # update accuracy
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = f

        if feature_to_add_at_this_level is None:
            print("No feature improved accuracy at this level stopping search.")
            break
            

        current_features.append(feature_to_add_at_this_level)
        level_time = time.time() - level_start
        print()
        print(f"Added feature {feature_to_add_at_this_level} at this level.")
        print(f"Feature set {current_features} was best, accuracy is {best_accuracy_so_far:.1f}% (Step Time: {level_time:.4f} sec)")
        
        # compares if accuracy increased or decreased
        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) # new best pathway in a new list
            print()

    total_time = time.time() - total_start

    print(f"Finished search!! The best feature subset is {best_overall_set}, which has an accuracy of {best_overall_accuracy:.1f}% (Total time: {total_time:.4f} sec)")

def backward_elimination(total_features):
    print()
    print("Welcome to Franklyn and Mani's Feature Selection Algorithm")
    total_start = time.time()
    current_features = list(range(total_features))
    starting_accuracy = evaluate_subset(current_features)
    print(f"Using all features, I get an accuracy of {starting_accuracy:.1f}%")
    print("Beginning search.")
    print()

    best_overall_accuracy = starting_accuracy
    best_overall_set = list(current_features)


    for level in range(total_features - 1):
        level_start = time.time()
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0.0
        print(f"--> Level {level + 1} of search")


        for f in current_features:
            step_start = time.time()
            temp_features = [feat for feat in current_features if feat != f]
            accuracy = evaluate_subset(temp_features)
            step_time = time.time() - step_start
            print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = f

        if feature_to_remove_at_this_level is not None:
            current_features.remove(feature_to_remove_at_this_level)

        level_time = time.time() - level_start
        print()
        print(f"Removed feature {feature_to_remove_at_this_level} at this level.")
        print(f"Feature set {current_features} was best, accuracy is {best_accuracy_so_far:.1f}% (Step Time: {level_time:.4f} sec)")

        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) 
            print()
    
    total_time = time.time() - total_start
    print(f"Finished search!! The best feature subset is {best_overall_set}, which has an accuracy of {best_overall_accuracy:.1f}% (Total time: {total_time:.4f} sec)")

def load_dataset(path): # [class] [feature1] [feature2] [feature3] ... [featureN] (PER FEATURE)
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            class_value = int(float(parts[0])) # class # is int
            feature_values = []
            for x in parts[1:]:
                feature_values.append(float(x)) # convert all item in parts to float, then putting it into a list
            row = [class_value] + feature_values
            data.append(row)
    return data

def normalize_dataset(X):
    if not X:
        return
    n = len(X)
    d = len(X[0])  # total columns (class + features)
    for j in range(1, d):
        mean = sum(X[i][j] for i in range(n)) / n
        var = sum((X[i][j] - mean) ** 2 for i in range(n)) / n
        stddev = math.sqrt(var)
        if stddev == 0:
            stddev = 1.0
        for i in range(n):
            X[i][j] = (X[i][j] - mean) / stddev
    return X

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_pair(dataset, f1, f2):
    data = np.array(dataset)
    labels = data[:, 0]
    x = data[:, 1 + f1]
    y = data[:, 1 + f2]

    plt.figure(figsize=(6, 6))
    for cls in np.unique(labels):
        mask = labels == cls
        plt.scatter(x[mask], y[mask], label=f"Class {int(cls)}", marker='o')

    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.title(f"Feature {f1} vs Feature {f2}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    small_data = load_dataset("small-test-dataset-2-2.txt")
    large_data = load_dataset("large-test-dataset-2.txt")
    titanic_data = load_dataset("titanic clean-2.txt")
    small_norm = normalize_dataset(small_data) 
    large_norm = normalize_dataset(large_data)
    titanic_norm = normalize_dataset(titanic_data)
    total_features_small = len(small_norm[0]) - 1 # 100 instances and 10 features
    total_features_large = len(large_norm[0]) - 1 # 1000 instances, and 40 features
    total_features_titanic = len(titanic_norm[0]) - 1

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = int(input())
    print("Please enter dataset type.")
    print("1) Small Dataset")
    print("2) Large Dataset")
    print("3) Titanic Dataset")
    choice2 = int(input())

    global global_dataset
    if choice2 == 1:
        total_features = total_features_small
        global_dataset = small_norm
    elif choice2 == 2:
        total_features = total_features_large
        global_dataset = large_norm
    elif choice2 == 3:
        total_features = total_features_titanic
        global_dataset = titanic_norm
    else:
        print("Ivalid choice")
        return
    
    print()
    print(f"This dataset has {total_features} features")
    print(f"This dataset has {len(global_dataset)} instances")

    if choice == 1:
        forward_selection(total_features)
    elif choice == 2:
        backward_elimination(total_features)
    else:
        print("Ivalid choice")
        return
    
    #plot_feature_pair(global_dataset, 2, 4)

    # print("Enter feature numbers separated by spaces (e.g., 3 5 7):")
    # user_input = input().strip()
    # feature_subset_original = list(map(int, user_input.split()))
    # feature_subset = [f - 1 for f in feature_subset_original]
    # evaluate_subset(global_dataset, feature_subset)

main()
