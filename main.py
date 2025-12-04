import time
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# This is based off index 1

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

def loocv_accuracy_with_all_features(dataset):
    """
    Convenience helper: given a dataset of [class, f1, f2, ...],
    compute LOOCV 1-NN accuracy using ALL features.
    """
    y = [row[0] for row in dataset]
    X = [row[1:] for row in dataset]
    num_features = len(X[0])
    feature_subset = list(range(num_features))  # indices 0..d-1 for X

    classifier = NearestNeighbor()
    cv = CrossValidation(classifier, X, y, feature_subset)
    return cv.leave_one_out_validation()

def evaluate_normalization_effect(small_raw, small_norm, large_raw, large_norm, titanic_raw, titanic_norm):
    """
    Prints a small table of LOOCV 1-NN accuracy with and without normalization.
    Uses all features for each dataset.
    """

    small_raw_acc  = loocv_accuracy_with_all_features(small_raw)
    small_norm_acc = loocv_accuracy_with_all_features(small_norm)
    large_raw_acc  = loocv_accuracy_with_all_features(large_raw)
    large_norm_acc = loocv_accuracy_with_all_features(large_norm)
    titanic_raw_acc  = loocv_accuracy_with_all_features(titanic_raw)
    titanic_norm_acc = loocv_accuracy_with_all_features(titanic_norm)   

    print("\n=== Effect of Normalization (1-NN with all features, LOOCV) ===")
    print(f"{'Dataset':<15} {'Normalized?':<12} {'Accuracy (%)':>12}")
    print("-" * 43)
    print(f"{'Small':<15} {'No':<12} {small_raw_acc:>12.1f}")
    print(f"{'Small':<15} {'Yes':<12} {small_norm_acc:>12.1f}")
    print(f"{'Large':<15} {'No':<12} {large_raw_acc:>12.1f}")
    print(f"{'Large':<15} {'Yes':<12} {large_norm_acc:>12.1f}")
    print(f"{'Titanic':<15} {'No':<12} {titanic_raw_acc:>12.1f}")
    print(f"{'Titanic':<15} {'Yes':<12} {titanic_norm_acc:>12.1f}")
    print("=============================================================\n")
    

    return small_raw_acc, small_norm_acc, large_raw_acc, large_norm_acc, titanic_raw_acc, titanic_norm_acc

def Franklyn_Algorithm(total_features):
    print("\nWelcome to Franklyn's SPEED-BOOSTED Feature Selection Algorithm")

    accuracy_cache = {}  # memoization cache

    def cached_eval(subset):
        key = tuple(sorted(subset))
        if key in accuracy_cache:
            return accuracy_cache[key]
        acc = evaluate_subset(subset)
        accuracy_cache[key] = acc
        return acc

    total_start = time.time()

    current = []
    best_overall_set = []
    best_overall_acc = cached_eval([])

    print(f"Using no features, accuracy = {best_overall_acc:.2f}%")
    print("Beginning accelerated search...\n")

    # skip features threshold
    SKIP_FACTOR = 0.90 # skip features 10% worse than current best

    for level in range(total_features):
        level_start = time.time()

        print(f"--> Level {level+1}")

        best_feature = None
        best_acc = -1

        skip_threshold = best_overall_acc * SKIP_FACTOR

        for f in range(total_features):
            if f in current:
                continue

            temp = current + [f]
            step_start = time.time()

            acc = cached_eval(temp)
            step_time = time.time() - step_start

            print(f"Using feature(s) {[x+1 for x in temp]}, "
                  f"acc = {acc:.2f}%  (Time: {step_time:.4f}s)")

            # SKIP BAD FEATURES
            if acc < skip_threshold:
                print(f"   → Skipped (below skip-threshold {skip_threshold:.2f}%)")
                continue

            if acc > best_acc:
                best_acc = acc
                best_feature = f

        if best_feature is None:
            print("No acceptable feature found, stopping early.")
            break

        current.append(best_feature)
        level_time = time.time() - level_start

        print(f"\nAdded feature {best_feature+1}")
        print(f"Current set: {[x+1 for x in current]}, accuracy = {best_acc:.2f}%")
        print(f"Level time: {level_time:.4f}s\n")

        # update global best
        if best_acc > best_overall_acc:
            best_overall_acc = best_acc
            best_overall_set = list(current)

        # stops early if gain is very little
        if abs(best_acc - best_overall_acc) < 0.001:
            print("Accuracy gain too small → early stopping.")
            break

    total_time = time.time() - total_start

    print("Finished search!")
    print(f"Best feature subset = {[x+1 for x in best_overall_set]}")
    print(f"Best accuracy = {best_overall_acc:.2f}%")
    print(f"Total runtime = {total_time:.4f}s")

    return best_overall_set, best_overall_acc


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
                print(f"Using feature(s) {[x+1 for x in temp_features]} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")
                if accuracy > best_accuracy_so_far: # update accuracy
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = f

        if feature_to_add_at_this_level is None:
            print("No feature improved accuracy at this level stopping search.")
            break
            

        current_features.append(feature_to_add_at_this_level)
        level_time = time.time() - level_start
        print()
        print(f"Added feature {feature_to_add_at_this_level + 1} at this level.")
        print(f"Feature set {[x+1 for x in current_features]} was best, accuracy is {best_accuracy_so_far:.1f}% (Step Time: {level_time:.4f} sec)")
        
        # compares if accuracy increased or decreased
        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) # new best pathway in a new list
            print()

    total_time = time.time() - total_start

    print(f"Finished search!! The best feature subset is {[f+1 for f in best_overall_set]}, "
      f"which has an accuracy of {best_overall_accuracy:.1f}% (Total time: {total_time:.4f} sec)")

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
            print(f"Using feature(s) {[x+1 for x in temp_features]} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = f

        if feature_to_remove_at_this_level is not None:
            current_features.remove(feature_to_remove_at_this_level)

        level_time = time.time() - level_start
        print()
        print(f"Removed feature {feature_to_remove_at_this_level + 1} at this level.")
        print(f"Feature set {[x+1 for x in current_features]} was best, accuracy is {best_accuracy_so_far:.1f}% (Step Time: {level_time:.4f} sec)")

        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) 
            print()
    
    total_time = time.time() - total_start
    print(f"Finished search!! The best feature subset is {[f+1 for f in best_overall_set]}, "
      f"which has an accuracy of {best_overall_accuracy:.1f}% (Total time: {total_time:.4f} sec)")
    
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

def normalize_dataset(data):
   
    if not data:
        return []

    n = len(data)
    d = len(data[0])

    # compute mean/std for each feature column (1..d-1)
    means = []
    stds = []
    for j in range(1, d):
        col = [data[i][j] for i in range(n)]
        mean = sum(col) / n
        var = sum((x - mean) ** 2 for x in col) / n
        std = math.sqrt(var)
        if std == 0:
            std = 1.0
        means.append(mean)
        stds.append(std)

    # build normalized copy
    norm = []
    for i in range(n):
        row = [data[i][0]]  # class label
        for j in range(1, d):
            m = means[j - 1]
            s = stds[j - 1]
            row.append((data[i][j] - m) / s)
        norm.append(row)

    return norm


#plotting features
def plot_feature_pair(dataset, feat_x, feat_y,
                      dataset_label="Dataset",
                      pair_label="",
                      save_name=None,
                      show=True):
  

    data = np.array(dataset)
    labels = data[:, 0]
    x = data[:, 1 + feat_x]
    y = data[:, 1 + feat_y]

    fig, ax = plt.subplots(figsize=(6, 6))
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(x[mask], y[mask], label=f"Class {int(cls)}", marker="o")

  
    ax.set_xlabel(f"Feature {feat_x}")
    ax.set_ylabel(f"Feature {feat_y}")

    title = dataset_label
    if pair_label:
        title += f" ({pair_label})"
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()

    if save_name is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, save_name)
        fig.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    small_data = load_dataset("small-test-dataset-2-2.txt")
    large_data = load_dataset("large-test-dataset-2.txt")
    titanic_data = load_dataset("titanic clean-2.txt")
    small_norm = normalize_dataset(small_data) 
    large_norm = normalize_dataset(large_data)
    titanic_norm = normalize_dataset(titanic_data)

    # sr, sn, lr, ln,tr,tn = evaluate_normalization_effect(
    #     small_data, small_norm, large_data, large_norm, titanic_data, titanic_norm
    # )

    total_features_small = len(small_norm[0]) - 1 # 100 instances and 10 features
    total_features_large = len(large_norm[0]) - 1 # 1000 instances, and 40 features
    total_features_titanic = len(titanic_norm[0]) - 1

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Franklyn's Algorithm")
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
    
  
    labels = ["Small (raw)", "Small (norm)", "Large (raw)", "Large (norm)", "Titanic (raw)", "Titanic (norm)"]
    accs = [sr, sn, lr, ln, tr, tn]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(labels)), accs)
    plt.xticks(range(len(labels)), labels, rotation=20)
    plt.ylabel("Accuracy (%)")
    plt.title("Effect of Normalization on 1-NN Accuracy (All Features)")
    plt.tight_layout()
    plt.savefig("normalization_effect.png", dpi=300)
    plt.show()
    plot_feature_pair(large_norm,
                  feat_x=15,  # feature 3 (0-based index)
                  feat_y=12,  # feature 4
                  dataset_label="large Dataset",
                  pair_label="bad pair (15,12)",
                  save_name="large_dataset_feature3_4.png")

    if choice == 1:
        forward_selection(total_features)
    elif choice == 2:
        backward_elimination(total_features)
    elif choice == 3:
        Franklyn_Algorithm(total_features)
    else:
        print("Ivalid choice")
        return

main()
