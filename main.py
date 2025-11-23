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

def leave_one_out_validation(dataset, feature_subset): # this calls the NN class and gives accuracy using NN
    correct = 0
    n = len(dataset)

    for leave_out in range(n):
        train_set = [] 
        for i in range(n): 
            if i != leave_out: # skip the index we want to leave out
                train_set.append(dataset[i]) # add the item to train_set

        nn = NearestNeighbor() # calls the class
        nn.train(train_set)

        test_instance = dataset[leave_out]

        filtered_test = [] # filter test instance
        filtered_test.append(test_instance[0])
        for f in feature_subset:
            filtered_test.append(test_instance[1 + f])

        filtered_train = [] # filter training set
        for row in train_set: 
            filtered_row = [row[0]]
            for f in feature_subset:
                filtered_row.append(row[1 + f])
            filtered_train.append(filtered_row)

        nn.training = filtered_train # replace training data with filtered version
        pred_class = nn.test(filtered_test) # predict class
        if pred_class == test_instance[0]: # prediction correct?
            correct += 1
    return (correct / n) * 100 # accuracy computation (from 0-100)

def temp_evaluate(n): # our project 1 function, it will be used to call the real evaulation function
    return leave_one_out_validation(global_dataset, n) # n is our feature subset

def forward_selection(total_features):
    print()
    print("Welcome to Franklyn and Mani's Feature Selection Algorithm")
    total_start = time.time()
    starting_accuracy = temp_evaluate([])
    print(f"Using no features, I get an accuracy of {starting_accuracy:.1f}%")
    print("Beginning search.")
    print()

    current_features = []
    best_overall_accuracy = starting_accuracy
    best_overall_set = []

    for level in range(total_features):
        level_start = time.time()
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0.0

        for f in range(total_features):
            if f not in current_features:
                step_start = time.time()
                temp_features = current_features + [f] # setup feature ID to print (note: feature # starts at 0)
                accuracy = temp_evaluate(temp_features)
                step_time = time.time() - step_start
                print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")
                if accuracy > best_accuracy_so_far: # update accuracy
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = f

        current_features.append(feature_to_add_at_this_level)
        level_time = time.time() - level_start
        print()
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
    starting_accuracy = temp_evaluate(current_features)
    print(f"Using all features, I get an accuracy of {starting_accuracy:.1f}%")
    print("Beginning search.")
    print()

    best_overall_accuracy = starting_accuracy
    best_overall_set = list(current_features)


    for level in range(total_features - 1):
        level_start = time.time()
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0.0


        for f in current_features:
            step_start = time.time()
            temp_features = [feat for feat in current_features if feat != f]
            accuracy = temp_evaluate(temp_features)
            step_time = time.time() - step_start
            print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}% (Time: {step_time:.4f} sec)")

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = f

        if feature_to_remove_at_this_level is not None:
            current_features.remove(feature_to_remove_at_this_level)

        level_time = time.time() - level_start
        print()
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
            row = list(map(float, parts)) # convert all item in parts to float, then putting it into a list
            data.append(row)
    return data

def normalize_dataset(data): # z score normalization (x - mean) / std
    # transpose for easier column operations
    cols = list(zip(*data))
    class_col = cols[0]
    features = cols[1:]

    normalized = []
    means = []
    stds = []

    # compute z-score for each feature column
    for col in features:
        m = sum(col) / len(col)
        s = math.sqrt(sum((x - m)**2 for x in col) / len(col))
        means.append(m)
        stds.append(s)

    # apply normalization
    for row in data:
        cls = row[0]
        feats = row[1:]

        norm_feats = []  
        for i in range(len(feats)):
            if stds[i] != 0: 
                normalized_value = (feats[i] - means[i]) / stds[i]
            else: # std cant be 0, we just keep the original feature value
                normalized_value = feats[i]
    
            norm_feats.append(normalized_value)
        normalized.append([cls] + norm_feats) # class label contatenate with list of normalized features

    return normalized

def main():
    small_data = load_dataset("small-test-dataset-2-2.txt")
    large_data = load_dataset("large-test-dataset-2.txt")
    small_norm = normalize_dataset(small_data) 
    large_norm = normalize_dataset(large_data)
    total_features_small = len(small_norm[0]) - 1 # 100 instances and 10 features
    total_features_large = len(large_norm[0]) - 1 # 1000 instances, and 40 features

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = int(input())
    print("Please enter dataset type.")
    print("1) Small Dataset")
    print("2) Large Dataset")
    choice2 = int(input())

    global global_dataset
    if choice2 == 1:
        total_features = total_features_small
        global_dataset = small_norm
    elif choice2 == 2:
        total_features = total_features_large
        global_dataset = large_norm
    else:
        print("Ivalid choice")
        return

    if choice == 1:
        forward_selection(total_features)
    elif choice == 2:
        backward_elimination(total_features)
    else:
        print("Ivalid choice")
        return

main()
