import random

def temp_evaluate(n):
    return random.uniform(0, 100) # random float 0 to 100 (represents percentage)

def forward_selection(total_features):
    print()
    print("Welcome to Franklyn and Mani's Feature Selection Algorithm")
    print(f"Using no features and 'random' evaluation, I get an accuracy of {temp_evaluate([]):.1f}%")
    print("Beginning search.")
    print()

    current_features = []
    best_overall_accuracy = 0.0
    best_overall_set = []

    for level in range(total_features):
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0.0

        for f in range(total_features):
            if f not in current_features:
                temp_features = current_features + [f] # setup feature ID to print (note: feature # starts at 0)
                accuracy = temp_evaluate(temp_features)
                print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}%")
                if accuracy > best_accuracy_so_far: # update accuracy
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = f

        current_features.append(feature_to_add_at_this_level)
        print()
        print(f"Feature set {current_features} was best, accuracy is {best_accuracy_so_far:.1f}%")

        # compares if accuracy increased or decreased
        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) # new best pathway in a new list
            print()

    print(f"Finished search!! The best feature subset is {best_overall_set}, which has an accuracy of {best_overall_accuracy:.1f}%")

def backward_elimination(total_features):
    print()
    print("Welcome to Franklyn and Mani's Feature Selection Algorithm")

    current_features = list(range(total_features))
    starting_accuracy = temp_evaluate(current_features)
    print(f"Using all features and 'random' evaluation, I get an accuracy of {starting_accuracy:.1f}%")
    print("Beginning search.")
    print()

    best_overall_accuracy = starting_accuracy
    best_overall_set = list(current_features)


    for level in range(total_features - 1):
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0.0


        for f in current_features:
            temp_features = [feat for feat in current_features if feat != f]
            accuracy = temp_evaluate(temp_features)
            print(f"Using feature(s) {temp_features} accuracy is {accuracy:.1f}%")

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = f

        if feature_to_remove_at_this_level is not None:
            current_features.remove(feature_to_remove_at_this_level)

        print()
        print(f"Feature set {current_features} was best, accuracy is {best_accuracy_so_far:.1f}%")

        if best_accuracy_so_far < best_overall_accuracy:
            print("(Warning, Accuracy has decreased!)")
            print()

        if best_accuracy_so_far > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_so_far
            best_overall_set = list(current_features) 
            print()

    print(f"Finished search!! The best feature subset is {best_overall_set}, which has an accuracy of {best_overall_accuracy:.1f}%")


def main():
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = int(input())
    total_features = int(input("Please enter total number of features: "))

    while True:
        if choice == 1:
            forward_selection(total_features)
            break
        elif choice == 2:
            backward_elimination(total_features)
            break
        else:
            print("Enter valid choice")

main()
