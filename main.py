import random

def temp_evaluate(n):
    return random.uniform(0, 100) # random float

def forward_selection(total_features):
    # stuff

def backward_elimination(total_features):
    # stuff

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
