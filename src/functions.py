import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

## Plot the output class percentage of instances for each datasets
def plot(dataset, filename, output_class):
    class_distribution = dataset[output_class].value_counts(normalize=True) * 100  # Convert to percentage

    axes = class_distribution.plot(kind='bar')
    class_name = output_class.capitalize()
    dataset_name = filename.capitalize()
    plt.xlabel(class_name)
    plt.ylabel('Percentage')
    plt.title(dataset_name + ' ' + class_name + ' Distribution')
    plt.xticks(rotation=0)  # Rotate x-axis labels to 0 degrees (horizontal)

    # Add percentage labels on each bar
    for i, value in enumerate(class_distribution):
        axes.text(i, value, f'{value:.2f}%', ha='center', va='bottom')

    # Customize the y-axis
    axes.yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentages
    axes.set_ylim(0, 50)  # Set the maximum value to 50%

    # Save the plot
    plt.savefig('images/' + filename + '-classes.png')
    plt.clf()

## Function to define the Base-DT model
def base_dt(file_path, image_path, abalone, X_train, X_test, y_train, y_test, iteration):
    # Initialize Decision Tree
    dt = DecisionTreeClassifier()
    # Train decision tree
    dt.fit(X_train, y_train)

    # Visualize the decision tree
    if iteration == 0:
        plt.figure(figsize=(20, 10))
        if abalone:
            plot_tree(dt, feature_names=X_train.columns, class_names=list(map(str, dt.classes_)), filled=True, rounded=True, max_depth=4, fontsize=6)
        else:
            plot_tree(dt, feature_names=X_train.columns, class_names=list(map(str, dt.classes_)), filled=True, rounded=True)
        plt.savefig(image_path, format='png')
        plt.show()

    # Test the decision tree
    y_pred = dt.predict(X_test)

    # Returns the average and variance metrics for this model
    return evaluate(y_test, y_pred, dt, iteration, file_path, 'Base-DT', dt)

## Function to define the Top-DT model
def top_dt(file_path, image_path, abalone, X_train, X_test, y_train, y_test, iteration):
    # Initialize the parameters for this model
    top_dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],  
        'min_samples_split': [2, 4, 6]
    }
    # Initialize Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    # Initialize Grid Search and train it
    top_dt_grid = GridSearchCV(dt, top_dt_params, cv=5, scoring='accuracy', n_jobs=-1)
    top_dt_grid.fit(X_train, y_train)

    # Get the model with the best hyperparameters and train it
    best_dt = top_dt_grid.best_estimator_
    best_dt.fit(X_train, y_train)

    # Visualize the decision tree
    if iteration == 0:
        plt.figure(figsize=(20, 10))  # Set to a large figure size to make the tree readable
        if abalone:
            plot_tree(best_dt, filled=True, feature_names=X_train.columns, class_names=True, max_depth=5, fontsize=6)
        else:
            plot_tree(best_dt, filled=True, feature_names=X_train.columns, class_names=True)
        plt.title('Best Decision Tree found with GridSearch')
        plt.savefig(image_path, format='png')  # Save the figure to a file
        plt.show()

    # Test the decision tree
    y_pred = best_dt.predict(X_test)

    # Returns the average and variance metrics for this model
    return evaluate(y_test, y_pred, dt, iteration, file_path, 'Top-DT', best_dt)

## Function to define the Base-MLP model
def base_mlp(file_path, X_train, X_test, y_train, y_test, iteration):
    # Initialize MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        activation='logistic',
        solver='sgd',
        random_state=0,
        max_iter=10000
    )

    # Train the classifier
    mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test)

    # Returns the average and variance metrics for this model
    return evaluate(y_test, y_pred, mlp, iteration, file_path, 'Base-MLP', mlp)

## Function to define the Top-MLP model
def top_mlp(file_path, X_train, X_test, y_train, y_test, iteration):
    # Initialize the parameters for this model
    top_mlp_params = {
        'hidden_layer_sizes': [(500, 500), (30, 30, 30)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'max_iter': [500]
    }
    # Initialize MLP
    mlp = MLPClassifier(random_state=0)
    # Initialize Grid Search and train it
    top_mlp_grid = GridSearchCV(mlp, top_mlp_params, cv=5)
    top_mlp_grid.fit(X_train, y_train)

    # Get the model with the best hyperparameters and train it
    best_mlp = top_mlp_grid.best_estimator_
    best_mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_mlp.predict(X_test)

    # Returns the average and variance metrics for this model
    return evaluate(y_test, y_pred, mlp, iteration, file_path, 'Top-MLP', best_mlp)

## This function evaluates the performace of the classification of a dataset
def evaluate(y_test, y_pred, model, iteration, file_path, name, model_used):
    # Array to store performance metrics
    accuracy_results = []
    macro_f1_results = []
    weighted_f1_results = []
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Classification report
    class_report = classification_report(y_test, y_pred, zero_division=0)
    
    # Evaluate classifier
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Store performance in a dictionary
    performance =  {
        'model': model,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

    # Add the performance metric of the current iteration
    accuracy_results.append(performance['accuracy'])
    macro_f1_results.append(performance['macro_f1'])
    weighted_f1_results.append(performance['weighted_f1'])

    # Only write the performance of the first iteration
    if iteration == 0:
        save_performance_to_file(file_path, name, model_used.get_params(), performance)

    # Return the arrays of performance
    return accuracy_results, macro_f1_results, weighted_f1_results
    
# Save the performance of a single iteration of a dataset to a file
def save_performance_to_file(file_path, model_name, hyperparameters, performance_metrics):
    with open(file_path, 'a') as file:
        # Model name and Hyperparameters
        file.write(f"(A) {model_name} Model\n")
        file.write("(A) Hyperparameters:\n")
        if model_name == 'Base-MLP' or model_name == 'Top-MLP':
            desired_params = ['activation', 'hidden_layer_sizes', 'learning_rate', 'max_iter', 'solver', 'random_state']
        else:
            desired_params = ['criterion', 'max_depth', 'min_samples_split', 'random_state']
        for param in desired_params:
                file.write(f"\t{param}: {hyperparameters[param]}\n")
        
        
        # Confusion Matrix
        file.write("\n(B) Confusion Matrix:\n")
        file.write(str(performance_metrics['confusion_matrix']) + "\n\n")

        # Precision, Recall, F1-measure for each class
        file.write("(C) Classification Report:\n")
        file.write(str(performance_metrics['classification_report']) + "\n\n")
        
        # Accuracy, Macro-average F1, Weighted-average F1
        file.write(f"(D) Accuracy: {performance_metrics['accuracy']}\n")
        file.write(f"(D) Macro-average F1: {performance_metrics['macro_f1']}\n")
        file.write(f"(D) Weighted-average F1: {performance_metrics['weighted_f1']}\n")

        # Separator
        file.write("\n" + "*" * 50 + "\n\n")

# Save the average and variance performance of all iteration of a dataset to a file
def save_variance_to_file(file_path, model_name, accuracy_results, macro_f1_results, weighted_f1_results):
    with open(file_path, 'a') as file:
        # Average accuracy and variance
        file.write(f"{model_name} Model\n")
        file.write(f"Average Accuracy: {np.mean(accuracy_results)}\n")
        file.write(f"Accuracy Variance: {np.var(accuracy_results)}\n\n")

        # Average macro-average f1 and variance
        file.write(f"Average Macro-average F1: {np.mean(macro_f1_results)}\n")
        file.write(f"Macro-average F1 Variance: {np.var(macro_f1_results)}\n\n")

        # Average weighted-average f1 and variance
        file.write(f"Average Weighted-average F1: {np.mean(weighted_f1_results)}\n")
        file.write(f"Weighted-average F1 Variance: {np.var(weighted_f1_results)}\n\n")

        # Separator
        file.write("\n" + "*" * 50 + "\n\n")