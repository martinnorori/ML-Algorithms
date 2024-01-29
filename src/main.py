import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from functions import *
from sklearn.model_selection import train_test_split

# Number of looping iterations
num_iterations = 5

## Extract data from datasets: penguins and abalone
penguins = pd.read_csv('datasets/penguins.csv')
abalone = pd.read_csv('datasets/abalone.csv')

## Turning Data to numerical
# One-hot encoding way
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_endoded = onehot_encoder.fit_transform(penguins[['island', 'sex']])
onehot_penguins = pd.DataFrame(onehot_endoded, columns=onehot_encoder.get_feature_names_out(['island', 'sex']))
penguins = pd.concat([penguins.drop(['island', 'sex'], axis=1), onehot_penguins], axis=1)

# Our own way
# penguins['island'] = penguins['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
# penguins['sex'] = penguins['sex'].map({'MALE': 0, 'FEMALE': 1})

## Extract features and target and split dataset for penguins 
X_penguin = penguins.drop('species', axis=1) # Features
y_penguin = penguins['species'] # Target
X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin, test_size=0.3, random_state=42)

## Extract features and target and split dataset for abalone 
X_abalone = abalone.drop('Type', axis=1) # Features
y_abalone = abalone['Type'] # Target
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone, test_size=0.2, random_state=42)

## Plot the output class percentage of instances for each datasets
plot(penguins, 'penguin', 'species')
plot(abalone, 'abalone', 'Type')

## Train and test each classifiers on each datasets
# Base-DT
# Train and test the model 5 times
for iteration in range(num_iterations):
    # Penguin dataset
    # accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = base_dt(
    #     'performance/penguin-performance.txt', 
    #     'images/base_dt_penguin.png', 
    #     False,
    #     X_train_penguin, 
    #     X_test_penguin, 
    #     y_train_penguin, 
    #     y_test_penguin, 
    #     iteration
    # )
    # Abalone dataset
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_dt(
        'performance/abalone-performance.txt', 
        'images/base_dt_abalone.png', 
        True,
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )
# Save Average and Variance metrics for penguin dataset
# save_variance_to_file(
#     'performance/penguin-performance.txt', 
#     'Base-DT', 
#     accuracy_penguin, 
#     macro_f1_penguin, 
#     weighted_f1_penguin
# )
# Save Average and Variance metrics for abalone dataset
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Base-DT', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Top-DT
# Train and test the model 5 times
for iteration in range(num_iterations):
    # Penguin dataset
    # accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = top_dt(
    #     'performance/penguin-performance.txt', 
    #     'images/top_dt_penguin.png', 
    #     False,
    #     X_train_penguin, 
    #     X_test_penguin, 
    #     y_train_penguin, 
    #     y_test_penguin, 
    #     iteration
    # )
    # Abalone dataset
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_dt(
        'performance/abalone-performance.txt', 
        'images/top_dt_abalone.png',
        True,
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )
# Save Average and Variance metrics for penguin dataset
# save_variance_to_file(
#     'performance/penguin-performance.txt', 
#     'Top-DT', 
#     accuracy_penguin, 
#     macro_f1_penguin, 
#     weighted_f1_penguin
# )
# Save Average and Variance metrics for abalone dataset
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Top-DT', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Base-MLP
# Train and test the model 5 times
for iteration in range(num_iterations):
    # Penguin dataset
    # accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = base_mlp(
    #     'performance/penguin-performance.txt', 
    #     X_train_penguin, 
    #     X_test_penguin, 
    #     y_train_penguin, 
    #     y_test_penguin, 
    #     iteration
    # )
    # Abalone dataset
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_mlp(
        'performance/abalone-performance.txt', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )
# Save Average and Variance metrics for penguin dataset
# save_variance_to_file(
#     'performance/penguin-performance.txt', 
#     'Base-MLP', 
#     accuracy_penguin, 
#     macro_f1_penguin, 
#     weighted_f1_penguin
# )
# Save Average and Variance metrics for abalone dataset
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Base-MLP', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Top-MLP
# Train and test the model 5 times
for iteration in range(num_iterations):
    # Penguin dataset
    # accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = top_mlp(
    #     'performance/penguin-performance.txt', 
    #     X_train_penguin, 
    #     X_test_penguin, 
    #     y_train_penguin, 
    #     y_test_penguin, 
    #     iteration
    # )
    # Abalone dataset
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = top_mlp(
        'performance/abalone-performance.txt', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )
# Save Average and Variance metrics for penguin dataset
# save_variance_to_file(
#     'performance/penguin-performance.txt', 
#     'Top-MLP', 
#     accuracy_penguin, 
#     macro_f1_penguin, 
#     weighted_f1_penguin
# )
# Save Average and Variance metrics for abalone dataset
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Top-MLP', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)
