
import numpy as np #Scientific computing
import matplotlib.pyplot as plt #Plotting library

np.random.seed(0) # for reproducibility
#Dog : higher ear flappiness index, lower whisker length
dog_ear_flappiness = np.random.normal(0.7, 0.1, 10)
dog_whisker_length = np.random.normal(0.5, 0.1, 10)
#Cat : lower ear flappiness index, higher whisker length
cat_ear_flappiness = np.random.normal(0.3, 0.1, 10)
cat_whisker_length = np.random.normal(0.7, 0.1, 10)
dog_whisker_length
#plot the data points
# plt.scatter(dog_ear_flappiness, dog_whisker_length, color='blue', label='Dog')
# plt.scatter(cat_ear_flappiness, cat_whisker_length, color='red', label='Cat')
# plt.title('Dog and Cat Classification')
# plt.xlabel('Ear Flappiness Index')
# plt.ylabel('Whisker Length')
# plt.legend()
# plt.show()




#Implementing random linear classifier
def random_linear_classifier(data_dogs, data_cats, k, d):
    best_error = float('inf')
    best_theta = None
    best_theta0 = None
    
    for _ in range(k):
        theta = np.random.normal(size = d)
        theta0 = np.random.normal()
        
        error = compute_error(data_dogs, data_cats, theta, theta0)
        
        if error < best_error:
            best_error = error
            best_theta = theta
            best_theta0 = theta0
            
    return best_theta, best_theta0, best_error  
        

#Compute the error of the classifier
def compute_error(data_dogs, data_cats, theta, theta0):
    error = 0
    for dog in data_dogs:
        if np.dot(theta, dog) + theta0 > 0:
            error += 1
    for cat in data_cats:   
        if np.dot(theta, cat) + theta0 < 0:
            error += 1
    return error
#Prepare the data
data_dogs = np.vstack([dog_ear_flappiness, dog_whisker_length]).T
data_cats = np.vstack([cat_ear_flappiness, cat_whisker_length]).T
data_dogs
# Ensure the function is defined and run the random linear classifier
best_theta, best_theta0, _ = random_linear_classifier(data_dogs, data_cats, 1000, 2)

print(best_theta)
print(best_theta0)

#combine the data
data = np.vstack([data_dogs, data_cats])
labels = np.hstack((np.zeros(len(data_dogs)), np.ones(len(data_cats))))

print(data)
print(labels)

#spliting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train)

#Plot the training data and testinig data points
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Dog')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Cat')
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='cyan', label='Dog Test')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='magenta', label='Cat Test')
plt.title('Dog and Cat Classification')
plt.xlabel('Ear Flappiness Index')
plt.ylabel('Whisker Length')
plt.legend()
plt.show()

#Random linear classifier for training data
k = 100
d = 2
best_theta_train, best_theta0_train, train_error = random_linear_classifier(X_train[y_train == 0], X_train[y_train == 1], k, d)

#Plot the decision boundary
x_vals_train = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
y_vals_train = (-best_theta_train[0] / best_theta_train[1]) * x_vals_train - (best_theta0_train / best_theta_train[1])

# Plot the training data, testing data, and decision boundary
plt.figure(figsize=(10, 6))

# Training data
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Dog (Train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Cat (Train)')

# Testing data
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='cyan', label='Dog (Test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='magenta', label='Cat (Test)')

# Decision boundary
plt.plot(x_vals_train, y_vals_train, color='black', label='Decision Boundary')

# Plot settings
plt.title('Dog and Cat Classification with Decision Boundary')
plt.xlabel('Ear Flappiness Index')
plt.ylabel('Whisker Length')
plt.legend()
plt.grid(True)
plt.show()

#compute training error
train_error = compute_error(X_train[y_train == 0], X_train[y_train == 1], best_theta_train, best_theta0_train)
print(f'Training Error: {train_error}')
#compute testing error
test_error = compute_error(X_test[y_test == 0], X_test[y_test == 1], best_theta_train, best_theta0_train)
print(f'Testing Error: {test_error}')
#compute overall error
overall_error = compute_error(data_dogs, data_cats, best_theta_train, best_theta0_train)
print(f'Overall Error: {overall_error}')
#compute accuracy
accuracy = 1 - (overall_error / len(data))
print(f'Accuracy: {accuracy * 100:.2f}%')
#compute precision
precision = np.sum((np.dot(data, best_theta_train) + best_theta0_train) > 0) / len(data)
print(f'Precision: {precision * 100:.2f}%')


#Define function for k-fold cross-validation
def cross_validate(data_dogs, data_cats, k_values, d, n_splits = 5):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    
    for k in k_values:
        errors = []
        for train_index, test_index in kf.split(data_dogs):
            train_dogs, test_dogs = data_dogs[train_index], data_dogs[test_index]
            train_cats, test_cats = data_cats[train_index], data_cats[test_index]
            
            best_theta, best_theta0, error = random_linear_classifier(train_dogs, train_cats, k, d)
            errors.append(error)
        
        results[k] = np.mean(errors)
    
    return results

# Define k values to test
k_values = [10, 50, 100, 200, 500, 1000]
# Perform k-fold cross-validation
cv_results = cross_validate(data_dogs, data_cats, k_values, d)
print(cv_results)
# Plot the k-fold cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, list(cv_results.values()), marker='o')
plt.xscale('log')
plt.title('K-fold Cross-Validation Results')
plt.xlabel('Number of Random Samples (k)')
plt.ylabel('Mean Error')
plt.grid(True)
plt.xticks(k_values, k_values)
plt.show()


