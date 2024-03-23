# %% [markdown]
# # COMP2211 PA1: K-Means Clustering and K-Nearest Neighbors for Forest Cover Classification
# 
# ## Introduction
# 
# The Roosevelt National Forest of northern Colorado is characterized by clusters of varying tree species (also known as Forest Covers). It has been found that cartographic data (or the information used to draw maps) can be reliable indicators for the kinds of tree species found in specific parts of the forest. In this PA, we will use cartographic data to categorize the Forest Cover Types over parts of the Roosevelt National Forest.
# 
# ## Task Overview
# 
# We will begin by classifying forest cover types using **K-Nearest Neighbors**. Then, we will further analyze the data through **K-Means Clustering**.

# %% [markdown]
# ## Task 0: Setting up
# 
# First, we need to upload all the relevant libraries.
# 
# Note: This part will not be graded!

# %% [markdown]
# ### Task 0.1: Import libraries
# It's a good habit to import all libraries at the beginning of the code, and it helps in the following aspects:
# *   Readability and clarity
# *   Avoiding namespace clashes
# *   Dependency management
# *   Consistency and convention
# 
# **Todo:**  
# Please import your libraries in the following cell.  
# 
# **Remarks:**
# 1. We use [Numpy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) in this PA. You may also import other modules as long as they are part of the [Python Standard Library](https://docs.python.org/3/library/).  
# 2. You are NOT allowed to use any other external libraries/functions
#  (especially any machine learning library, e.g., sklearn) in todo.

# %%
# task 0.1: import libraries
# todo start #
import pandas as pd
import numpy as np
# todo end #

# %% [markdown]
# ### Task 0.2: Read Dataset
# Now you have the needed libraries in hand. Next, read the dataset from the source file to the project.  
# 
# We assume you are working in Google Colab. One way to read a dataset in Google Colab:
# 1. Download the source file and put it on your Google Drive
# 2. Import the `drive` module from `google.colab` package
# 3. Run `drive.mount` to mount your Google Drive to the Colab notebook
# 4. Use `pandas.read_csv` to read the data from Google Drive and store the data in pandas DataFrame
# 
# **Todo:** \
# Modify `YourFilePath` depending on the directory to read the data to this notebook.

# %%
# Task 0.2: read dataset
if __name__ == '__main__':
    #from google.colab import drive
    #drive.mount('/content/drive')
# todo start #
    YourFilePath = '/Users/wayd/COMP2211-PA1'
# todo end #
    train_features = pd.read_csv(YourFilePath+'/train_features.csv')
    test_features = pd.read_csv(YourFilePath+'/test_features.csv')
    train_labels = pd.read_csv(YourFilePath+'/train_labels.csv')
    test_labels = pd.read_csv(YourFilePath+'/test_labels.csv')

# %% [markdown]
# ## Dataset Description
# 
# This dataset is for the task of Forest Cover Type Classification. The 7 Forest Cover Types are represented by integers (1-7):
# 
# *   1 - Spruce/Fir
# *   2 - Lodgepole Pine
# *   3 - Ponderosa Pine
# *   4 - Cottonwood/Willow
# *   5 - Aspen
# *   6 - Douglas-fir
# *   7 - Krummholz
# 
# These integers are the labels to be predicted. The labels for the training set can be found in the data file *train_labels.csv*, and the labels for the test set can be found in the data file *test_labels.csv*
# 
# The goal is to predict the Forest Cover type using cartographic or mapping data. These can be found in the files *train_features.csv* and *test_features.csv*
# 
# You can check the features using the Pandas's *dataframe.describe()*. This will show you that the features have a very different range of values (compare Elevation with Slope). Because of this, we need to perform some data preprocessing.

# %%
if __name__ == '__main__':
  print(train_features.describe())

# %% [markdown]
# ## Task 1: Data Preprocessing
# 
# Data preprocessing ensures the fair treatment of features, efficient computation, and easier interpretability among features. For this assignment, we will use **Z-score Normalization**.
# 
# **Note:** For this assignment, you can treat all the features as numerical features.
# 
# Suppose $X:(x_1, x_2, ..., x_n)$ is a column (corresponding to a feature), then
# $\displaystyle X_{\text{Z-score-normalized}} = \frac{X-\mu_X}{\sigma_X}$
# 
# **Todo:**  
# Implement `z_score_normalization(input_array)`.  
# 
# **Suggested Numpy functions:**
# `numpy.mean`, `numpy.std` ...
# 
# 

# %%
# Task 1: Data Preprocessing
def z_score_normalization(input_array):
  # input_array: numpy array of shape (num_rows, num_features)
  # todo start #
  normalized_array = (input_array - np.mean(input_array,axis=0))/(np.std(input_array,axis=0))
  # todo end #
  return normalized_array

# %% [markdown]
# ## Task 2: Measuring "Nearness" of Neighbors
# 
# Now, we can check if it's possible to determine forest cover based on cartographical area. The forest cover serves as a label for the data point. It is possible that cartographical information may be good features for classifying forest cover types.
# 
# In **K-Nearest Neighbors**, a test sample is classified based on the "distance" of its features to the features of labelled training samples. The predicted class is based on the label of the majority of the K nearest neighbors.
# 
# Evidently, this involves calculating distance measures. Aside from Euclidean distance, here are two other distance metrics that can be used in K-Nearest Neighbors:
# 
# ### Task 2.1: Manhattan Distance
# $\displaystyle d(X, Y) = \sum_{i=1}^{n} | x_i - y_i |$
# 
# ### Task 2.2: Cosine Distance
# $\displaystyle d(X, Y) = 1 - \frac{ \sum_{i=1}^{n} x_i \times y_i }{ \sqrt{\sum_{i=1}^{n} x_i^2 } \sqrt{\sum_{i=1}^{n} y_i^2 }}$
# 
# **Todo:** \
# Implement the functions `manhattan_distance` and `cosine_distance`, which will calculate the distance of each training sample in `X_train` to each test sample in `X_test` based on their feature values.
# 
# **Suggested Numpy functions:** \
# `numpy.expand_dims`, `numpy.dot`, `numpy.sqrt`, `numpy.sum` ...
# 
# 

# %%
 # Task 2.1: Manhattan distance
def manhattan_distance(X_train, X_test):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # X_test: numpy array of shape (num_rows_test, num_features)
  # todo start #
  distance = np.sum(np.abs((X_test[:,:,None]-X_train.T[None,:,:])),axis=1)
  # todo end #
  return distance
  # distance: numpy array of shape (num_rows_test, num_rows_train)

# Task 2.2: Cosine distance
def cosine_distance(X_train, X_test):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # X_test: numpy array of shape (num_rows_test, num_features)
  # todo start #
  denominator = np.matmul(X_test,X_train.T)
  dividend = (np.sqrt(np.sum((X_test**2),axis=1))[:,None]*(np.sqrt(np.sum(X_train**2,axis=1)))[None,:])
  distance = 1 - denominator/dividend
  # todo end #
  return distance
  # distance: numpy array of shape (num_rows_test, num_rows_train)

# %% [markdown]
# ## Task 3: Classification based on the Nearest Neighbors
# 
# After calculating the distance measures, it is now possible to get the index of the K Nearest Neighbors. More importantly, we can identify the labels of the K Nearest Neighbors. You can do this very efficiently using a few Numpy functions.
# 
# **Todo:**
# * Implement the function `knn_prediction` that predicts the class of the test samples `X_test`. The predicted class will be based on the majority class of its nearest neighbors.
# * If there are ties in the number of classes, calculate an inverse distance weight to break the tie:
#   * For example, let `k = 5` and let the nearest neighbors for a test sample be `neighbors = [1, 1, 2, 2, 3]`
#   * Because the prediction is tied between Class 1 and Class 2, we calculate the distance of the test sample to its nearest neighbors. Let the distance measures corresponding to each value in `neighbors` be `distances = [5, 6, 7, 8, 10]`
#   * The inverse distances will be `inv_distances = [0.2, 0.16666667, 0.14285714, 0.125, 0.1]`
#   * Calculate the **inverse distance weight** as the sum of the inverse distances for each class. So, the inverse distance weight for Class 1 would be 0.36666667, and for Class 2 it would be 0.26785714.
#   * Since Class 1 has the larger inverse distance weight, the predicted class is Class 1.
# * You can assume there will be no ties in distance.
# * The function should also return `y_classes`, which is just an array of all the possible values of `y_train`.
# 
# **Suggested Numpy functions:** `numpy.arange`, `numpy.argsort`, `numpy.take`, `numpy.expand_dims`, `numpy.where`...
# 
# **Suggested approach:**
# 1. Determine the `k` nearest neighbors for each sample in `y_test`
# 2. Count the number of neighbors belonging to each class and identify the majority class (i.e., the one with the most number of neighbors)
# 3. Check if the number of neighbors in the other classes is equal to the number of neighbors in the majority class, resulting in a tie
# 4. If there is a tie, calculate the **inverse distance weight** for those samples and break the tie
# 
# **Programming challenge for this part:** Accomplish this task without using for loops. This will teach you more efficient ways to write code.

# %%
# Task 3: Classification based on the Nearest Neighbors
def knn_prediction(distances, y_train, k):
  # distance: numpy array of shape (num_rows_test, num_rows_train), return value from previous distance functions
  # y_train: numpy array of shape (num_rows_train, ),  the labels of training data
  # k: integer, k in "K-nearest neighbors"

  # todo start #
  y_classes = np.unique(y_train)
  sort_order = np.argsort(distances,axis=1)
  noZeroD = np.where(distances!=0,distances,np.finfo(float).eps)
  kDistances = 1/(np.take_along_axis(noZeroD,sort_order,axis=1)[:,:k])
  kY_train = np.take_along_axis(y_train.T,sort_order,axis=1)[:,:k]
  bool_kY_train = np.arange(np.max(y_train)+1)[None,None,:] == kY_train[:,:,None]
  Y_count = np.sum(bool_kY_train,axis=1)
  Dist = kDistances[:,:,None] * bool_kY_train
  sumDist = np.sum(Dist,axis=1)
  maxx = np.max(Y_count,axis=1)
  Abool = Y_count == maxx[:,None]
  sumDist = sumDist * Abool
  prediction = np.argmax(sumDist,axis=1)
  # todo end #

  return prediction, y_classes
  # prediction: 1-D numpy array of shape (num_rows_test, )
  # y_classes: 1-D numpy array of shape (num_classes, )

# %% [markdown]
# ## Task 4: Evaluation of the K Nearest Neighbors Classifier
# 
# Now it is time to evaluate the classifier you made in the previous task. In this task, you will make this evaluation by calculating the [F-score](https://en.wikipedia.org/wiki/F-score).
# 
# In summary, here are the relevant calculations for the F-score.
# $$\displaystyle precision = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}} $$
# 
# $$\displaystyle recall = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}} $$
# 
# $$\displaystyle F\text{-score} = \frac{2 * precision * recall}{precision+recall}$$
# 
# **Note:**
# * This means each class has its own F-score.
# * If some classes do not appear in the `X_test` (i.e., $\text{True Positive} + \text{False Positive}$ **or** $\text{True Positive} + \text{False Negative}$ are 0), you will get a value `np.nan`.
#   * Add the code `np.finfo(float).eps` to the denominator when calculating $precision$ or $recall$. This is a very small value approaching 0.
#   * For example, the calculation for $precision$ would become:
#   $$\displaystyle precision = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive } + \epsilon }  $$
#   
# 
# 
# **Todo:** \
# Implement a function `f_score` to calculate the accuracy of the classifier. A score of 1.0 indicates perfect precision and recall, while a score of 0.0 indicates 0 precision or recall.
# 

# %%
# Task 4: Evaluation of the K Nearest Neighbors Classifier
def f_score(y_test, prediction, y_classes):
  num_classes = len(y_classes)
  print("Y_classes",y_classes)
  # todo start #
  f_score_array = np.zeros(num_classes)
  for i in range(num_classes):
      # Fill in the missing code/s #
      bool_test = y_test == y_classes[i]
      bool_prediction = prediction == y_classes[i]
      TP = np.sum(bool_test*bool_prediction)
      FP = np.sum(bool_prediction*np.invert(bool_test))
      FN = np.sum(np.invert(bool_prediction)*bool_test)
      precision = TP / (TP + FP + np.finfo(float).eps)
      recall = TP/(TP + FN + np.finfo(float).eps)
      f1_score = (2*precision*recall) / (precision+recall)
      f_score_array[i] = f1_score
    # todo end #

  return f_score_array
  # f_score: 1-D array with shape (num_classes, )

# %% [markdown]
# ## Task 5: Assigning Data Points to Clusters
# 
# In **K-Means Clustering**, data points are assigned to clusters based on their distances to the cluster centroid. We will divide this process into the following steps:
# 
# ### Task 5.1: Euclidean Distance Calculation
# 
# Typically, Euclidean distance is used in K-Means Clustering as the model may fail to converge in some cases when other distance measures are used.
# 
# The Euclidean distance equation is given below. Suppose we are calculating the distance between a data point $X:(x_1, x_2, ... ,x_n)$ and a cluster centroid $C:(c_1, c_2, ... c_n)$ in n-dimensional space.
# 
# $$\displaystyle d(X, C) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}$$
# 
# This calculation is performed for each sample in the training set.
# 
# ### Task 5.2: Cluster Assignment
# 
# Once the cluster distances have been calculated, we can assign the data points to clusters. Each data point is simply **assigned to the cluster to which it has the minimum distance**.
# 
# ### Task 5.3: Calculate New Centroids
# 
# After the data points have been assigned to their new clusters, these new cluster assignments will be used to determine the new centroids. The new centroid is simply the **mean of the data point features assigned to that cluster**.
# 
# **Programming challenge for this part:** Accomplish this task without using for loops. This will teach you more efficient ways to write code.
# 
# **Todo:**
# * Implement the function `centroid_euclidean_distance` that calculates the distance between each data point and the centroid.
# * Implement the function `cluster_assignment` that returns the index of the cluster assignment for each data point.
# * Implement the function `calculate_centroids` that returns the new centroid for each cluster.
# * Make your solution as efficient as possible (i.e., minimize redundant code, reduce for loops.)
# 
# **Suggested Numpy functions:**
# `numpy.square`, `numpy.sum`, `numpy.sqrt`, `numpy.argmin`, `numpy.newaxis`, `numpy.arange` ...

# %%
# Task 5.1: Euclidean Distance Calculation
def centroid_euclidean_distance(X_train, centroids):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # centroids: numpy array of shape (num_clusters, num_features)
  # todo start #
  distance = np.sum(np.square((centroids[:,:,None]-(X_train.T)[None,:,:])),axis=1)
  # todo end #
  #distance = np.sum(np.square(X_train[:,:,None] - centroids[None,:,:].transpose(0,2,1)),axis=1).transpose(1,0)
  return np.sqrt(distance)
  # distance: numpy array of shape (num_clusters, num_rows_train)
#print(centroid_euclidean_distance(np.array([[1,0,0,1,1],[1,3,45,5,1],[1,2,4,5,6]]),np.array([[1,0,0,1,1],[1,3,45,5,1]])))
# Task 5.2: Cluster Assignment
def cluster_assignment(distance):
  # distance: numpy array of shape (num_clusters, num_rows_train)
  # todo start #
  assignments = np.argmin(distance,axis=0)
  # todo end #
  return assignments
  # assignment: 1-D numpy array of shape (num_rows_train, )

# Task 5.3: Calculate New Centroids
def calculate_centroids(X_train, assignment, k):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # assignment: 1-D numpy array of shape (num_rows_train, )
  # k: a scalar value for the number of clusters (NOTE: Include empty clusters in counting k)
  # todo start #
  bool_assignment = np.arange(k)[None,:] == assignment[:,None]
  mean = X_train[:,:,None] * bool_assignment[:,None,:]
  new_centroids = np.sum(mean,axis=0) / np.sum(bool_assignment,axis=0)
  # todo end #
  return new_centroids.T
  # new_centroids: numpy array of shape (num_clusters, num_features)
#calculate_centroids(train_features.to_numpy()[:5,:3],np.array([1,0,0,1,1]),2)

# %% [markdown]
# ## Task 6: Improving the K-Means Clustering Model
# 
# Usually, we need to run the K-Means Clustering algorithm a few times to find better centroids. This means we will repeatedly apply the functions in Task 5 until a stopping criterion is met.
# 
# We will now try to refine the K-Means Clustering model until some common stopping criteria are met:
# 1. **Task 6.1: Stop When There Are No More Cluster Reassignments**
# 2. **Task 6.2: Stop When Centroid Change is Below Threshold**
# 
# **Todo:**
# * Implement a function `k_means_cluster_reassignment` that continuously refines the centroid until the current and previous iterations result in the same cluster assignments for each data point.
# * Implement a function `k_means_centroid_value` that continuously refines the centroid until the current and previous iterations result in roughly the same centroid values for each feature (with a maximum allowable difference of `threshold_value`).
# * The functions should return:
#   * `assignment` - the final cluster assignments
#   * `centroid` - the final centroids
# * We also need to limit the number of iterations to `max_iterations` in case the model fails to converge.
# * Reminder: You need to use the functions in Task 2 to implement these tasks.
# 
# **Suggested methods:** `break`, `numpy.abs` ...

# %%
# Task 6.1: Stop When There Are No More Cluster Reassignments
def k_means_cluster_reassignment(X_train, initial_centroids, max_iterations=100):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # initial_centroids: numpy array of shape (num_clusters, num_features)
  # max_terations: the maximum number of iterations for refining the model
  centroids = initial_centroids
  k = centroids.shape[0]
  for iteration in range(max_iterations):
    distance = centroid_euclidean_distance(X_train, centroids)
    assignment = cluster_assignment(distance)
    centroids = calculate_centroids(X_train, assignment, k)
    # todo start #
    if iteration != 0 :
      if np.sum(np.abs(assignment - original)) == 0: 
        break
    original = assignment.copy()
    
    # todo end #
  return assignment, centroids
  # assignment: 1-D numpy array of shape (num_rows_train, )
  # centroids: numpy array of shape (num_clusters, num_features)

# Task 6.2: Stop When Centroid Change is Below Threshold
def k_means_centroid_value(X_train, initial_centroids, max_iterations=100, threshold_value=0.0001):
  # X_train: numpy array of shape (num_rows_train, num_features)
  # initial_centroids: numpy array of shape (num_clusters, num_features)
  # max_terations: the maximum number of iterations for refining the model
  # threshold_value: a scaler value for the allowable difference between iterations
  centroids = initial_centroids
  k = centroids.shape[0]
  for iteration in range(max_iterations):
    distance = centroid_euclidean_distance(X_train, centroids)
    assignment = cluster_assignment(distance)
    centroids = calculate_centroids(X_train, assignment, k)
    # todo start #
    if iteration != 0:
      if np.prod((np.abs(centroids - original) >= threshold_value).astype(int)): 
        break
    original = centroids.copy()
    # todo end #
  return assignment, centroids
  # assignment: 1-D numpy array of shape (num_rows_train, )
  # centroids: numpy array of shape (num_clusters, num_features)

# %% [markdown]
# ### Task 6.3: Evaluating the value of k
# 
# Currently, we are arbitrarily deciding on the number of clusters (k). There are also metrics for identifying the best k. One of these is the silhouette score [(see link for more info)](https://en.wikipedia.org/wiki/Silhouette_(clustering)#:~:text=The%20silhouette%20score%20is%20specialized,distance%20or%20the%20Manhattan%20distance.). Below is a summary of the relevant formulas for calculating the silhouette score for a single value of k.
# 
# For each data point $i$ in cluster $C_I$, define $a(i) = \frac{1}{|C_I|-1} \sum_{j \in C_I, i \neq j} d(i, j)$
# where $|C_I|$ is the number of data points in the cluster
# 
# Then we define $b(i) = min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} d(i,j) $
# 
# So, the silhouette score of one data point is
# $$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$
# 
# Then, the silhouette score for a specific value of k is calculated by taking the mean $s(i)$ over the entire dataset.
# 
# **Todo:** \
# Implement a function `silhouette_score_for_k` that calculates the silhouette score for a specific k and the k_means functions you made for previous tasks.
# 
# **Warning:**
# If you are using Google Colab, Colab's CPU capacity may not be able to handle this computation. We recommend using a subset of `X_train`, e.g. `X_train[:300,:]` and `assignments[:300]`
# 
# **Suggested methods:** `numpy.mean`, `numpy.min`, `numpy.maximum`, ...

# %%
# Task 6.3: Evaluating the value of k
def silhouette_score_for_k(X_train, assignments):
  # Note: If you are using Google Colab, the CPU capacity may not be enough for large datasets
  # X_train: numpy array of shape (num_rows_train, num_features)
  # todo start #
  n_samples = len(X_train)
  silhouette_scores = []

  for i in range(n_samples):
      point = X_train[i]
      assignment = assignments[i]
      same_cluster_distances = np.sqrt(np.sum((X_train[assignments == assignment] - point) ** 2, axis=1))
      same_cluster_distances = same_cluster_distances[same_cluster_distances != 0]


      other_cluster_distances = []
      for j in range(len(X_train)):
          if assignments[j] != assignment:
              other_cluster_distance = np.mean(np.sqrt(np.sum((X_train[assignments == assignments[j]] - point) ** 2, axis=1)))
              other_cluster_distances.append(other_cluster_distance)

      if len(other_cluster_distances) == 0:
          silhouette_score = 0
      else:
          min_other_cluster_distance = np.min(other_cluster_distances)
          silhouette_score = (min_other_cluster_distance - np.mean(same_cluster_distances)) / max(min_other_cluster_distance, np.mean(same_cluster_distances))

      silhouette_scores.append(silhouette_score)
      silhouette_coef = np.mean(silhouette_scores)
  # todo end #
  return silhouette_coef
  # silhouette_coef: a scalar value

# %% [markdown]
# ## Final Reminder: ##
# * Review the changelog and FAQ on the <a href="https://course.cse.ust.hk/comp2211/assignments/pa1">assignment webpage</a>.
# * While we provided you with some sample test cases on ZINC, the test cases used for final grading may be different. This means that if you hard code the answers, or make your model specific for this dataset in some way, your final PA1 grade may be much lower than the grade given by ZINC.

# %% [markdown]
# ## Playground: Try out your model here
# 
# You can run the following codes to test your functions. This part will not be graded.

# %%
if __name__ == '__main__':
  X_train = np.array(train_features)
  X_test = np.array(test_features)
  y_train = np.array(train_labels)
  y_test = np.array(test_labels)
  np.random.seed(0)
  n_clusters=3
  initial_centroid = np.random.rand(n_clusters, X_train.shape[1])
  n_neighbors=5

  # Task 1: Data Preprocessing
  X_train = z_score_normalization(X_train)
  X_test = z_score_normalization(X_test)
  print("The normalized features of the first 5 samples of X_train are: ", X_train[:5, :])

  # Task 2.1: Manhattan distance
  m_distance = manhattan_distance(X_train, X_test)
  print('The manhattan distance between the first 5 X_train and first 5 X_test are', m_distance[:5, :5])

  # Task 2.2: Cosine distance
  c_distance = cosine_distance(X_train, X_test)
  print('The cosine distance between the first 5 X_train and first 5 X_test are', c_distance[:5, :5])

  # Task 3: Classification based on the Nearest Neighbors
  prediction, y_classes = knn_prediction(m_distance, y_train, n_neighbors)
  print('The predicted classes for the first 10 samples in X_test are ', prediction[:10])
  print('The true classes of these samples are ', y_test[:10].flatten())

  # Task 4: Evaluation of the K Nearest Neighbors Classifier
  f_score_array = f_score(y_test.flatten(), prediction, y_classes)
  print('The F-scores for each class are ', f_score_array)

  # Task 5-6: K-Means Clustering
  assignment, centroid = k_means_cluster_reassignment(X_train, initial_centroid, max_iterations=100)
  print('With cluster reassignment as the stopping criteria, the assignments for the first 5 samples of X_train are: ', assignment[:5])
  assignment, centroid = k_means_centroid_value(X_train, initial_centroid, max_iterations=100)
  print('With centroid value as the stopping criteria, the assignments for the first 5 samples of X_train are: ', assignment[:5])

  # Task 6.3: Evaluating the value of k
  silhouette_avg = silhouette_score_for_k(X_train[:300, :], assignment[:300])
  print('The silhouette coefficient is ', silhouette_avg)


