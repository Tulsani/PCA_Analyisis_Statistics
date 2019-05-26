#PCA (principal component analysis) algorithm used for feature
# extraction and for dimensionality reduction.
#also can be used for explaining data behaviour
#Used in analysis of genome data and gene expression

#helps identify patterns based on correlation between features
#PCA aims to find directions of maximum variance in high-dimesnional
#data and project it into a new subspace with equal or fewer dimensions

# PCA for dimensionality reduction can be used for 
# reducing a [DxK] vector to [n x D] vector where D>n
# Thus leads to vertical collapse of the table

# the first principal component will have largest possible 
# variance and all consequent principal components are uncorrleated
# E(x).E(y)=E(x.y) or orthogonal in nature

# 1. standardizing the d - dimensional matrix
# 2. construct a covariance matrix
# 3. Decompose the covariance matrix into eignvalues and eignmatrix
# 4. Sort the eignvalues by decreasing order to rank the corresponding eigenvectors
# 5. Sort k eigenvectors which correspond to the k largest eigenvalues ,
#    where k is the dimesionality of the new feature subspace
# 6. Cosntruct a projection matrix W from the top k eigen values
# 7. Transform the d-dimensional input dataset using projection matrix
#    W to obtain the new k-dimesnional feature subspace

import pandas as pd 

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

print(df_wine.head())

#dividing wine data into training and test set
# 70 : 30 split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#split into training and test sets

X , y = df_wine.iloc[:, 1:].values , df_wine.iloc[:,0].values
X_train , X_test , y_train , y_test = train_test_split(
  X , y , test_size = 0.3, stratify = y,
    random_state = 0
)

# standadizing the features

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

# constructing co variance matrix
# the symetrix [d x d] matrix stores pairwise
# covariances between the different features

# covaraince between two features Xj and Xk given as
# 1/n * summision(X(i)j - Uj)*(X(i)k)-Uk)
# where U is the sample mean

# Sample means are zero if we standardized  the dataset
# A postive covariance shows that sample incerease and decrease together
# A negative covariance shows features vary in opposite manner

# eignvectors of covariance matrix represent the principal components 
# (the direction of maximum covariance ) whereas the corresponding 
# eign values define their magnitude
# we obtain 13 eigne vectors and eignvalues from 13*13 covariance matrix

# we need to obatian eigen pairs of covariance matrix
# eigen vector v satisfies  sum of v = lamda v 
# lamda is a scale or the eigen value

import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals , eigen_vecs = np.linalg.eig(cov_mat)

# to reduce dimesionality of our dataset by comporessing it ,
# we only select the subset of eigenvectors (principal components) 
# that have the most information (variance)

# plotting varince explained ratios

import matplotlib.pyplot as plt

# calculating cumulative sum of explained variances 
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp) # cumsum returns cummalitve sum along a axis

#plot explained variances
plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='indivisual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',label="cummalative explained varaince")
plt.xlabel("Principal component index")
plt.ylabel("Explained variance ratio")
plt.legend(loc="best")
plt.show()

# the plot shows that  the first principal component alone accounts for 40% of the variance
# the plot also shows tha the first and second principal component combined account for 60% 
# variance

# sort the eigen pairs in descending manner , construct a porjection matrix
#from slected eigenvectors and use projection matrix to transform the data to lower dimension

#making list of eigenvalue , eigenvector tuples
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]

#sort the (eigenvalue , eigenvector ) tuple from high to low
eigen_pairs.sort(key=lambda k:k[0], reverse=True)

#now we capture the top two eigen pairs to capture 60% of the variance
# selected two eigen values since ploting two dimensional values
# number of principal components has to be trade off between computational
# effiecny and performance

w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W: \n',w)

# hence W = projection matrix is [13*2] matix formed using top eigen vectors
# using w we now transform sample x represtend as [1*13] onto the PCA subsapce 
# obtaining x' , x' = x . W

X_train_std[0].dot(w)

# similarly we can transform the entire [124* 13] training set onto 
# two principal components by calculating the dot product
# X' = X.W

X_train_pca = X_train_std.dot(w)

# hence visualising wine data set as 124 * 2 dimesnional scatter plot

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()