
But my dataset has many more rows than columns, so what am I supposed to do about that?" Just wait! It'll be ok. We're not actually going to take the eigenvectors of the dataset 'directly', we're going to take the eigenvectors of the covariance matrix of the dataset.

Note: your dataset having more rows than columns is not an issue since we will be using the eigenvectors of the covariance matrix of the dataset.




***


## 2. Use the eigenvalues to get the eigenvectors

Although it was anncounced in mid 2019 that [you can get eigenvectors directly from eigenvalues](https://arxiv.org/abs/1908.03795), the usual way people have done this for a very long time is to go back to the matrix $\bf{A}$ and solve the *linear system* of equation (1) above, for each of the eigenvalues.  For example, for $\lambda_1=-1$, we have 

$$
{\bf}A \vec{v}_1 = -\vec{v}_1
$$

$$
\begin{bmatrix}
-2 & 2 & 1 \\
-5 & 5 & 1 \\
-4 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
v_{1x} \\
v_{1y} \\
v_{1z} \\
\end{bmatrix}
= -
\begin{bmatrix}
v_{1x} \\
v_{1y} \\
v_{1z} \\
\end{bmatrix}$$

This amounts to 3 equations for 3 unknowns, which I'm going to assume you can handle. For the other eigenvalues things proceed similarly.  The solutions we get for the 3 eigenvalues are: 

$$\lambda_1 = 3: \ \ \ \vec{v}_1 = (1,2,1)^T$$
$$\lambda_2 = 2: \ \ \ \vec{v}_2 = (1,1,2)^T$$
$$\lambda_3 = 1: \ \ \ \vec{v}_3 = (1,1,1)^T$$


Since our original equation (1) allows us to scale eigenvectors by any artibrary constant, often we'll express eigenvectors as *unit* vectors $\hat{v}_i$.  This will amount to dividing by the length of each vector, i.e. in our example multiplying by $(1/\sqrt{6},1/\sqrt{6},1/\sqrt{3})$.  

In this setting 

$$\lambda_1 = 3: \ \ \ \hat{v}_1 = (1/\sqrt{6},2/\sqrt{6},1/\sqrt{6})^T$$
$$\lambda_2 = 2: \ \ \ \hat{v}_2 = (1/\sqrt{6},1/\sqrt{6},2/\sqrt{6})^T$$
$$\lambda_3 = 1: \ \ \ \hat{v}_3 = (1,1,1)^T/\sqrt{3}$$

Checking our answers (left) with numpy's answers (right):

***

## What is PCA?

`https://setosa.io/ev/eigenvectors-and-eigenvalues/`

- Based on the data find a new set of orthogonal feature vectors in such a way that the data spread is maximum in the direction of the feature vector (or dimension)
- Rates the feature vector in the decreasing order of data spread (or variance)
- The datapoints have maximum variance in the first feature vector, and minimum variance in the last feature vector
- The variance of the datapoints in the direction of feature vector can be termed as a measure of information in that direction.

### Steps

1. Standardize the datapoints $X_{new} = {X - mean(X) \over std(X)}$
2. Find the covariance matrix from the given datapoints $C[i, j] = cov(x_{i}, x_{j})$
3. Carry out eigen-value decomposition of the covariance matrix $C = V\Sigma V^{-1}$
4. Sort the eigenvalues and eigenvectors $\Sigma_{sort} = sort(\Sigma) V_{sort} = sort(V, \Sigma_{sort})$

## Dimensionality Reduction with PCA

- Keep the first m out of n feature vectors rated by PCA. These m vectors will be the best m vectors preserving the maximum information that could have been preserved with m vectors on the given dataset

### Steps

1. Carry out steps 1-4 from above
2. Keep first m feature vectors from the sorted eigenvector matrix $V_{reduced} = V[:, 0:m]$
3. Transform the data for the new basis (feature vectors) $X_{reduced} = X_{new} \times V_{reduced}$
4. The importance of the feature vector is proportional to the magnitude of the eigen value

***

Our focus is on feature extraction through PCA and it is done through 2 ways,

- Singular Value Decomposition (SVD) or
- Performing Eigen Value Decomposition over the Covariance Matrix of the dataset

The salience of eigen value decomposition involves

- Computing eigenvectors and values as the foundation
- Eigenvectors are the Principal Components that determines the direction of the new dimension(feature space)
- Eigenvalues determine their magnitude, it explains the variance of the data along the new feature axes

## Eigen Decomposition

Eigen decomposition of the covariance matrix is PCA, It is a square matrix makes it conducive for linear transformations like shearing and rotation.

Covariance Matrix
$$
\mathbb{A} = \sigma_{jk} = {1 \over n-1} \sum_{i=0}^{n} (x_{ij} - \bar{x}_{j})(x_{ik} - \bar{x}_{k})
$$
then  
Eigen Decomposition
$$\mathbb{A}v = \lambda v$$

Where

- $\mathbb{A} \in \mathbb{R}^{m \times m}$ Covariance matrix
- $v \in \mathbb{R}^{m \times 1}$ Eigenvectors, column vector
- $\lambda \in \mathbb{R}^{m \times m}$ Eigenvalues, diagonal matrix

$\mathbb{A}v - \lambda v = 0$ 
e.g. ($\mathbb{A} - \lambda \mathbb{I})v = 0$  
This is possible only when  
$det(\mathbb{A} - \lambda \mathbb{I}) = |\mathbb{A} - \lambda \mathbb{I}| = 0$

Let us say, we want to reduce the dimensions to a count of 3 - then the top 3 eigen vecors are the principal components of our interest. i.e

$$v \in \mathbb{R}^{3 \times 1}$$

### Transformations and Basis Vectors

We do transformations across vector spaces for identifying the convergence points during training or to construct complex/sophisticated distributions like normalizing flows that was discussed in some of the posts in the the past Eigen Decomposition can be rewritten as $T(\vec{v}) = \lambda \vec{v}$, where $T$ is the transofmriation ($T: \mathbb{R}^{n} \to \mathbb{R}^{n}$) with an associated eigenvalue $\lambda$ for the vector $\vec{v}$. This transformation is nothing but the vector that scales(or reverses) up the matrix and nothing more, when this property is observed then we call that vector an eigenvector. Then

$$(\mathbb{A} - \lambda \mathbb{I})\vec{v} = \vec{0}$$

Since, $\vec{v} \ne \vec{0}$ makes $\vec{v}$ is a non trivial member of the null space of the matrix $\mathbb{A} - \lambda \mathbb{I}$, e.g.

Eigenvector as a member of a 
