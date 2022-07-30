
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
