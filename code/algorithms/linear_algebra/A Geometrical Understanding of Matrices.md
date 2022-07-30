# A Geometrical Understanding of Matrices

```
https://gregorygundersen.com/blog/2018/10/24/matrices/
```

## Matrices as linear transformations
***

In linear algebra, we are interested in equations of the following form:

$$
\mathbf{b} = A\mathbf{x}
$$

Where $\mathbf{b} \in \mathbb{R}^{m \times n}$, and $\mathbf{x} \in \mathbb{R}^{n}$. One way to think about this equation is that $A$ represents a system of $m$ linear equations, each with $n$ variables, and $\textbf{x}$ represents a solution to this system. 

But there is another way to think of the matrix $A$, which is as a linear function $f$ from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$: 

$$
f(\mathbf{x}) = A\mathbf{x}
$$

In my mind, the easiest way to see how matrices are linear transformations is to observe that the columns of a matrix $A$ represent where the standard basis vectors in $\mathbb{R}^{n}$ map to in $\mathbb{R}^{m}$. Let’s look at an example. Recall that the standard basis vectors in $\mathbb{R}^{3}$ are:

$$
\mathbf{e}_{1} =
\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}
\;\;
\mathbf{e}_{2} =
\begin{bmatrix}
0 \\ 1 \\ 0
\end{bmatrix}
\;\;
\mathbf{e}_{3} =
\begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}
$$

By definition, this means that every vector in $\mathbb{R}^{3}$ is a linear combination of this set of basis vectors:

$$
\mathbf{x} =
\begin{bmatrix}
x_{1} \\ x_{2} \\ x_{3}
\end{bmatrix}
\;\;
= x_{1}
\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}
\;\;
= x_{2}
\begin{bmatrix}
0 \\ 1 \\ 0
\end{bmatrix}
\;\;
= x_{3}
\begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}
$$

Now let’s see what happens when we matrix multiply one of these standard basis vectors, say $\textbf{e}_{2}$, by a transformation matrix $A$:

$$
\overbrace{
\left[
\begin{array}{c|c|c}
1 & \color{green}{2} & 0 \\
-5 & \color{green}{3} & 1 \\
\end{array}
\right]
}^{\mathbf{A}}
\overbrace{
\left[
\begin{array}{ccc}
0 \\ 1 \\ 0
\end{array}
\right]
}^{\mathbf{e_{2}}}
=\
\begin{bmatrix}
0 + \color{green}{2} + 0 \\ 
0 + \color{green}{3} + 0
\end{bmatrix}
=\
\begin{bmatrix}
\color{green}{2} \\ 
\color{green}{3}
\end{bmatrix}            
$$

In words, the second column of $A$ tells us where the second basis vector in $\mathbb{R}^{R}$ maps to in $\mathbb{R}^{2}$. If we horizontally stack the standard basis vectors into a $3 \times 3$ matrix, we can see where each basis vector maps to in $\mathbb{R}^{2}$ with a single matrix multiplication:

$$
\left[
\begin{array}{c|c|c}
1 & 2 & 0 \\
-5 & 3 & 1 \\
\end{array}
\right]
\left[
\begin{array}{c|c|c}
\color{Orange}{1} & \color{green}{0} & \color{Violet}{0} \\
\color{Orange}{0} & \color{green}{1} & \color{Violet}{0} \\
\color{Orange}{0} & \color{green}{0} & \color{Violet}{1} 
\end{array}
\right]
=\
\left[
\begin{array}{c|c|c}
\color{Orange}{1} & \color{green}{2} & \color{Violet}{0} \\
\color{Orange}{-5} & \color{green}{3} & \color{Violet}{1} 
\end{array}
\right]
$$

Now here’s the cool thing. We can express any transformed 2-vector as a linear combination of $f(\textbf{e}_{1})$, $f(\textbf{e}_{2})$, and $f(\textbf{e}_{3})$, where the coefficients are the components of the untransformed 3-vector. For example, if $\textbf{x} = [1.2 1.5 −2]^{\top}$, then:

$$
1.2
\overbrace{
\left[
\begin{array}{c}
1 \\
-5 \\
\end{array}
\right]
}^{f \mathbf{e_{1}}}
+ 1.5
\overbrace{
\left[
\begin{array}{c}
2 \\
3
\end{array}
\right]
}^{f \mathbf{e_{2}}}
- 2
\overbrace{
\left[
\begin{array}{c}
0 \\
1
\end{array}
\right]
}^{f \mathbf{e_{3}}}
=\
\left[
\begin{array}{c}
1.2 + 3 + 0\\
-6 + 4.5 -2
\end{array}
\right]
=\
\overbrace{
\left[
\begin{array}{c}
4.2 \\
-3.5
\end{array}
\right]
}^{f \mathbf{x}}
$$

It’s worth staring at these equations for a few minutes. In my mind, seeing Equations 1 and 2 together helped me understand why matrix multiplication is defined as it is. When we perform matrix multiplication, we are projecting a vector or vectors into a new space defined by the columns of the transformation matrix. And $f(\textbf{x})$ is just a linear combination of the columns of $A$, where the coefficients are the components of $\textbf{x}$.

In my mind, changing how we see the equation $\textbf{b} = Ax$ is not trivial. In the textbook Numerical Linear Algebra (Trefethen & Bau III, 1997), the authors claim that seeing matrices this way is "essential for a proper understanding of the algorithms of numerical linear algebra."

This linearity is what allows us to say concretely that any linear transformation, $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$, can be represented as a linear combination of the transformed basis vectors of $\textbf{x}$, which can be encoded as a matrix-vector multiplication:

$$
\begin{equation} 
\begin{split}
\mathbf{x} & = x_{1}\mathbf{e}_{1} + x_{2}\mathbf{e}_{2} + \dots + x_{n}\mathbf{e}_{n} \\
f \mathbf{x} & = f(x_{1}\mathbf{e}_{1} + x_{2}\mathbf{e}_{2} + \dots + x_{n}\mathbf{e}_{n}) \\
& = f(x_{1}\mathbf{e}_{1}) + f(x_{2}\mathbf{e}_{2}) + \dots + f(x_{n}\mathbf{e}_{n}) \; \text{ Additivity} \\
& = x_{1}f(\mathbf{e}_{1}) + x_{2}f(\mathbf{e}_{2}) + \dots + x_{n}f(\mathbf{e}_{n}) \; \text{ Homogeneity of degree 1} \\ 
& = [ f(\mathbf{e}_{1}) | f(\mathbf{e}_{2}) | \dots | f(\mathbf{e}_{n}) ] \mathbf{x}
\end{split}
\end{equation}
$$

`https://gregorygundersen.com/blog/2018/10/24/matrices/#1-proof-that-matrices-are-linear-transformations`

`https://en.wikibooks.org/wiki/LaTeX/Colors`

## Visualizing matrix transformations

`https://gregorygundersen.com/blog/2018/10/24/matrices/`

The linearity of matrix transformations can be visualized beautifully. For ease of visualization, let’s only consider $2 \times 2$ matrices, which represent linear transformations from $\mathbb{R}^{2}$ to $\mathbb{R}^{2}$. For example, consider the following matrix transformation $A$ of a vector $\textbf{x} = [1 \; 2]^{\top}$ :

$$
\overbrace{
\left[
\begin{array}{c|c}
\color{Orange}{2} & \color{green}{-1} \\
\color{Orange}{0} & \color{green}{1} \\
\end{array}
\right]
}^{\mathbf{A}}
\overbrace{
\left[
\begin{array}{cc}
1 \\ 2
\end{array}
\right]
}^{\mathbf{x}}
=\
\overbrace{
\left[
\begin{array}{cc}
0 \\ 2
\end{array}
\right]
}^{f \mathbf{x}}
$$

We can visualize two important properties of this operation (Figure 1). First, the columns of $A$ represent where the standard basis vectors in $\mathbb{R}^{2}$ land in this transformed vector space.

# Singular Value Decomposition as Simply as Possible

`https://gregorygundersen.com/blog/2018/12/10/svd/`

Ok

`https://www.3blue1brown.com/topics/linear-algebra`

`https://www.3blue1brown.com/topics/neural-networks`


