# README

Reattempting this. However, starting from scratch.


<blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
    <strong>What's In a Title?</strong><br>
The <strong>gradient</strong> is a vector that tells us in what direction the weights need to go. More precisely, it tells us how to change the weights to make the loss change <em>fastest</em>. We call our process gradient <strong>descent</strong> because it uses the gradient to <em>descend</em> the loss curve towards a minimum. <strong>Stochastic</strong> means "determined by chance." Our training is <em>stochastic</em> because the minibatches are <em>random samples</em> from the dataset. And that's why it's called SGD!
</blockquote>

- this will appear differently in a notebook

<hr>


<h1> README </h1>

```python
import numpy as np

a = array([2, 3, 4, 5])	
# Input is a 3,4 array
a = array([
  [11, 12, 13, 14],
  [21, 22, 23, 24],
  [31, 32, 33, 34]])
```

<details>
<summary> <h2> Slicing </h2> </summary>

|                        |                               |
|------------------------|-------------------------------|
| Element 2, 3 (row,col) | `a[1, 2]`                     |
| First row              | `a[0,]`                       |
| First column           | `a[:, 0]`                     |
| Skip the first element | `a[1:]`                       |
| Last element           | `a[-1]`                       |
| Last two elements      | `a[-2:]`                      |
| All, except first row  | `a[1: ,]`                     |
| Last two rows          | `a[-2: ,]`                    |
| Remove one column      | `a.take([0, 2, 3], axis = 1)` |
| Reverse                | `a[::-1]`                     |

</details>

<details>
<summary> <h2> Data Wrangling </h2> </summary>

|                          |                                                            |               |
|--------------------------|------------------------------------------------------------|---------------|
| Sum of each column       | `a.sum(axis = 0)`                                          |               |
| Sum of each row          | `a.sum(axis = 1)`                                          |               |
| Sum of all elements      | `a.sum()`                                                  |               |
| Cumulative sum (columns) | `a.cumsum(axis = 0)`                                       |               |
| sequence                 | `np.arange(start = 1, stop = 11, step = 1, dtype = float)` |               |
| repeat array             | `np.concatenate((a, a))`                                   |               |
| Repeat values 2 times    | `a.repeat(3)`                                              | `a.repeat(a)` |
| Unique values            | `unique1d(a)`                                              | `unique(a)`   |
|                          | `set(a)`                                                   |    


</details>

<details>
<summary> <h2> Reshaping </h2> </summary>

|                           |                                              |                    |
|---------------------------|----------------------------------------------|--------------------|
| Bind rows                 | `np.concatenate((a, b), axis = 0)`           | `vstack((a, b))`   |
| Bind columns              | `np.concatenate((a, b), axis = 1)`           | `hstack((a, b))`   |
| 0 filled array            | `zeros((3, 5), Float)`                       |                    |
| 1 filled array            | `ones((3, 5), Float)`                        |                    |
| Reshaping (rows first)    | `np.arange(1, 7).reshape(2, -1)`             | `a.setshape(2, 3)` |
| Reshaping (columns first) | `np.arange(1, 7).reshape(-1, 2).transpose()` |                    |

</details>

<details>
<summary> <h2> Linear Algebra </h2> </summary>

|                 |                       |
|-----------------|-----------------------|
| Determinant     | `np.linalg.det(a)`    |
| Inverse         | `np.linalg.inv(a)`    |
| Pseudoinverse   | `np.linalg.pinv(a)`   |
| Norms           | `np.linalg.norm(a)`   |
| Eigenvalues     | `np.linalg.eig(a)[0]` |
| Singular-values | `np.linalg.svd(a)`    |
| Eigenvectors    | `np.linalg.eig(a)[1]` |
| Rank            | `pd.rank(a)`          |

</details>

<details>
<summary> <h2> Strip Time Formatting </h2> </summary>

| Directive | Meaning                                                                                                                                                                          | Example                           |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| %a        | Weekday as locale’s abbreviated name.                                                                                                                                            | Sun, Mon, etc.                    |
| %A        | Weekday as locale’s full name.                                                                                                                                                   | Sunday, Monday, etc.              |
| %w        | Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.                                                                                                                | 0, 1, ..., 6                      |
| %d        | Day of the month as a zero-padded decimal number.                                                                                                                                | 01, 02, ..., 31                   |
| %b        | Month as locale’s abbreviated name.                                                                                                                                              | Jan, Feb, etc.                    |
| %B        | Month as locale’s full name.                                                                                                                                                     | January, February, etc.           |
| %m        | Month as a zero-padded decimal number.                                                                                                                                           | 01, 02, ..., 12                   |
| %y        | Year without century as a zero-padded decimal number.                                                                                                                            | 00, 01, ..., 99                   |
| %Y        | Year with century as a decimal number.                                                                                                                                           | 0001, 0002, ..., 2013, 2014, etc. |
| %H        | Hour (24-hour clock) as a zero-padded decimal number.                                                                                                                            | 00, 01, ..., 23                   |
| %I        | Hour (12-hour clock) as a zero-padded decimal number.                                                                                                                            | 01, 02, ..., 12                   |
| %M        | Minute as a zero-padded decimal number.                                                                                                                                          | 00, 01, ..., 59                   |
| %S        | Second as a zero-padded decimal number.                                                                                                                                          | 00, 01, ..., 59                   |
| %j        | Day of the year as a zero-padded decimal number.                                                                                                                                 | 001, 002, ..., 366                |
| %U        | Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0. | 00, 01, ..., 53                   |
| %W        | Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0. | 00, 01, ..., 53                   |

`https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior`

</details>

<hr>

<details>
<summary> <h2> Git Commands </h2> </summary>

### Basic Snapshotting

| Command | Description |
| ------- | ----------- |
| `git status` | Check status |
| `git add [file-name.txt]` | Add a file to the staging area |
| `git add -A` | Add all new and changed files to the staging area |
| `git commit -m "commit message"` | Commit changes |
| `git rm -r [file-name.txt]` | Remove a file (or folder) |

### Branching & Merging

| Command | Description |
| ------- | ----------- |
| `git branch` | List branches (the asterisk denotes the current branch) |
| `git branch -a` | List all branches (local and remote) |
| `git branch [branch name]` | Create a new branch |
| `git branch -d [branch name]` | Delete a branch |
| `git push origin --delete [branch name]` | Delete a remote branch |
| `git checkout -b [branch name]` | Create a new branch and switch to it |
| `git checkout -b [branch name] origin/[branch name]` | Clone a remote branch and switch to it |
| `git checkout [branch name]` | Switch to a branch |
| `git checkout -` | Switch to the branch last checked out |
| `git checkout -- [file-name.txt]` | Discard changes to a file |
| `git merge [branch name]` | Merge a branch into the active branch |
| `git merge [source branch] [target branch]` | Merge a branch into a target branch |
| `git stash` | Stash changes in a dirty working directory |
| `git stash clear` | Remove all stashed entries |

### Sharing & Updating Projects

| Command | Description |
| ------- | ----------- |
| `git push origin [branch name]` | Push a branch to your remote repository |
| `git push -u origin [branch name]` | Push changes to remote repository (and remember the branch) |
| `git push` | Push changes to remote repository (remembered branch) |
| `git push origin --delete [branch name]` | Delete a remote branch |
| `git pull` | Update local repository to the newest commit |
| `git pull origin [branch name]` | Pull changes from remote repository |
| `git remote add origin ssh://git@github.com/[username]/[repository-name].git` | Add a remote repository |
| `git remote set-url origin ssh://git@github.com/[username]/[repository-name].git` | Set a repository's origin branch to SSH |

### Inspection & Comparison

| Command | Description |
| ------- | ----------- |
| `git log` | View changes |
| `git log --summary` | View changes (detailed) |
| `git diff [source branch] [target branch]` | Preview changes before merging |
</details>

<hr>

<details>
<summary> <h2> LaTeX in Notebooks </h2> </summary>

```python
from IPython.display import display, Math

a = 3
b = 5
print("The equation is:")
display(Math(f"y= {a}x+{b}"))
```

```
# output shows it properly
The equation is:
\displaystyle y= 3x+5y=3x+5
```

<h3> More LaTeX </h3>

```python
# !pip install latexify-py

import math
import latexify

@latexify.with_latex
def solve(a, b, c):
    return (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)

print(solve(1, 4, 3))
print(solve)
print()
solve
```

```
# print(solve) gives latex output
\mathrm{solve}(a, b, c)\triangleq \frac{-b + \sqrt{b^{2} - 4ac}}{2a}

# solve gives the formatted output
```

```python
@latexify.with_latex
def sinc(x):
    if x == 0:
        return 0
    else:
        return 1

# gives nested output
```

</details>


<h2> Enumerate: Get Counter and Value While Looping </h2>

```python
arr = ['a', 'b', 'c', 'd', 'e']

for i, val in enumerate(arr):
    print(i, val)
```

```
0 a
1 b
2 c
3 d
4 e
```

<h2> Get count of items in list </h2>

```python
from collections import Counter

char_list = ["a", "b", "c", "a", "d", "b", "b"]
Counter(char_list)
```

```
Counter({'a': 2, 'b': 3, 'c': 1, 'd': 1})
```



<h2> **kwargs: Pass Multiple Arguments to a Function in Python </h2>

```python
# Once **kwargs argument is passed, you can treat it like a Python dictionary.

parameters = {'a': 1, 'b': 2}

def example(c, **kwargs):
    print(kwargs)
    for val in kwargs.values():
        print(c + val)

example(c=3, **parameters)
```

```
{'a': 1, 'b': 2}
4
5
```

<h2> Directory </h2>

```python
from pathlib import Path

# Create a new directory
folder = Path('new')
folder.mkdir(exist_ok=True)

# Create new file inside new directory
file = folder / 'new_file.txt'
file.touch()

# !tree new 

# If you want to get the path to folders/files from the home directory
path = Path.home()

docs = path / 'Documents'
pictures = path / 'Pictures'
```


