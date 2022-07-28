
import numpy as np

a = array([2, 3, 4, 5])	
# Input is a 3,4 array
a = array([
  [11, 12, 13, 14],
  [21, 22, 23, 24],
  [31, 32, 33, 34]])

reshaping = {
  'Bind rows': ['''np.concatenate((a, b), axis = 0)''', '''vstack((a, b))'''],
  'Bind columns': ['''np.concatenate((a, b), axis = 1)''', '''hstack((a, b))'''],
  '0 filled array': '''zeros((3, 5), Float)''',
  '1 filled array': '''ones((3, 5), Float)''',
  'Reshaping (rows first)': ['''np.arange(1, 7).reshape(2, -1)''', '''a.setshape(2, 3)'''],
  'Reshaping (columns first)': '''np.arange(1, 7).reshape(-1, 2).transpose()''',
}

slicing = {
  'Element 2, 3 (row,col)': '''a[1, 2]''',
  'First row': '''a[0,]''',
  'First column': '''a[:, 0]''',
  'Miss the first element': '''a[1:]''',
  'Last element': '''a[-1]''',
  'Last two elements': '''a[-2:]''',
  'All, except first row': '''a[1: ,]''',
  'Last two rows': '''a[-2: ,]''',
  'Remove one column': '''a.take([0, 2, 3], axis = 1)''',
  'Reverse': '''a[::-1]'''
}

linear_alg = dict(
  Determinant = '''np.linalg.det(a)''', 
  Inverse = '''np.linalg.inv(a)''',
  Pseudoinverse = '''np.linalg.pinv(a)''',
  Norms = '''np.linalg.norm(a)''',
  Eigenvalues = '''np.linalg.eig(a)[0]''',
  Singular-values = '''np.linalg.svd(a)''',
  Eigenvectors = '''np.linalg.eig(a)[1]''',
  Rank = '''pd.rank(a)'''
)

wrangling = {
  'Sum of each column': '''a.sum(axis = 0)''',
  'Sum of each row': '''a.sum(axis = 1)'''
  'Sum of all elements': '''a.sum()'''
  'Cumulative sum (columns)': '''a.cumsum(axis = 0)''',
  'sequence': '''np.arange(start = 1, stop = 11, step = 1, dtype = float)''',
  'repeat array': '''np.concatenate((a, a))''',
  'Repeat values 2 times': ['''a.repeat(3)''', '''a.repeat(a)'''],
  'Unique values': ['''unique1d(a)''', '''unique(a)''', '''set(a)''']
}





