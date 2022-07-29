import pathlib
import json
import numpy as np

from typing import List, TypeVar
# generic type to represent a data point
X = TypeVar('X')  
Y = TypeVar('Y') 
Vector = List[float]
Matrix = List[List[float]]

## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)

# type check

x = [1, 2, 3]
type(x.is_integer)

######################    Python Annotations    #########################################

def foo1(x: 'insert x here', y: 'insert x^2 here') -> 'Hi!':
    print('Hello, World')
    return

foo1.__annotations__

def foo2(x: str, y: str) -> 'Hi!':
    print('Hello, World')
    return

foo2.__annotations__

######################    lambda functions    #########################################

# Program to show the use of lambda functions
double = lambda x: x * 2
print(double(5))

# The filter() function in Python takes in a function and a list as arguments.
# Program to filter out only the even items from a list
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(filter(lambda x: (x % 2 == 0) , my_list))
print(new_list)

# The map() function in Python takes in a function and a list.
# Program to double each item in a list using map()
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(map(lambda x: x * 2 , my_list))
print(new_list)

map(lambda x: x + " doing stuff", df['col1'])
list(map(lambda n: n * 2, [1, 2, 3, 4, 5]))

strs = ['Summer', 'is', 'coming']
list(map(lambda s: s.upper() + '!', strs))

def double(n):
    return n * 2
 
nums = [1, 2, 3, 4, 5, 6]
list(map(double, nums))    # use name of function "double"

######################    error messages    #########################################

def add_two(t):
    if not (isinstance(t, int) or isinstance(t, float)):
        raise ValueError('t must be numeric')
    else:
        print(t + 2)

######################    %in% operator    #########################################

x = np.array([1, 2, 3, 10])
y = np.array([10, 11, 2])
np.isin(y, x)

# This is equivalent to:
# c(10, 11, 2) %in% c(1, 2, 3, 10)

integers = range(1, 10)
even = list(filter(lambda x: x % 2 == 0, integers))
print(even)

integersList = range(1, 6)
squares = list(map(lambda x: x * x, integersList))
print(squares)

######################    Regex    #########################################

sample_string1 = 'here_without_space'
sample_string1.replace("_", " ").title()

sample_string2 = 'here.with.dots'
sample_string2.replace(".", " ").capitalize()

sample_string3 = 'Here Is Capital'
sample_string3.replace(" ", "_").lower()

print(*list(map('s{}'.format, range(1, 10))), sep = ', ')

######################    Reminder Functions    #########################################

def show_me_how_venv(installed = False, venv_name = False, venv_input = None, activate = False, version = False):
  
  if installed == False:
    print("#check if you have it installed \n which virtualenv \n #install it \n python3 -m pip install virtualenv")
    
  if venv_input != None:
      name = venv_input
  else: 
      name = 'venv_name'

  if venv_name == False:
    print(f"#create a venv named {name} \n virtualenv {name}")
    
  if activate == False:
    print("#Activate the virtual environment \n source python_notes/bin/activate")
    
  if version == False:
    print("which python")
    
#show_me_how_venv(venv_input = 'cool_venv_name')


def viz_help():
    print("from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap")
    print("bottom = df[np.isin(df['variable'].to_numpy(), bottom_pop['variable'])]")
    print("(ggplot(bottom, aes('year', 'population')) \n + geom_point() \n + facet_wrap('~country'))")
    print("")
    print("")
    print("")
    
def pandas_helper():
    print("df = pd.read_csv('data.csv')")
    print("df.columns = df.columns.str.strip().str.replace('string', 'replacement').str.lower()")
    print('')
    print('df.nunique()')
    print("df.query('variable  < value').sort_values(by=['variable'], ascending=False)")
    print("df.groupby('variable').size().reset_index().sort_values(by=['variable'])")
    print("pd.set_option('display.float_format', '{:.3f}'.format)")
    print("pd.options.display.max_rows = 20")
    
def numpy_mat(arg = 'zeros'):
    if arg == 'zeros':
        print("np.zeros((3, 4))")
        print("3x4 matrix consisiting of 0s")
    if arg == 'ones':
        print("np.ones((3, 4))")
        print("3x4 matrix consisiting of 1s")

######################    Pandas    #########################################

import pandas as pd

df.loc[:, df.isna().any()]
data['race'].value_counts()
df.agg(['count', 'size', 'nunique'])
df.groupby('mID').agg(['count', 'size', 'nunique']).stack()

df.column_name.value_counts().sort_index()

pd.options.<TAB>

pd.set_option('display.max_rows', 500)


# some error messages examples
def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    if len(v) != len(w):
        raise ArithmeticError("cannot add matrices with different shapes")

    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]



# pandas summing
import pandas as pd
df = pd.DataFrame()
df.loc[:, ['a', 'c', 'd']].sum(axis=0)
df[['a', 'c', 'd']].sum(axis=0)
df[['a', 'c', 'd']].values.sum(axis=0)
[df[col].values.sum(axis=0) for col in ('a', 'c', 'd')]

# Counting the number of missing values
import numpy as np

np.count_nonzero(np.isnan(df))

np.count_nonzero(~np.isnan(df))

print('total sum:', np.nansum(df))
print('column sums:', np.nansum(df, axis=0))
print('row sums:', np.nansum(df, axis=1))

# Removing all rows that contain missing values
df[~np.isnan(df).any(1)]

# Convert missing values to 0
df0 = np.nan_to_num(df)

# Converting certain numbers to NaN
df0[df0==0] = np.nan

# Remove all missing elements from an array
df[~np.isnan(df)]
