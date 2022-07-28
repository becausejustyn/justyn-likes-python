import pathlib
import json
import numpy as np

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


######################    Pandas    #########################################

import pandas as pd

df.loc[:, df.isna().any()]
data['race'].value_counts()
df.agg(['count', 'size', 'nunique'])
df.groupby('mID').agg(['count', 'size', 'nunique']).stack()

df.column_name.value_counts().sort_index()

pd.options.<TAB>

pd.set_option('display.max_rows', 500)
