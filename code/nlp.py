

import numpy as np
import pandas as pd
import re
import string
import csv
import random
from collections import Counter

from textblob import TextBlob

# Most simple and dummy method possible for tokenizing your text
def get_words(text):
    return text.strip().split()

# Our function to apply. When applied to a dataframe this will get a row as input
def extract_text_basic_stats(row):
    # tokenize our message text
    words = get_words(row['text'])
    # Compute message stats and add entries to the row
    # For demonstration purposes, but otherwise clearly inefficient way to do it
    row['text_len'] = len(row['text'])
    row['num_tokens'] = len(words)
    row['num_types'] = len(set(words))
    return row

# We apply row wise, so axis = 1
#msgs_stats = msgs.apply(extract_text_basic_stats, axis=1)

# Our function to apply. When applied to a series/column this will get a cell value as input
def extract_text_basic_stats(text):
    words = get_words(text)
    # Return results as tuples. You can also return a dictionary for automatic insertion in a dataframe
    return len(text), len(words), len(set(words))

# Apply function to text column
#stats = msgs['text'].apply(extract_text_basic_stats).values
# Pack results in a new dataframe and join it with our original one
#msgs_stats = pd.DataFrame(list(stats), columns=['text_len', 'num_tokens', 'num_types'])
#msgs_stats = msgs.join(msgs_stats)

def palindrome(my_str):
    """
    Returns True if an input string is a palindrome. Else returns False.
    """
    stripped_str = "".join(l.lower() for l in my_str if l.isalpha())
    return stripped_str == stripped_str[::-1]

example = "Go hang a salami. I'm a lasagna hog."
print('example',  palindrome(example))

# (If you're not familiar with the `Counter` object, 
# [return to this tutorial for a refresher](https://gist.github.com/aparrish/4b096b95bfbd636733b7b9f2636b8cf4). 
# If you're not familiar with TextBlob, 
# [I wrote a tutorial about it here](http://rwet.decontextualize.com/book/textblob/).)
# read in the contents of genesis.txt

text = open("genesis.txt").read()
blob = TextBlob(text)

all_pos = list()
for word, pos in blob.tags:
    all_pos.append(pos)
    
pos_count = Counter(all_pos)

# open the file "genesis_pos.csv" for writing...
with open("genesis_pos.csv", "w") as csvfile:
    # create a csv "writer" object
    writer = csv.writer(csvfile)
    # write the header row
    writer.writerow(["part of speech", "count"])
    # write out each pair as a line in the CSV
    for item, count in pos_count.most_common():
        writer.writerow([item, count])

# Here's a program that reads in this data and then prints out five random dog names:

all_names = list()

for row in csv.DictReader(open("dogs-of-nyc.csv")):
    all_names.append(row['dog_name'])
    
print(", ").join(random.sample(all_names, 5))
print(", ".join(random.sample(all_names, 5)))

# prints out the most common dog colors:

all_colors = list()

for row in csv.DictReader(open("dogs-of-nyc.csv")):
    all_colors.append(row['dominant_color'])
    
color_count = Counter(all_colors)

for item, count in color_count.most_common():
    print(item, count)


#############################################################
#### case-insensitive ignoring punctuation characters
############################################################

def palindrome_short(my_str):
    stripped_str = "".join(l.lower() for l in my_str if l.isalpha())
    return stripped_str == stripped_str[::-1]

def palindrome_regex(my_str):
    return re.sub('\W', '', my_str.lower()) == re.sub('\W', '', my_str[::-1].lower())

def palindrome_stringlib(my_str):
    LOWERS = set(string.ascii_lowercase)
    letters = [c for c in my_str.lower() if c in LOWERS]
    return letters == letters[::-1]

LOWERS = set(string.ascii_lowercase)
def palindrome_stringlib2(my_str):
    letters = [c for c in my_str.lower() if c in LOWERS]
    return letters == letters[::-1]

def palindrome_isalpha(my_str):
    stripped_str = [l for l in my_str.lower() if l.isalpha()]
    return stripped_str == stripped_str[::-1]

############################################################
#### functions considering all characters (case-sensitive)
############################################################

def palindrome_reverse1(my_str):
    return my_str == my_str[::-1]

def palindrome_reverse2(my_str):
    return my_str == ''.join(reversed(my_str))

def palindrome_recurs(my_str):
    if len(my_str) < 2:
        return True
    if my_str[0] != my_str[-1]:
        return False
    return palindrome(my_str[1:-1])