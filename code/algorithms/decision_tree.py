
import csv
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Dict, NamedTuple, Any
from collections import Counter, defaultdict

# Calculate the Gini impurity for a list of values
# See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
def gini(data: List[Any]) -> float:
    counter: Counter = Counter(data)
    classes: List[Any] = list(counter.keys())
    num_items: int = len(data)
    result: float = 0
    item: Any
    for item in classes:
        p_i: float = counter[item] / num_items
        result += p_i * (1 - p_i)
    return result

assert gini(['one', 'one']) == 0
assert gini(['one', 'two']) == 0.5
assert gini(['one', 'two', 'one', 'two']) == 0.5
assert 0.8 < gini(['one', 'two', 'three', 'four', 'five']) < 0.81

# Helper function to filter down a list of data points by a `feature` and its `value`
def filter_by_feature(data_points: List[DataPoint], *args) -> List[DataPoint]:
    result: List[DataPoint] = deepcopy(data_points)
    for arg in args:
        feature: str = arg[0]
        value: Any = arg[1]
        result = [data_point for data_point in result if getattr(data_point, feature) == value]
    return result

assert len(filter_by_feature(data_points, ('outlook', 'sunny'))) == 5
assert len(filter_by_feature(data_points, ('outlook', 'sunny'), ('temp', 'mild'))) == 3
assert len(filter_by_feature(data_points, ('outlook', 'sunny'), ('temp', 'mild'), ('humidity', 'high'))) == 2

# Helper function to extract the values the `feature` in question can assume
def feature_values(data_points: List[DataPoint], feature: str) -> List[Any]:
    return list(set([getattr(dp, feature) for dp in data_points]))

assert feature_values(data_points, 'outlook').sort() == ['sunny', 'overcast', 'rainy'].sort()


# Calculate the weighted sum of the Gini impurities for the `feature` in question
def gini_for_feature(data_points: List[DataPoint], feature: str, label: str = 'play') -> float:
    total: int = len(data_points)
    # Distinct values the `feature` in question can assume
    dist_values: List[Any] = feature_values(data_points, feature)
    # Calculate all the Gini impurities for every possible value a `feature` can assume
    ginis: Dict[str, float] = defaultdict(float)
    ratios: Dict[str, float] = defaultdict(float)
    for value in dist_values:
        filtered: List[DataPoint] = filter_by_feature(data_points, (feature, value))
        labels: List[Any] = [getattr(dp, label) for dp in filtered]
        ginis[value] = gini(labels)
        # We use the ratio when we compute the weighted sum later on
        ratios[value] = len(labels) / total
    # Calculate the weighted sum of the `feature` in question
    weighted_sum: float = sum([ratios[key] * value for key, value in ginis.items()])
    return weighted_sum

assert 0.34 < gini_for_feature(data_points, 'outlook') < 0.35
assert 0.44 < gini_for_feature(data_points, 'temp') < 0.45
assert 0.36 < gini_for_feature(data_points, 'humidity') < 0.37
assert 0.42 < gini_for_feature(data_points, 'windy') < 0.43


# NOTE: We can't use type hinting here due to cyclic dependencies

# A `Node` has a `value` and optional out `Edge`s
class Node:
    def __init__(self, value):
        self._value = value
        self._edges = []

    def __repr__(self):
        if len(self._edges):
            return f'{self._value} --> {self._edges}'
        else:
            return f'{self._value}'
    
    @property
    def value(self):
        return self._value

    def add_edge(self, edge):
        self._edges.append(edge)
    
    def find_edge(self, value):
        return next(edge for edge in self._edges if edge.value == value)

# An `Edge` has a value and points to a `Node`
class Edge:
    def __init__(self, value):
        self._value = value
        self._node = None

    def __repr__(self):
        return f'{self._value} --> {self._node}'
    
    @property
    def value(self):
        return self._value
    
    @property
    def node(self):
        return self._node
    
    @node.setter
    def node(self, node):
        self._node = node
        
        
# Recursively build a tree via the CART algorithm based on our list of data points
def build_tree(data_points: List[DataPoint], features: List[str], label: str = 'play') -> Node:
    # Ensure that the `features` list doesn't include the `label`
    features.remove(label) if label in features else None

    # Compute the weighted Gini impurity for each `feature` given that we'd split the tree at the `feature` in question
    weighted_sums: Dict[str, float] = defaultdict(float)
    for feature in features:
        weighted_sums[feature] = gini_for_feature(data_points, feature)

    # If all the weighted Gini impurities are 0.0 we create a final `Node` (leaf) with the given `label`
    weighted_sum_vals: List[float] = list(weighted_sums.values())
    if (float(0) in weighted_sum_vals and len(set(weighted_sum_vals)) == 1):
        label = getattr(data_points[0], 'play')
        return Node(label)    
    
    # The `Node` with the most minimal weighted Gini impurity is the one we should use for splitting
    min_feature = min(weighted_sums, key=weighted_sums.get)
    node: Node = Node(min_feature)
        
    # Remove the `feature` we've processed from the list of `features` which still need to be processed
    reduced_features: List[str] = deepcopy(features)
    reduced_features.remove(min_feature)

    # Next up we build the `Edge`s which are the values our `min_feature` can assume
    for value in feature_values(data_points, min_feature):
        # Create a new `Edge` which contains a potential `value` of our `min_feature`
        edge: Edge = Edge(value)
        # Add the `Edge` to our `Node`
        node.add_edge(edge)
        # Filter down the data points we'll use next since we've just processed the set which includes our `min_feature`
        reduced_data_points: List[DataPoint] = filter_by_feature(data_points, (min_feature, value))
        # This `Edge` points to the new `Node` (subtree) we'll create through recursion
        edge.node = build_tree(reduced_data_points, reduced_features)

    # Return the `Node` (our `min_feature`)
    return node