
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/iris_mod.csv', index_col='Id')

X = df.drop('Species', axis=1)
y = df['Species']

label_dict = {'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2}

y = y.map(label_dict)

# set up a `Pipeline` that performs certain preprocessing steps only on the numerical features:

numeric_features = ['SepalLength[cm]', 'SepalWidth[cm]', 'PetalLength[cm]', 'PetalWidth[cm]']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('feature_extraction', PCA(n_components=2))])

# Above, we weren't interested in performing these preprocessing steps on the categorical feature(s); instead, we apply **different** preprocessing steps to the categorical variable like so:

categorical_features = ['Color_IMadeThisUp']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))])


# - Scikit-learn's `ColumnTransformer` now allows us to merge these 2 seperate preprocessing pipelines, which operate on different feature sets in our dataset:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# - As a result, we get a 4 dimensional feature array (design matrix) if we apply this preprocessor. What are these 4 columns?

temp = preprocessor.fit_transform(X)
temp.shape

# The preprocessor can now also be conveniently be used in a Scikit-learn pipeline as shown below:
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=0)

clf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier(p=3))])


clf_pipe.fit(X_train, y_train)
print(f'Test accuracy: {clf_pipe.score(X_test, y_test)*100}%')

## Displaying Pipelines

# More info here: https://scikit-learn.org/dev/auto_examples/miscellaneous/plot_pipeline_display.html#sphx-glr-auto-examples-miscellaneous-plot-pipeline-display-py

clf_pipe

from sklearn import set_config


set_config(display="diagram")
clf_pipe
