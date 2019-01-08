# rankboostplus
Implementations of various ranking-by-boosting algorithms including Rankboost+. Documentation can be found [here](https://rail-cwru.github.io/rankboostplus/).

## Installation
NumPy, SciPy, and Python 3.5 (as type hinting is used) are required for this project. To install, run `python setup.py install` from the base directory.

## Usage
A simple example can be seen below, where `y.shape = (M, 2)`, and `x.shape = [N,D]`, and each `y[i,0], y[i,1]` is a critical pair with `x[y[i, 0]]` ranked higher than `x[y[i,1]]`. 

```python
r = RBD(max_iter)
r.fit(x, y)
cumulative_predictions = r.predict_cumulative(x, y) # (y.shape[0], max_iter)
preds = r.predic(x, y) # (y.shape[0], )
````
