import mlflow
import numpy as np
from random import randint
from sklearn.linear_model import LinearRegression

def main():
    print("Running demo")    
    mlflow.sklearn.autolog()
    fit_model()

def fit_model():
    X = np.array([[randint(0, 4), 1], [1, 2], [2, 2], [2, randint(0, 4)]])
    y = np.dot(X, np.array([1, 2])) + 3
    print(f"X:{X}")
    print(f"y:{y}")
    model = LinearRegression()
    with mlflow.start_run() as run:
        model.fit(X, y)


if __name__ == '__main__':
    main()
