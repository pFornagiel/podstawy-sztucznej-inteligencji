"""Checker functions for the lab_3.ipynb notebook. You do not need to read this file."""

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "check_task_1",
    "check_task_2",
    "check_task_3",
    "check_task_4",
    "check_task_5",
    "check_task_6",
    "check_task_7",
    "check_task_8",
    "check_mlp_training",
]


def _checker(fn):
    # prints the result of a check instead of raising an AssertionError
    def wrapper(*args):
        try:
            fn(*args)
        except NotImplementedError:
            print("⏳ Not implemented yet.")
        except AssertionError as e:
            print(f"❌ {e}")
        else:
            print("✅ Solution is correct!")

    return wrapper


def _linear_data(n=200):
    # linearly separable points, as in the notebook
    X = np.random.default_rng(0).uniform(-3, 3, size=(3 * n, 2))
    z = X @ np.array([1.5, -1.0]) + 0.5
    X, z = X[np.abs(z) > 0.6][:n], z[np.abs(z) > 0.6][:n]
    return X, (z > 0).astype(float)


@_checker
def check_task_1(analytical_df, df):
    x = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(analytical_df(x), 6 * x - 4), "analytical_df(x) should return 6x - 4"
    assert np.allclose(df(np.sin, x), np.cos(x), atol=1e-8), "df(np.sin, x) should be close to cos(x) - use the central difference"


@_checker
def check_task_2(sigmoid, LogisticRegression):
    assert np.allclose(sigmoid(np.array([-1.0, 0.0, 2.0])), [0.268941, 0.5, 0.880797]), "wrong sigmoid"
    model = LogisticRegression(2)
    assert np.shape(model.w) == (2,) and model.b == 0, "self.w should have shape (n_features,) and self.b should be 0"
    X = np.array([[1.0, 2.0], [-1.0, 0.5]])
    assert np.allclose(model.forward(X), sigmoid(X @ model.w + model.b)), "forward(X) should return sigmoid(X @ w + b)"


@_checker
def check_task_3(bce_per_sample, bce):
    p_hat, y = np.array([0.9, 0.2, 0.5]), np.array([1.0, 0.0, 1.0])
    costs = -np.log([0.9, 0.8, 0.5])
    assert np.allclose(bce_per_sample(p_hat, y), costs), "bce_per_sample should return the cost of every point"
    assert np.isclose(bce(p_hat, y), costs.mean()), "bce should return the mean cost"


@_checker
def check_task_4(backward, LogisticRegression):
    X, y = _linear_data(50)
    model = LogisticRegression(2)
    p_hat = 1 / (1 + np.exp(-(X @ model.w + model.b)))
    grad_w, grad_b = backward(model, X, y)
    assert np.allclose(grad_w, X.T @ (p_hat - y) / len(y), atol=1e-6), "wrong grad_w"
    assert np.isclose(grad_b, np.mean(p_hat - y), atol=1e-6), "wrong grad_b"


@_checker
def check_task_5(train, LogisticRegression):
    X, y = _linear_data()
    model = LogisticRegression(2)
    history = train(model, X, y, lr=0.5, epochs=100)
    assert len(history) == 100 and history[-1] < history[0], "train should return one cost per epoch, and the cost should decrease"
    assert np.all(model.predict(X) == y), "the trained model should classify all points correctly"


@_checker
def check_task_6(LogisticRegressionTorch):
    model = LogisticRegressionTorch(3)
    X = torch.randn(4, 3)
    assert isinstance(model.linear, nn.Linear) and model(X).shape == (4, 1), "the model should map (N, n_features) to (N, 1)"
    assert torch.allclose(model(X), torch.sigmoid(model.linear(X))), "forward should return sigmoid(linear(X))"


@_checker
def check_task_7(train_torch, LogisticRegressionTorch):
    X, y = _linear_data()
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    torch.manual_seed(0)
    model = LogisticRegressionTorch(2)
    history = train_torch(model, torch.optim.SGD(model.parameters(), lr=0.5), X, y, epochs=100)
    assert len(history) == 100 and isinstance(history[0], float), "train_torch should return one float per epoch (loss.item())"
    assert abs(history[-1] - 0.0411) < 1e-3, "the final cost should be about 0.041 - check zero_grad, backward and step"


@_checker
def check_task_8(MoonMLP):
    model = MoonMLP(hidden_size=4)
    assert [type(layer) for layer in model.layers] == [nn.Linear, nn.Tanh, nn.Linear, nn.Sigmoid] \
        and model.layers[0].out_features == 4, "self.layers should be Linear(2, hidden_size), Tanh, Linear(hidden_size, 1), Sigmoid"
    assert model(torch.randn(5, 2)).shape == (5, 1), "the model should map (N, 2) to (N, 1)"


@_checker
def check_mlp_training(model, X, y):
    with torch.no_grad():
        acc = ((model(X) >= 0.5).float() == y).float().mean().item()
    assert acc >= 0.97, f"accuracy on make_moons is {acc:.1%}, expected at least 97%"
