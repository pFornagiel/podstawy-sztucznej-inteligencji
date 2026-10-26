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
    "check_task_9",
    "check_task_10",
    "check_task_11",
    "check_task_12",
    "check_task_13",
    "check_extra_task_1",
    "check_extra_task_2",
    "check_experiment",
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


# ---- tabular data (HIGGS) ---------------------------------------------------

# minimum test AUROC of the experiments in the notebook (with some margin for other random seeds)
_MIN_TEST_AUROC = {
    "MLP + dropout": 0.775,
}


def _not_implemented(*objects):
    # templates leave `...` in place of the solution
    if any(obj is Ellipsis for obj in objects):
        raise NotImplementedError


@_checker
def check_task_9(TabularDataset, train_loader):
    from torch.utils.data import DataLoader, Dataset, RandomSampler

    X = torch.arange(30.0).reshape(10, 3)
    y = torch.arange(10.0).unsqueeze(1)
    dataset = TabularDataset(X, y)
    assert isinstance(dataset, Dataset), "TabularDataset should inherit from torch.utils.data.Dataset"
    _not_implemented(dataset.__len__(), dataset[0], train_loader)
    assert dataset.__len__() == 10, "__len__ should return the number of examples, len(self.X)"
    item = dataset[4]
    assert isinstance(item, tuple) and len(item) == 2, "__getitem__(i) should return a pair (X[i], y[i])"
    assert torch.equal(item[0], X[4]) and torch.equal(item[1], y[4]), "__getitem__(i) should return a pair (X[i], y[i])"
    X_batch, y_batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert X_batch.shape == (4, 3) and y_batch.shape == (4, 1), "DataLoader should create batches of shape (4, 3) and (4, 1) from TabularDataset"

    assert isinstance(train_loader, DataLoader), "train_loader should be a DataLoader"
    assert isinstance(train_loader.dataset, TabularDataset), "create train_loader from TabularDataset(X_train_t, y_train_t)"
    assert len(train_loader.dataset) == 38400, "train_loader should use the training set (X_train_t, y_train_t)"
    assert train_loader.batch_size == 256, f"batch_size should be 256, got {train_loader.batch_size}"
    assert isinstance(train_loader.sampler, RandomSampler), "train_loader should shuffle the data before every epoch (shuffle=True)"


def _reference_epoch(model, loader, loss_fn, optimizer, zero_grad=True):
    model.train()
    total = 0.0
    for X_batch, y_batch in loader:
        if zero_grad:
            optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X_batch)
    return total / len(loader.dataset)


@_checker
def check_task_10(train_epoch):
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(70, 3, generator=torch.Generator().manual_seed(0))
    y = (X[:, :1] > 0).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=16)  # 5 batches, the last one has 6 points

    def run(epoch_fn, **kwargs):
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        loss = epoch_fn(model, loader, nn.BCEWithLogitsLoss(), optimizer, **kwargs)
        return loss, model.weight.detach().clone()

    loss, weight = run(train_epoch)
    ref_loss, ref_weight = run(_reference_epoch)
    _, no_zero_grad_weight = run(_reference_epoch, zero_grad=False)
    assert isinstance(loss, float), "train_epoch should return a Python float (use loss.item())"
    assert not torch.allclose(weight, run(lambda *args: 0.0)[1]), "the parameters did not change - call optimizer.step() for every batch"
    assert not torch.allclose(weight, no_zero_grad_weight, atol=1e-6), "gradients accumulate - call optimizer.zero_grad() before every backward()"
    assert torch.allclose(weight, ref_weight, atol=1e-5), "wrong parameters after the epoch - make one step (zero_grad, forward, backward, step) per batch"
    assert abs(loss - ref_loss) < 1e-5, "wrong mean cost - sum loss.item() * len(X_batch) and divide by len(loader.dataset)"


@_checker
def check_task_11(TabularMLP):
    model = TabularMLP(5, hidden_size=8, n_layers=2, dropout=0.1)
    types = [type(layer) for layer in model.layers]
    assert types == [nn.Linear, nn.ReLU, nn.Dropout] * 2 + [nn.Linear], \
        f"self.layers should be (Linear, ReLU, Dropout) x n_layers + Linear, got {[t.__name__ for t in types]}"
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in linears] == [(5, 8), (8, 8), (8, 1)], \
        "wrong layer sizes - n_features -> hidden_size -> ... -> hidden_size -> 1"
    assert model.layers[2].p == 0.1, "nn.Dropout should get the dropout argument"
    assert model(torch.randn(4, 5)).shape == (4, 1), "the model should map (N, n_features) to (N, 1)"
    assert sum(isinstance(layer, nn.Linear) for layer in TabularMLP(5, n_layers=4).layers) == 5, \
        "the number of hidden layers should be equal to n_layers"


def _reference_fit(model, optimizer, loader, X_val, y_val, epochs):
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        history["train_loss"].append(_reference_epoch(model, loader, loss_fn, optimizer))
        model.eval()
        with torch.no_grad():
            history["val_loss"].append(loss_fn(model(X_val), y_val).item())
    return history


@_checker
def check_task_12(fit):
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(0)
    X, X_val = torch.randn(64, 3, generator=generator), torch.randn(32, 3, generator=generator)
    y, y_val = (X[:, :1] > 0).float(), (X_val[:, :1] > 0).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    def run(fit_fn):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 8), nn.Dropout(0.5), nn.Linear(8, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        return fit_fn(model, optimizer, loader, X_val, y_val, 3)

    history = run(fit)
    _not_implemented(history)
    ref = run(_reference_fit)
    assert isinstance(history, dict) and {"train_loss", "val_loss"} <= set(history), \
        'fit should return a dict with the keys "train_loss" and "val_loss"'
    assert len(history["train_loss"]) == 3 and len(history["val_loss"]) == 3, "the history should have one value per epoch"
    assert all(isinstance(value, float) for value in history["train_loss"] + history["val_loss"]), \
        "store Python floats in the history (use .item())"
    assert abs(history["train_loss"][0] - ref["train_loss"][0]) < 1e-5, \
        "wrong training cost - call train_epoch(model, train_loader, loss_fn, optimizer) once per epoch and store its result"
    assert np.allclose(history["val_loss"], ref["val_loss"], atol=1e-5), \
        "wrong validation cost - compute loss_fn(model(X_val), y_val) after every epoch, in evaluation mode (model.eval())"
    assert np.allclose(history["train_loss"], ref["train_loss"], atol=1e-5), "wrong training costs in later epochs"


@_checker
def check_task_13(MyDropout):
    X = torch.ones(200, 500)
    dropout = MyDropout(p=0.3)
    dropout.eval()
    _not_implemented(dropout(X))
    assert torch.equal(dropout(X), X), "in evaluation mode (model.eval()) dropout should return X unchanged"
    dropout.train()
    out = dropout(X)
    assert out.shape == X.shape, "the output should have the same shape as X"
    zeros = (out == 0).float().mean().item()
    assert abs(zeros - 0.3) < 0.02, f"about 30% of values should be zeroed for p=0.3, got {zeros:.1%}"
    assert torch.allclose(out[out != 0], torch.tensor(1 / 0.7)), "the remaining values should be divided by (1 - p)"
    assert not torch.equal(out, dropout(X)), "the mask should be random and different in every call"
    zero_p = MyDropout(p=0.0)
    zero_p.train()
    assert torch.equal(zero_p(X), X), "for p=0 dropout should return X unchanged"


@_checker
def check_extra_task_1(ResidualBlock, TabularResNet):
    block = ResidualBlock(8, dropout=0.1)
    types = [type(layer) for layer in block.block]
    assert types == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear], \
        f"self.block should be Linear, ReLU, Dropout, Linear, got {[t.__name__ for t in types]}"
    assert block.block[0].in_features == 8 and block.block[3].out_features == 8, "both Linear layers should be hidden_size -> hidden_size"
    block.eval()
    X = torch.randn(4, 8)
    with torch.no_grad():
        out = block(X)
        assert not torch.allclose(out, block.block(X)), "forward should add the input to the output of the block (skip connection)"
        assert torch.allclose(out, X + block.block(X)), "forward should return X + self.block(X)"

    model = TabularResNet(5, hidden_size=8, n_blocks=3, dropout=0.1)
    types = [type(layer) for layer in model.layers]
    assert types == [nn.Linear] + [ResidualBlock] * 3 + [nn.ReLU, nn.Linear], \
        f"self.layers should be Linear, n_blocks blocks, ReLU, Linear, got {[t.__name__ for t in types]}"
    assert (model.layers[0].in_features, model.layers[0].out_features) == (5, 8), "the first layer should be n_features -> hidden_size"
    assert model.layers[-1].out_features == 1, "the output layer should return one logit"
    assert model(torch.randn(4, 5)).shape == (4, 1), "the model should map (N, n_features) to (N, 1)"


@_checker
def check_extra_task_2(proba_onnx, model, X):
    _not_implemented(proba_onnx)
    proba_onnx = np.asarray(proba_onnx)
    assert proba_onnx.shape == (len(X),), \
        f"proba_onnx should have shape ({len(X)},) - take the first output of session.run and squeeze it, got {proba_onnx.shape}"
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(X)).squeeze(1).numpy()
    assert np.allclose(proba_onnx, expected, atol=1e-5), \
        'the probabilities differ from PyTorch - run session.run(None, {"features": X_test_t.numpy()})'


@_checker
def check_experiment(results, name):
    assert name in results, f'"{name}" not found in results - run the experiment first'
    auroc = results[name]["test_auroc"]
    assert auroc >= _MIN_TEST_AUROC[name], f"test AUROC of {name} is {auroc:.3f}, expected at least {_MIN_TEST_AUROC[name]}"
