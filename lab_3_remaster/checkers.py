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
    "check_task_14",
    "check_task_15",
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
    "MLP + early stopping": 0.76,
    "+ dropout": 0.775,
    "+ LR scheduler": 0.78,
    "ResNet": 0.78,
    "ResNet + batch norm": 0.79,
}


def _not_implemented(*objects):
    # templates leave `...` in place of the solution
    if any(obj is Ellipsis for obj in objects):
        raise NotImplementedError


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
def check_task_9(train_epoch):
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
def check_task_10(TabularMLP):
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


@_checker
def check_task_11(EarlyStopping):
    model = nn.Linear(1, 1)
    early_stopping = EarlyStopping(patience=3)
    losses = [1.0, 0.8, 0.9, 0.85, 0.7, 0.75, 0.76, 0.77]
    stops = []
    for epoch, val_loss in enumerate(losses):
        with torch.no_grad():
            model.weight.fill_(epoch)  # in-place change of the weights, like optimizer.step()
        stops.append(early_stopping.step(val_loss, model))
    assert stops == [False] * 7 + [True], \
        f"step should return True after `patience` epochs without improvement, got {stops} for costs {losses}"
    assert early_stopping.best_loss == 0.7 and early_stopping.best_epoch == 4, \
        f"best_loss and best_epoch should be 0.7 and 4, got {early_stopping.best_loss} and {early_stopping.best_epoch}"
    early_stopping.restore_best(model)
    assert model.weight.item() == 4, (
        f"after restore_best the model has the weights from epoch {model.weight.item():.0f}, not from the best epoch 4 - "
        "store a copy in step (copy.deepcopy(model.state_dict())) and load it in restore_best (model.load_state_dict)"
    )


@_checker
def check_task_12(MyDropout):
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
def check_task_13(scheduler, histories):
    _not_implemented(scheduler)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau), "scheduler should be optim.lr_scheduler.ReduceLROnPlateau"
    assert scheduler.factor == 0.5 and scheduler.patience == 2, "use factor=0.5 and patience=2"
    assert "+ LR scheduler" in histories, 'train the model with run_experiment("+ LR scheduler", ...)'
    lrs = histories["+ LR scheduler"]["lr"]
    assert min(lrs) < lrs[0], "the learning rate never decreased - pass the scheduler to run_experiment"


@_checker
def check_task_14(ResidualBlock, TabularResNet):
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

    class OtherBlock(ResidualBlock):
        pass

    assert isinstance(TabularResNet(5, 8, 2, block=OtherBlock).layers[1], OtherBlock), "create the blocks with the `block` argument: block(hidden_size, dropout)"


@_checker
def check_task_15(NormResidualBlock, ResidualBlock):
    block = NormResidualBlock(8, dropout=0.0)
    assert isinstance(block, ResidualBlock) and hasattr(block, "block"), "call super().__init__(hidden_size, dropout) to create self.block"
    _not_implemented(getattr(block, "norm", None))
    assert isinstance(block.norm, nn.BatchNorm1d) and block.norm.num_features == 8, "self.norm should be nn.BatchNorm1d(hidden_size)"
    X = torch.randn(16, 8) * 3 + 1
    with torch.no_grad():
        for mode in (block.train, block.eval):
            mode()
            out = block(X)
            assert not torch.allclose(out, X + block.block(X)), "normalize the input of the branch: X + self.block(self.norm(X))"
            assert torch.allclose(out, X + block.block(block.norm(X)), atol=1e-6), "forward should return X + self.block(self.norm(X))"


@_checker
def check_experiment(results, name):
    assert name in results, f'"{name}" not found in results - run the experiment first'
    auroc = results[name]["test_auroc"]
    assert auroc >= _MIN_TEST_AUROC[name], f"test AUROC of {name} is {auroc:.3f}, expected at least {_MIN_TEST_AUROC[name]}"
