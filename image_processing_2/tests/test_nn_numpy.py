import importlib

import numpy as np
import pytest


@pytest.fixture(scope="session")
def module_path(request):
    """Получает путь к модулю, переданный через CLI."""
    module_path = request.config.getoption("--module_path")
    # if no path were provided - use default
    if module_path is None:
        module_path = "assignments.02_nn_numpy.nn_numpy_funcs"

    # convert slash to dot in order to be able to import module later
    return module_path.replace("\\", ".").removesuffix(".py")


@pytest.fixture(scope="module")
def tested_module(module_path):
    """Импортирует модуль."""
    tested_module = importlib.import_module(module_path)
    return tested_module


# ===================
# Class with tests
# ===================
class TestNN:
    """Класс с тестами для nn_numpy_funcs."""

    def test_relu_forward(self, tested_module):
        """Проверяет прямое прохождение функции ReLU."""
        x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
        result = tested_module.relu(x)
        expected = np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
        assert np.allclose(result, expected), "ReLU: Неверный результат forward pass"

    def test_relu_gradient(self, tested_module):
        """Проверяет вычисление градиента функции ReLU."""
        x = np.array([[-1, 2], [0, -3]])
        upstream_grad = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = tested_module.relu(x, upstream_grad=upstream_grad, get_grad=True)
        expected = np.array([[0, 0.2], [0, 0]])
        assert np.allclose(result, expected), "ReLU: Неверный результат backward pass"

    def test_relu_dtype_preservation(self, tested_module):
        """Проверяет сохранение типа данных."""
        x = np.array([-1, 0, 1]).astype(float)
        result = tested_module.relu(x)
        assert result.dtype == float, "ReLU: тип данных не сохранен"

        # Тест для градиента
        upstream_grad = np.array([0.1, 0.2, 0.3], dtype=float)
        result_grad = tested_module.relu(x, upstream_grad=upstream_grad, get_grad=True)
        assert result_grad.dtype == float, "ReLU: тип данных градиента не сохранен"

    def test_linear_forward_batch(self, tested_module):
        """Проверяет forward pass на батче."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        W = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = np.array([[0.5, -0.5]])  # (1, 2)

        out = tested_module.linear(x, W, b)
        expected = np.array([[7.5, 9.5], [15.5, 21.5]])

        assert np.allclose(out, expected), "Linear: Неверный результат forward pass"

    def test_backward_batch(self, tested_module):
        """Проверяет backward pass на батче."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        W = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = np.array([[0.5, -0.5]])  # (1, 2)
        upstream_grad = np.array([[1.0, -1.0], [2.0, 3.0]])  # (2, 2)

        dx, dW, db = tested_module.linear(x, W, b, get_grad=True, upstream_grad=upstream_grad)

        expected_dx = np.array([[-1.0, -1.0], [8.0, 18.0]])
        expected_dW = np.array([[7.0, 8.0], [10.0, 10.0]])
        expected_db = np.array([3.0, 2.0])

        assert np.allclose(dx, expected_dx, atol=1e-7), "Linear: Неверный результат backward pass dx"
        assert np.allclose(dW, expected_dW, atol=1e-7), "Linear: Неверный результат backward pass dw"
        assert np.allclose(db, expected_db, atol=1e-7), "Linear: Неверный результат backward pass db"

    def test_backward_shapes(self, tested_module):
        """Проверяет shape градиентов."""
        batch_size, in_dim, out_dim = 5, 3, 4
        x = np.random.randn(batch_size, in_dim)
        W = np.random.randn(in_dim, out_dim)
        b = np.random.randn(1, out_dim)
        upstream_grad = np.random.randn(batch_size, out_dim)

        dx, dW, db = tested_module.linear(x, W, b, get_grad=True, upstream_grad=upstream_grad)

        assert dx.shape == (batch_size, in_dim), "Linear: shape градиентов неверный"
        assert dW.shape == (in_dim, out_dim), "Linear: shape градиентов неверный"
        assert db.shape == (out_dim,), "Linear: shape градиентов неверный"

    def test_loss_shape(self, tested_module):
        """Проверяет shape лосса."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        x = np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        loss = tested_module.softmax_ce(y_true, x)
        assert loss.shape == (2,)

    def test_softmax_ce_grad_shape(self, tested_module):
        """Проверяет shape градиента softmax + cross entropy."""
        y_true = np.array([[1, 0, 0]])
        x = np.array([[2.0, 1.0, 0.1]])
        grad = tested_module.softmax_ce(y_true, x, get_grad=True)
        assert grad.shape == x.shape

    def test_perfect_prediction(self, tested_module):
        """Проверяет значения лосса на очень "уверенном" предсказании."""
        y_true = np.array([[1, 0, 0]])
        x = np.array([[15.0, 0.0, -5.0]])
        loss = tested_module.softmax_ce(y_true, x)[0]
        assert loss < 1e-5

    def test_wrong_prediction(self, tested_module):
        """Проверяет лосс при ошибочном предсказании."""
        y_true = np.array([[1, 0, 0]])
        x = np.array([[-5.0, 0.0, 10.0]])
        loss = tested_module.softmax_ce(y_true, x)[0]
        assert loss > 5.0

    def test_weights_shapes(self, tested_module):
        """Проверяет размеры инициализированных матриц."""
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 4, 5, 3
        W1, b1, W2, b2 = tested_module.init_weights(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        assert W1.shape == (INPUT_SIZE, HIDDEN_SIZE)
        assert b1.shape == (1, HIDDEN_SIZE)
        assert W2.shape == (HIDDEN_SIZE, OUTPUT_SIZE)
        assert b2.shape == (1, OUTPUT_SIZE)

    def test_get_batches_output_types(self, tested_module):
        """Проверяет типы выходных данных функции get_batches."""
        x_data = np.random.randn(10, 5)
        y_data = np.random.randint(0, 3, size=(10,))

        x_batches, y_batches = tested_module.get_batches(x_data, y_data, batch_size=3)
        # Проверяем, что x_batches - это список
        assert isinstance(x_batches, list), "Тип x_batches должен быть list"
        # Проверяем, что y_batches - это список
        assert isinstance(y_batches, list), "Тип y_batches должен быть list"

    def test_get_batches_batch_size(self, tested_module):
        """Проверяет размер батча."""
        x_data = np.random.randn(15, 10)
        y_data = np.random.randint(0, 3, size=(15,))

        x_batches, y_batches = tested_module.get_batches(x_data, y_data, batch_size=4)

        # Потенциально некоторые/последний батчи могут быть другого размера
        assert x_batches[0].shape[0] == 4
        assert y_batches[0].shape[0] == 4

    def test_accuracy_all_correct(self, tested_module):
        """Проверяет значение метрики accuracy."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        assert np.isclose(tested_module.accuracy(y_pred, y_true), 1.0)

    def test_accuracy_partially_correct(self, tested_module):
        """Проверяет значение метрики accuracy."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([4, 2, 2, 0])
        assert np.isclose(tested_module.accuracy(y_pred, y_true), 0.25)
