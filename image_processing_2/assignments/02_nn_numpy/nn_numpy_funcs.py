import numpy as np


def linear(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    get_grad: bool = False,
    upstream_grad: np.ndarray = None,
) -> np.ndarray:
    """Выполняет линейное преобразование (полносвязный слой).

    Args:
        x (np.ndarray): Входные данные, размера (batch_size, input_dim).
        W (np.ndarray): Матрица весов, размера (input_dim, output_dim).
        b (np.ndarray): Вектор смещений, размера (1, output_dim).
        get_grad (bool, optional): Если True, вычисляет и возвращает градиенты по x, W и b.
        upstream_grad (np.ndarray, optional): Градиент от следующего слоя, размера (batch_size, output_dim).

    Returns:
        np.ndarray: Если get_grad == False, возвращает результат линейного преобразования (batch_size, output_dim).
        tuple: Если get_grad == True, возвращает кортеж (dx, dW, db), где:
            dx (np.ndarray): Градиент по x, размера (batch_size, input_dim).
            dW (np.ndarray): Градиент по W, размера (input_dim, output_dim).
            db (np.ndarray): Градиент по b, размера (output_dim,).

    """
    pass


def relu(x: np.ndarray, get_grad: bool = False, upstream_grad: np.ndarray = None) -> np.ndarray:
    """Функция активации ReLU (Rectified Linear Unit).

    Args:
        x (np.ndarray): Входной массив, размера (batch_size, ...)
        get_grad (bool, optional): Если True, возвращает градиент функции ReLU по x
        upstream_grad (np.ndarray, optional): Градиент от следующего слоя.
            Необходим при вычислении градиента (того же размера, что и x)

    Returns:
        np.ndarray: Если get_grad == False, возвращает результат действия ReLU
                    Если get_grad == True, возвращает градиент по x

    """
    pass


def softmax_ce(y_true: np.ndarray, x: np.ndarray, get_grad: bool = False) -> np.ndarray:
    """Вычисляет softmax, объединенный с кросс-энтропией (softmax + cross-entropy loss) и её градиент.

    Args:
        y_true (np.ndarray): Массив истинных меток в one-hot представлении, размера (batch_size, num_classes).
        x (np.ndarray): Массив логитов (выходов последнего линейного слоя), размера (batch_size, num_classes).
        get_grad (bool, optional): Если True, возвращает градиент по x.

    Returns:
        np.ndarray:
            - Если get_grad == False: массив значений функции потерь для каждого объекта в батче, размер (batch_size,).
            - Если get_grad == True: массив градиентов по x, форма (batch_size, num_classes).

    """
    pass


def init_weights(input_size: int, hidden_size: int, output_size: int):
    """Инициализирует веса и смещения для двухслойной нейронной сети.

    Args:
        input_size (int): Размер входного слоя (число признаков).
        hidden_size (int): Размер скрытого слоя.
        output_size (int): Размер выходного слоя (число классов).

    Returns:
        tuple:
            W1 (np.ndarray): Матрица весов первого слоя, размера (input_size, hidden_size).
            b1 (np.ndarray): Вектор смещений первого слоя, размера (1, hidden_size).
            W2 (np.ndarray): Матрица весов второго слоя, размера (hidden_size, output_size).
            b2 (np.ndarray): Вектор смещений второго слоя, размера (1, output_size,).

    """
    pass


def get_batches(x_data: np.ndarray, y_data: np.ndarray, batch_size: int):
    """Разбивает данные на батчи заданного размера.

    Args:
        x_data (np.ndarray): Массив входных данных, размера (num_samples, ...).
        y_data (np.ndarray): Массив меток, размера (num_samples, ...).
        batch_size (int): Размер батча.

    Returns:
        tuple:
            x_batches (list[np.ndarray]): Список батчей входных данных.
            y_batches (list[np.ndarray]): Список батчей меток.

    """
    pass


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Вычисляет метрику accuracy между предсказанными и истинными метками.

    Аргументы:
        y_pred (np.ndarray): Предсказанные метки, размера (batch_size,)
        y_true (np.ndarray): Истинные метки, размера (batch_size,)

    Возвращает:
        float: Значение accuracy.
    """
    pass
