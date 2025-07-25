import numpy as np


def downsample_no_filter(img: np.ndarray) -> np.ndarray:
    """Уменьшает размер изображения без использования сглаживающего фильтра.

    Args:
        img (np.ndarray): изображение

    Returns:
        np.ndarray: уменьшенное изображение

    """
    pass


def get_pad_img(img: np.ndarray, zeros_num: int) -> np.ndarray:
    """Осуществляет процесс паддинга.

    Args:
        img (np.ndarray): изображение
        zeros_num (int): количество нулей, которое необходимо дополнить на границах изображения

    Returns:
        np.ndarray: изображение c нулями на границах

    """
    pass


def convolve_2d(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Вычисляет свертку изображения и фильтра.

    Args:
        img (np.ndarray): изображение
        kern (np.ndarray): ядро фильтра

    Returns:
        np.ndarray: результат свертки

    """
    # img_padded = get_pad_img(img, zeros_num=)
    pass


def gauss_kernel(kern_size: tuple, sigma: float) -> np.ndarray:
    """Возвращает гауссовское ядро.

    Args:
        kern_size (tuple): размер ядра вдоль осей x и y
        sigma (float): среднеквадратическое отклонение

    Returns:
        np.ndarray: гауссовское ядро

    """
    pass


def expand_img(img: np.ndarray, row_col_num: int) -> np.ndarray:
    """Добавляет нулевые строки и столбцы между отсчетами.

    Args:
        img (np.ndarray): изображение
        row_col_num (int): количество нулевых строк и столбцов

    Returns:
        np.ndarray: расширенное изображение

    """
    pass


def bilin_kernel(l_param: int):
    """Возвращает ядро фильтра билинейной интерполяции.

    Args:
        l_param (int): размер фильтра (одинаков вдоль обоих измерений)

    """
    pass
