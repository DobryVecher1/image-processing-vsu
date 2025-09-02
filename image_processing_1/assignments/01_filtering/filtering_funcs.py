import numpy as np


def get_expand_img(img: np.ndarray) -> np.ndarray:
    """Дополняет изображение нулями.

    Args:
        img (np.ndarray): изображение

    Returns:
        np.ndarray: изображение, дополненное нулями

    """
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


def get_filtered_image(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Осуществляет фильтрацию изображения в частотной области.

    Args:
        img (np.ndarray): изображение
        kern (np.ndarray): ядро фильтра, используемого для фильтрации

    Returns:
        np.ndarray: отфильтрованное изображение

    """
    pass


def high_pass_kernel(kern_size: tuple, cutoff: float, order: int) -> np.ndarray:
    """Возвращает ядро фильтра Баттерворта в частотной области.

    Args:
        kern_size (tuple): размер фильтра
        cutoff (float): частота среза
        order (int): порядок фильтра

    Returns:
        np.ndarray: ядро фильтра

    """
    pass


def get_img_with_impulse_noise(img: np.ndarray, ratio: float) -> np.ndarray:
    """Добавляет импульсный шум на картинку.

    Args:
        img (np.ndarray): изображение
        ratio (float): доля пикселей, подвергающихся изменению

    Returns:
        np.ndarray: изображение с шумом

    """
    pass


def median_filter(img: np.ndarray, kern_size: tuple) -> np.ndarray:
    """Производит фильтрацию с помощью медианного фильтра.

    Args:
        img (np.ndarray): изображение
        kern_size (tuple): размер ядра

    Returns:
        np.ndarray: отфильтрованное изображение

    """
    pass
