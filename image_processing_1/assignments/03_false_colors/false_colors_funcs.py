import numpy as np


def get_levels(num_intervals: int) -> np.ndarray:
    """Возвращает массив с границами интервалов яркости.

    Args:
        num_intervals (int): количество интервалов

    Returns:
        np.ndarray: массив со значениями границ интервалов

    """
    pass


def get_color_map(num_intervals: int) -> np.ndarray:
    """Возвращает color map.

    Args:
        num_intervals (int): количество интервалов

    Returns:
        np.ndarray: color map

    """
    pass


def get_false_color_img(img: np.ndarray, levels: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """Присваивает пикселям grayscale изображения цвет в соответствии с color map.

    Args:
        img (np.ndarray): изображение
        levels (np.ndarray): границы интервалов яркости
        color_map (np.ndarray): color map

    Returns:
        np.ndarray: цветное изображение

    """
    pass
