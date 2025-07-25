import numpy as np
from assignments.false_colors.false_colors_funcs import get_color_map, get_false_color_img, get_levels


def test_get_levels():
    """Проверяет get_levels."""
    num_intervals = 5
    levels = get_levels(num_intervals=num_intervals)

    assert levels.dtype == int, "Значения должны быть int"
    assert len(levels) == num_intervals + 1, "Неправильное количество границ интервалов"


def test_get_color_map():
    """Проверяет get_color_map."""
    num_intervals = 5
    color_map = get_color_map(num_intervals=num_intervals)

    assert color_map.shape == (num_intervals, 3), "Неправильный размер"


def test_get_false_color_img():
    """Проверяет get_false_color_img."""
    levels = get_levels(num_intervals=5)
    color_map = get_color_map(num_intervals=5)
    img = np.random.randint(low=0, high=256, size=(25, 10))

    img_color = get_false_color_img(img, levels, color_map)

    assert img_color.shape == (25, 10, 3)
