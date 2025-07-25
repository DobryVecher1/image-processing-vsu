import importlib

import numpy as np
import pytest


@pytest.fixture(scope="session")
def module_path(request):
    """Получает путь к модулю, переданный через CLI."""
    module_path = request.config.getoption("--module_path")
    # if no path were provided - use default
    if module_path is None:
        module_path = "assignments.03_false_colors.false_colors_funcs"

    # convert slash to dot in order to be able to import module later
    return module_path.replace("\\", ".")


# import the module
@pytest.fixture(scope="module")
def tested_module(module_path):
    """Импортирует модуль."""
    tested_module = importlib.import_module(module_path)
    return tested_module


# ===================
# Class with tests
# ===================
class TestFalseColors:
    """Класс с тестами для false_colors_funcs."""

    def test_get_levels(self, tested_module):
        """Проверяет get_levels."""
        num_intervals = 5
        levels = tested_module.get_levels(num_intervals=num_intervals)

        assert levels.dtype == int, "Значения должны быть int"
        assert len(levels) == num_intervals + 1, "Неправильное количество границ интервалов"

    def test_get_color_map(self, tested_module):
        """Проверяет get_color_map."""
        num_intervals = 5
        color_map = tested_module.get_color_map(num_intervals=num_intervals)

        assert color_map.shape == (num_intervals, 3), "Неправильный размер"

    def test_get_false_color_img(self, tested_module):
        """Проверяет get_false_color_img."""
        levels = tested_module.get_levels(num_intervals=5)
        color_map = tested_module.get_color_map(num_intervals=5)
        img = np.random.randint(low=0, high=256, size=(25, 10))

        img_color = tested_module.get_false_color_img(img, levels, color_map)

        assert img_color.shape == (25, 10, 3)
