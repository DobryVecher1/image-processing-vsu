import importlib

import numpy as np
import pytest


@pytest.fixture(scope="session")
def module_path(request):
    """Получает путь к модулю, переданный через CLI."""
    module_path = request.config.getoption("--module_path")
    # if no path were provided - use default
    if module_path is None:
        module_path = "assignments.01_filtering.filtering_funcs"

    # convert slash to dot in order to be able to import module later
    return module_path.replace("\\", ".")


@pytest.fixture(scope="module")
def tested_module(module_path):
    """Импортирует модуль."""
    tested_module = importlib.import_module(module_path)
    return tested_module


# ===================
# Class with tests
# ===================
class TestFiltering:
    """Класс с тестами для filtering_funcs."""

    def test_get_expand(self, tested_module):
        """Проверяет размер выходного изображения."""
        img = np.ones((5, 10))
        img_expand = tested_module.get_expand_img(img)

        assert img_expand.shape == (img.shape[0] * 2, img.shape[1] * 2), "Неправильный размер изображения на выходе"

    def test_gauss_kernel(self, tested_module):
        """Проверяет правильность реализации гауссовского ядра."""
        test_kern = np.array(
            [
                [0.05854983, 0.09653235, 0.05854983],
                [0.09653235, 0.15915494, 0.09653235],
                [0.05854983, 0.09653235, 0.05854983],
            ]
        )
        kern = tested_module.gauss_kernel(kern_size=(3, 3), sigma=1.0)

        assert np.allclose(test_kern, kern), "Неправильные значения для гауссовского ядра"

    def test_get_filtered_image(self, tested_module):
        """Проверяет параметры изображения после фильтрации."""
        img = np.ones((34, 28))
        kern = np.ones_like(img)

        img_filt = tested_module.get_filtered_image(img, kern)

        assert img_filt.shape == (img.shape[0] // 2, img.shape[1] // 2), "Неправильный размер изображения на выходе"
        assert isinstance(img_filt[0, 0], float), "Значения должны быть float"

    def test_high_pass_kernel(self, tested_module):
        """Проверяет реализацию ФВЧ."""
        kern_high = tested_module.high_pass_kernel(kern_size=(101, 51), cutoff=10, order=2)
        min_ind = np.unravel_index(kern_high.argmin(), kern_high.shape)

        assert min_ind == (kern_high.shape[0] // 2, kern_high.shape[1] // 2), "АЧХ фильтра нецентрирована"

    def test_median_filter(self, tested_module):
        """Проверяет реализацию медианного фильтра."""
        img = np.ones((5, 10))
        img_filt = tested_module.median_filter(img, kern_size=(3, 3))

        assert img.shape == (img_filt.shape[0], img_filt.shape[1]), "Размер изображений не совпадает"
