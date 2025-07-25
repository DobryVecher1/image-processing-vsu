import importlib

import numpy as np
import pytest
import scipy


@pytest.fixture(scope="session")
def module_path(request):
    """Получает путь к модулю, переданный через CLI."""
    module_path = request.config.getoption("--module_path")
    # if no path were provided - use default
    if module_path is None:
        module_path = "assignments.02_resize.resize_funcs"

    # convert slash to dot in order to be able to import module later
    return module_path.replace("\\", ".").removesuffix(".py")


# import the module
@pytest.fixture(scope="module")
def tested_module(module_path):
    """Импортирует модуль."""
    tested_module = importlib.import_module(module_path)
    return tested_module


# ===================
# Class with tests
# ===================
class TestResize:
    """Класс с тестами для resize_funcs."""

    def test_downsample_no_filter(self, tested_module):
        """Проверяет downsampling без предварительной фильтрации."""
        img = np.ones((101, 151))
        img_small = tested_module.downsample_no_filter(img)

        assert img_small.shape == (img.shape[0] // 2, img.shape[1] // 2)

    def test_pad(self, tested_module):
        """Проверяет паддинг."""
        img_test = np.array([[2, 4, 2, 2], [0, 1, 0, 1], [2, 1, 0, 2], [5, 3, 1, 2]])
        img_pad = tested_module.get_pad_img(img_test, zeros_num=2)

        assert (img_pad == np.pad(img_test, pad_width=2)).all()

    def test_convolve(self, tested_module):
        """Проверяет реализацию вычисления свертки."""
        img = np.random.randint(low=0, high=10, size=(25, 10)).astype(float)
        kern = np.random.randint(low=0, high=5, size=(5, 5)).astype(float)

        img_filt = tested_module.convolve_2d(img, kern)
        img_scipy = scipy.signal.convolve2d(img, kern, mode="same")

        assert (img_scipy == img_filt).all(), "Неверный результат вычисления свертки"
        assert img_filt.shape == img.shape, "Размеры до и после свертки не совпадают"

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

    def test_bilin(self, tested_module):
        """Проверяет реализацию ядра билинейной интерполяции."""
        kern = tested_module.bilin_kernel(l_param=3)
        max_ind = np.unravel_index(kern.argmax(), kern.shape)

        assert max_ind == (kern.shape[0] // 2, kern.shape[1] // 2), "Ядро фильтра нецентрировано"
