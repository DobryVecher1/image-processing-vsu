import numpy as np
from assignments.filtering.filtering_funcs import (
    gauss_kernel,
    get_expand_img,
    get_filtered_image,
    high_pass_kernel,
    median_filter,
)


def test_get_expand():
    """Проверяет размер выходного изображения."""
    img = np.ones((5, 10))
    img_expand = get_expand_img(img)

    assert img_expand.shape == (img.shape[0] * 2, img.shape[1] * 2), "Неправильный размер изображения на выходе"


def test_gauss_kernel():
    """Проверяет правильность реализации гауссовского ядра."""
    test_kern = np.array(
        [
            [0.05854983, 0.09653235, 0.05854983],
            [0.09653235, 0.15915494, 0.09653235],
            [0.05854983, 0.09653235, 0.05854983],
        ]
    )
    kern = gauss_kernel(kern_size=(3, 3), sigma=1.0)

    assert np.allclose(test_kern, kern), "Неправильные значения для гауссовского ядра"


def test_get_filtered_image():
    """Проверяет параметры изображения после фильтрации."""
    img = np.ones((34, 28))
    kern = np.ones_like(img)

    img_filt = get_filtered_image(img, kern)

    assert img_filt.shape == (img.shape[0] // 2, img.shape[1] // 2), "Неправильный размер изображения на выходе"
    assert isinstance(img_filt[0, 0], float), "Значения должны быть float"


def test_high_pass_kernel():
    """Проверяет реализацию ФВЧ."""
    kern_high = high_pass_kernel(kern_size=(101, 51), cutoff=10, order=2)
    min_ind = np.unravel_index(kern_high.argmin(), kern_high.shape)

    assert min_ind == (kern_high.shape[0] // 2, kern_high.shape[1] // 2), "АЧХ фильтра нецентрирована"


def test_median_filter():
    """Проверяет реализацию медианного фильтра."""
    img = np.ones((5, 10))
    img_filt = median_filter(img, kern_size=(3, 3))

    assert img.shape == (img_filt.shape[0], img_filt.shape[1]), "Размер изображений не совпадает"
