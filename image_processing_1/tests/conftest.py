def pytest_addoption(parser):
    """Добавляет аргумент, переданный через CLI."""
    parser.addoption("--module_path", action="store", default=None)
