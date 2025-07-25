# Компьютерные методы обработки изображений
- **Первый семестр**: обработка сигналов и "традиционные" методы обработки изображений
- **Второй семестр**: машинное обучение для обработки изображений

# Установка
Для выполнения домашних заданий вам потребуется установить некоторые библиотеки
## venv + pip
1. В папке курса рекомендуется создать виртуальное окружение:
```
python -m venv .venv
```
2. Чтобы активировать виртуальное окружение, нужно выполнить одну из команд:

  - Windows (cmd.exe): `.venv\Scripts\activate.bat`
  - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
  - POSIX: `source .venv/bin/activate`

3. После того, как окружение активировано, установите библиотеки из `requirements.txt`:
```
pip install -r requirements.txt
```
Больше подробностей: [docs](https://docs.python.org/3/library/venv.html)

<details>
<summary> <h2>Poetry (опционально)</h2> </summary>
  В качестве альтернативы для управления зависимостями можно воспользоваться библиотекой Poetry.
  
  Для этого необходимо:
  - Установить Poetry: `pip install poetry`
  - Создать виртуальное окружение для проекта (например, в [VS Code](https://code.visualstudio.com/docs/python/environments), в [PyCharm](https://www.jetbrains.com/help/pycharm/poetry.html) или как описано в пункте "venv + pip")
  - Выполнить `poetry install` 
  
  При установке Poetry использует файл `pyproject.toml`, который содержит информацию об используемых зависимостях, линтере/форматтере и т.д. 
  
  Если не создавать виртуальное окружение вручную, то Poetry создаст его за вас (см. [docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment))

</details>

## Jupyter Notebooks
Работать с Jupyter Notebook можно в разных средах и используя разные инструменты, например:
- Работа с `.ipynb` в [VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Installing Jupyter](https://jupyter.org/install) (JupyterLab или Jupyter Notebook)
- [Google Colaboratory](https://colab.google/)

# Тестирование
Перед отправкой заданий на проверку настоятельно рекомендуется воспользоваться тестами из папки `tests` для проверки правильности реализации функций из заданий.

Для тестирования всех функций из всех заданий необходимо выполнить 
```
pytest image_processing_<НОМЕР СЕМЕСТРА>\tests
```
Для проверки конкретного задания нужно указать соответствующий файл с тестами, например:
```
pytest image_processing_1\tests\test_filtering.py
```
**_Обратите внимание_**, что успешно пройденные тесты сами по себе не означают правильно выполненное задание!
