# D2 — DLinear: Are Transformers Effective for Time Series Forecasting?

Групповой проект по курсу Deep Learning. Подготовлен проверенный технический пилот для pre-registration.

## Команда

- Горбенко Антон Русланович.
- Рыбалкин Максим Иванович.

## Краткое содержание

Проект посвящён воспроизведению результатов статьи **Are Transformers Effective for Time Series Forecasting?** (Zeng et al., AAAI 2023). Планируем проверить, сохраняется ли преимущество простой линейной модели DLinear перед трансформерными моделями при долгосрочном прогнозировании временных рядов на выбранных наборах данных ETT. Сравнение проведём по метрикам MSE и MAE. Отдельно исследуем влияние неполных тестовых батчей (`drop_last`) и способа усреднения ошибки по горизонту прогноза. В качестве расширения проверим, как длина входного окна влияет на соотношение качества моделей.

Технический пилот выполнен: DLinear на ETTh1, вход 336 часов → прогноз 96 часов,
7 переменных. Авторские исходники не изменены. Два запуска с seed 2021 дали
побитово одинаковые прогнозы. Это проверка запуска, а не полное воспроизведение
сравнения с трансформерами. Документ pre-registration ещё не включён.

| Результат | Значение |
|---|---:|
| Test MSE | 0.3841443955898285 |
| Test MAE | 0.40471312403678894 |
| Покрытие теста | 2785 из 2785 окон |
| Эпохи / лучший checkpoint | 7 / эпоха 4 |
| Вычисления | CPU Apple M4 Pro, 2 потока |
| Время процесса | 32.30 с первый запуск; 5.02 с повторный |

[Полные результаты и ограничения](RESULTS.md) · [Журнал решений](DECISIONS.md) ·
[План, записанный до запуска](BEFORE_RUN.md) · [Лог](runs/pilot-01/train.log) ·
[Независимая проверка](runs/pilot-01/verification.json)

## Получить код

Авторская реализация подключена как git submodule по точному коммиту
`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`, с исходной лицензией Apache-2.0.
Клонируйте в новую папку:

```sh
git clone --recurse-submodules https://github.com/AntonGra/dlinear-reproduction.git
cd dlinear-reproduction
```

Если репозиторий уже клонирован без submodule:

```sh
git submodule update --init upstream
```

## Подготовить отдельное окружение

Проверенная среда: **Python 3.9, macOS ARM64**, PyTorch 1.9.0.
Lock содержит хеши конкретных macOS wheels; для Linux/Colab нужен другой lock.
Историческая среда статьи использовала Python 3.6.9; это отличие документировано.
Команды ниже предназначены для новой копии без существующей `.venv`.
Нужен уже установленный Python 3.9 (`python3.9` или другой путь к этой версии).
На исходном Mac использовался `/usr/bin/python3` версии 3.9.6.

```sh
python3.9 -m venv .venv
mkdir -p downloads tmp .cache/matplotlib .cache/pip
TMPDIR="$PWD/tmp" .venv/bin/python -m pip --isolated download \
  --index-url https://pypi.org/simple --only-binary=:all: --require-hashes \
  --dest downloads -r requirements-macos-arm64.lock
TMPDIR="$PWD/tmp" .venv/bin/python -m pip --isolated install \
  --no-index --find-links downloads --only-binary=:all: --require-hashes \
  -r requirements-macos-arm64.lock
.venv/bin/python -m pip check
.venv/bin/python scripts/acquire.py data
```

Данные скачиваются из официального `zhouhaoyi/ETDataset` по закреплённому коммиту;
хеши проверяются по [манифесту](records/data-source.json).
Лицензия и README создателей данных сохраняются рядом с CSV.
Пакеты поступают только из PyPI, установка выполняется из локальных wheels.

## Обучить и проверить

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B scripts/run.py my-run-01
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B scripts/verify.py my-run-01
```

Для нового запуска используйте новое имя. Существующая папка запуска не
перезаписывается. Обёртка вызывает оригинальный `upstream/run_longExp.py`,
сохраняет команду, версии, хеши и время, ограничивает CPU двумя потоками
и время процесса 15 минутами. Новые результаты остаются локально.

[Конфигурация](configs/etth1-336-96.json) соответствует первой команде авторского
`scripts/EXP-LongForecasting/Linear/etth1.sh`. Из условий выполнения изменены
`num_workers=0` вместо 10, CPU и пути вывода. Гиперпараметры модели/обучения
не менялись: seed=2021, batch_size=32, learning_rate=0.005, максимум 10 эпох,
early stopping patience=3, Adam, MSE.

В `runs/my-run-01/` появятся `train.log`, `metadata.json`, `environment.txt`,
`verification.json`, checkpoint, `pred.npy` и авторские PDF-графики.
Проверка независимо восстанавливает цели и нормализацию из CSV, пересчитывает
MSE/MAE и проверяет прогнозы по локально обученному checkpoint.
Чужие checkpoint-файлы не скачиваются.

## Что сохранено в репозитории

В `runs/pilot-01/` и `runs/pilot-02/` находятся небольшие текстовые свидетельства
выполненных запусков; [проверка повторяемости](records/repeatability.json) подтверждает
совпадение прогнозов. Абсолютные локальные пути заменены на `${PROJECT_ROOT}`;
числа и хеши не менялись. Веса, массивы прогнозов, данные, `.venv`, wheels и кэши
не публикуются — их можно получить повторным запуском. Поэтому проверять нужно
свой новый запуск, а не вызывать `verify.py pilot-01` без локальных артефактов.

Один seed не заменяет минимум три сида для основного результата исследования.
Результат пока не доказывает преимущество DLinear перед трансформерами.
Метрики посчитаны в стандартизованном пространстве по всему горизонту и всем каналам.

## Источники

- [Задание курса и тема D2](https://github.com/xufana/dl-knad-course/blob/main/project.md).
- [Статья](https://arxiv.org/abs/2205.13504).
- [Авторский код](https://github.com/cure-lab/LTSF-Linear).
- [Создатели данных ETT](https://github.com/zhouhaoyi/ETDataset).

## Использование LLM

Codex использовался для технической подготовки и проверки пробного запуска: настройки окружения, скриптов запуска и оформления результатов. Реализация модели взята из авторского репозитория без изменений.
