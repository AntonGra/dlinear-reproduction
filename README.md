# D2: DLinear

Групповой проект по курсу Deep Learning по статье
[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
(Zeng et al., AAAI 2023).

Команда: Горбенко Антон Русланович, Рыбалкин Максим Иванович.

Планируем воспроизвести сравнение DLinear с трансформерами на ETT, проверить
влияние `drop_last` и способа усреднения ошибки, а также длины входного окна.
[Условия проекта](https://github.com/xufana/dl-knad-course/blob/main/project.md).

Пока выполнен пробный запуск DLinear на ETTh1: вход 336 часов, прогноз 96 часов,
7 переменных. Получены MSE **0.3841444** и MAE **0.4047131**. Сравнение с
трансформерами ещё не проводилось.

[Результаты](RESULTS.md) · [Журнал решений](DECISIONS.md) · [Лог обучения](runs/pilot-01/train.log)

## Установка

Проверено на macOS ARM64 с Python 3.9.6 и PyTorch 1.9.0.
`requirements-macos-arm64.lock` содержит версии и хеши пакетов для этой платформы.
Для Linux/Colab потребуется отдельное окружение; там запуск пока не проверялся.

Авторский код подключён в `upstream/` как submodule, commit
`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`. Исходники не менялись.

```sh
git clone --recurse-submodules https://github.com/AntonGra/dlinear-reproduction.git
cd dlinear-reproduction
python3.9 -m venv .venv
mkdir -p downloads tmp
TMPDIR="$PWD/tmp" .venv/bin/python -m pip --isolated download \
  --index-url https://pypi.org/simple --only-binary=:all: --require-hashes \
  --dest downloads -r requirements-macos-arm64.lock
TMPDIR="$PWD/tmp" .venv/bin/python -m pip --isolated install \
  --no-index --find-links downloads --only-binary=:all: --require-hashes \
  -r requirements-macos-arm64.lock
.venv/bin/python -m pip check
.venv/bin/python scripts/acquire.py data
```

Команды рассчитаны на новую копию проекта с установленным Python 3.9.
Если репозиторий уже скачан без submodule: `git submodule update --init upstream`.

ETTh1 загружается из [репозитория ETT](https://github.com/zhouhaoyi/ETDataset)
по фиксированному коммиту. Скрипт проверяет хеши из
[records/data-source.json](records/data-source.json) и сохраняет лицензию данных.

## Запуск

```sh
.venv/bin/python -B scripts/run.py etth1-01
.venv/bin/python -B scripts/verify.py etth1-01
```

Для каждого запуска нужно новое имя. В `runs/etth1-01/` сохраняются лог,
команда, версии пакетов, время, checkpoint и прогнозы. `verify.py` пересчитывает
метрики по CSV и проверяет, что checkpoint воспроизводит сохранённые прогнозы.

[Конфигурация](configs/etth1-336-96.json) взята из первой команды авторского
`scripts/EXP-LongForecasting/Linear/etth1.sh`: seed 2021, batch size 32,
learning rate 0.005, максимум 10 эпох, early stopping с patience 3.
Для macOS установлен `num_workers=0` вместо 10. Обучение идёт на CPU
с двумя потоками; лимит процесса 15 минут.

В репозитории оставлены лог, версии окружения и проверка первого запуска.
Веса, прогнозы и датасет создаются локально, поэтому `verify.py` нужно вызывать
для своего запуска, а не для опубликованного `pilot-01`.

## Использование LLM

Codex использовался для технической подготовки и проверки пробного запуска: настройки окружения, скриптов запуска и оформления результатов. Реализация модели взята из авторского репозитория без изменений.

Claude использовался для разбора кода LTSF-Linear и FEDformer, ноутбука для Colab и черновика pre-registration. Подробная декларация: [PREREGISTRATION.md](PREREGISTRATION.md), раздел 8.
