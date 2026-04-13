# Raw Data And DVC

Каталог `data/raw/` предназначен для исходных обезличенных датасетов.

Правило работы:

- сырые данные не коммитятся в Git напрямую;
- сами файлы датасетов отслеживаются через DVC;
- в Git попадают `.dvc`-файлы и конфигурация DVC;
- реальные датасеты должны храниться в remote storage DVC.

Типовой сценарий:

```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
```

После этого файл `data/raw/dataset.csv` останется локально, но в Git будет храниться только файл-описание `data/raw/dataset.csv.dvc`.
