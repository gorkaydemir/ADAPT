# ADAPT - Dataset Preprocessing

This data preprocessing/formation is built on preprocessing of [DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT) and variation of the one used in [FTGN](https://github.com/gorkaydemir/FTGN).


1\. Download [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html). After extracting, dataset structure should be: 
```
argoverse
├── train
|   └── data
|       ├── 1.csv
|       ├── 2.csv
|       ├── ...
└── val
    └── data
        ├── 1.csv
        ├── 2.csv
        ├── ...
└── test
    └── data
        ├── 1.csv
        ├── 2.csv
        ├── ...
```

2\. Install [Argoverse 1 API](https://github.com/argoai/argoverse-api).


3\. You can create the preprocessed data file, given:
- `/path/to/train/root`, path to raw train data,
- `/path/to/val/root`, path to raw validation data,
- `/path/to/data`, path to (preprocessed) output data parent directory,

you can create train data as `ex_list`:
```
python dataset/preprocess_data.py --data_dir /path/to/train/root \
--output_dir /path/to/data
```
validation data as `eval.ex_list`:
```
python dataset/preprocess_data.py --data_dir /path/to/val/root \
--output_dir /path/to/data --validation
```

After that, you can extend the training data by providing the extracted ex_list path:
```
python dataset/extend_dataset.py --data_dir /path/to/data/ex_list \
--output_dir /path/to/data
```

After preprocessing, dataset structure should be: 
```
/path/to/data/
├── ex_list
├── eval.ex_list
└── extended_ex_list
```