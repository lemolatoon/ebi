## plotting

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ cd tools 
$ pip install -r requirements.txt
```

### Time-Series and Non-Time-Series Dataset
```
$ python3 plot2.py GENERAL_DATASET_JSON_PATH XOR_SYNTHETIC_JSON_FILE_PATH UCR2018_DIR EMBEDDING_JSON_PATH
# e.g
$ python3 001.json xor_001.json ucr2018_result embedding_result.json
```
Will get figures under `001/` (the directory name will be the file stem of GENERAL_DATASET_JSON_PATH)

### Xor Synthetic Dataset
```
$ python3 plot_xor.py XOR_SYNTHETIC_JSON_FILE_PATH
# e.g
$ python3 plot_xor.py xor_001.json
```
Will get figures under `xor_001/` (the directory name will be the file stem of XOR_SYNTHETIC_JSON_FILE_PATH)

### UCR2018 Dataset
```
$ python3 plot_xor.py UCR2018_DIR
# e.g
$ python3 plot_xor.py 003
```
Will get figures under `results/ucr2018` (constant path)

### Embeddings Dataset
```
$ python3 embedding_plot.py
# e.g
$ python3 embedding_plot.py embedding_result.json
```
Will get figures under `results/embedding` (contant path)