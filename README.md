# EBI
Float Compressions and Original File Format for databases

Currently under developments.

## Reporoducing the results

## About SIMD
Configure target cpu by setting envvars when building:

AVX2:
```bash
$ RUSTFLAGS="-C target-cpu=haswell" TARGET_CPU=haswell cargo build --release
```
AVX512: (if your CPU supports)
```bash
$ RUSTFLAGS="-C target-cpu=native" TARGET_CPU=native cargo build --release
```

### [De]Compression and Queries
Ensure that your datasets are in the `DATA_DICT`
```bash
# create binary files from csv
$ cargo run --bin csv2bin --release -- "DATA_DICT/*"
$ cd experimenter
# create directory for saving experiment checkpoints
$ mkdir save compressor_configs data/filter_config
$ cargo run --release --bin experimenter -- all -c compressor_configs --exact-precision -f DATA_DICT/filter_config -b DATA_DICT/binary --create-config --n 10 --in-memory -s save
```
Results will be saved in `DATA_DICT/result`

### 1-NN on UCRArchive_2018
```bash
$ cd experimenter
$ cargo run --release -- -i DATA_DICT/UCRArchive_2018/ ucr2018 -o DATA_DICT
```
Results will be saved in `DATA_DICT/result`

### Matrix Multiplication with CUDA
```bash
$ cd experimenter
$ cargo run --release --features=cuda -- matrix-cuda -o OUTDIR
```

### embeddings
```bash
$ cd experimenter
$ cargo run --release -- -i DATA_DICT embedding -o DATA_DICT
```
Results will be saved in `DATA_DICT/result`

### TPC-H
```bash
$ cd tools
$ uv run gen-tpch.py # generate files on tools/tpch_data/
$ cd experimenter
$ cargo run --release -- tpch -i DATA_DIR -o OUT_DIR
```
Results will be saved in `OUT_DIR/tpch`

### Plotting
```bash
$ cd tools
$ python3 plot.py DATA_DICT/result/000.json
$ python3 ucr2018_plot.py DATA_DICT/result/ucr2018/000
$ python3 embedding_plot.py DATA_DICT/result/embedding/000/embedding_result.json
$ uv run matrix_plot2.py OUTDIR/result/matrix_cuda/000 -cuda
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributions

By contributing to this project, you agree that your contributions will be licensed under the MIT License, unless you explicitly state otherwise in your submission.