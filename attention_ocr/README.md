# attention ocr

requirements

- install the tensorflo library
  ```
  python3 -m venv ~/.tensorflow
  source ~/.tensorflow/bin/active
  pip install --upgrade pip
  pip install --upgrede tensorflow-gpu=1.15
  ```
- requirement disk 158GB free disk to download fsns dataset
  ```
  cd attention_ocr/python/datasets
  aria2c -c j 20 ../../../stree/python/fsns_urls.txt
  cd ..
  ```
- 16GB of ram or more; 32 gb is recommended


## dataset

the french street name sign dataset is split into subsets, each of wich is composed of multiple files. Note that the datasets are very large. the approximate size are
- Train 512 files of 300MB each
- validation 64 file of 40MB each
- test 64 files of 50MB each
- the datasets download includes a directory ``testdata`` that contains some small datasets that are big enough to test that models can actually learn something