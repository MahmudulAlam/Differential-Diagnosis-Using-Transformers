## Dataset

DDXPlus is a dataset of synthetically generated 1.3M patient information that contains patient details (e.g. age, sex), evidence, ground truth differential diagnoses, and condition. The dataset has a total of 49 pathologies that cover various age groups, sexes, and patients with a broad spectrum of medical history.

[![GitHub](https://img.shields.io/badge/GitHub-ddxplus-black.svg?style=popout-flat&logo=github)](https://github.com/bruzwen/ddxplus)
[![ArXiv](https://img.shields.io/badge/ArXiv-ddxplus-red.svg?style=popout-flat&logo=arxiv)](https://arxiv.org/abs/2205.09148)

[Download](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374) the dataset and extract the files in the ```data/``` directory. For more detail information, please check the [paper](https://arxiv.org/abs/2205.09148).

Data folder should contain: 

```
.
├── data/                   
    ├── release_conditions.json          # conditions metadata
    ├── release_evidences.json           # evidences metadata
    ├── release_train_patients.csv       # training data
    ├── release_validate_patients.csv    # validation data
    ├── release_test_patients.csv        # test data 
    └── README.md 
```