# Snapshot-Ensemble-Colorectal-Cancer
Based on our paper "SnapEnsemFS: A Snapshot Ensembling-based Deep Feature Selection Model for Colorectal Cancer Histological Analysis" under review in Scientific Reports, Nature.

# Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

# Running the codes:
Required directory structure:

```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- PSO.py
+-- __init__.py
+-- main.py
+-- model.py

```
Then, run the code using the command prompt as follows:

`python main.py --data_directory "data"`

Available arguments:
- `--epochs`: Number of epochs of training. Default = 100
- `--learning_rate`: Learning Rate. Default = 0.0002
- `--batch_size`: Batch Size. Default = 4
- `--momentum`: Momentum. Default = 0.9
- `--num_cycles`: Number of cycles. Default = 5
