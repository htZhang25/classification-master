# classification-pytorch

Currently Tested on pytorch 1.0.1 with cuda9 and python3.7.

# Installation

```
pip install -r requirements.txt
```

# Usage

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path /path/to/model --test-manifest /path/to/test_manifest.mat --save-path /path/to/result.txt
```

