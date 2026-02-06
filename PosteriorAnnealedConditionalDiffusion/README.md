# Posterior-Annealed Conditional Diffusion

Minimal PyTorch implementation of a posterior-annealed conditional diffusion model for research use.

## Requirements
Python 3.8+, PyTorch, torchvision, numpy, tqdm, pillow, scikit-image, opencv-python  
```bash
pip install -r requirements.txt
```

## Usage
```bash
python MainCondition.py --config configs/default.json --state train
python MainCondition.py --config configs/default.json --state eval --test_load_weight <checkpoint>
```

## Data
Training data is not included. Users should provide their own datasets and update paths in the config file.

## License
MIT License.
