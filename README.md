# FPSniff

Implementation of Posterior-Annealed Conditional Diffusion.

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
## License
MIT License.
