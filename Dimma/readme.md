# Dimma: Few-shot low light image enhancing with adaptive dimming

To run this code install requirements 
```bash
pip install -r requirements.txt
```
and run one of the following commands:

```bash
python train_unsupervised.py
python train_supervised.py
python finetune.py
```
For different config file use --config flag. There are many configs in config folder.

Please, bear in mind that you need to first train unsupervised model before running finetune.py. Data and models are not included in this repository. You can get them from the following link: [drive](https://drive.google.com/drive/folders/1WTWZ34L_35FkKT-GhYNbj1W27KUS32ej?usp=share_link).