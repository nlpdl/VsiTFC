The project is based on [fairseq](https://github.com/facebookresearch/fairseq). 
For more detailed settings, please refer to [doc](https://fairseq.readthedocs.io/en/latest/index.html)

## Requirements
* Ubuntu 20.0.4
* Python 3.7.0
* Pytorch 1.8.2+cu111
* fairseq 1.0.0a0+31d94f5

**************************************************************

## Preprocessing data
```
bash press_fr.sh
```
Remember to modify the `TEXT` and other fields in the file to process your dataset

**************************************************************

## Preprocessing Image
For the extraction of image features, please refer to [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch)<br>
The two relational matrices can be constructed by `get_matrix_dict.ipynb` and `get_matrix.ipynb`<br>

**************************************************************

## Training
```
bash train_gmnmt.sh
```
For detailed script settings, please refer to [doc](https://fairseq.readthedocs.io/en/latest/index.html)<br>

Before training you have to modify the file [doc](https://fairseq.readthedocs.io/en/latest/index.html)

****************************************************************
