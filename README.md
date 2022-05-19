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

You can download all preprocessed data in [Google Cloud Drive](https://drive.google.com/drive/folders/1Fy9be_udTY2i23gcxk9rWFbFPvnNE0b-?usp=sharing)
**************************************************************

## Training
```
bash train_gmnmt.sh
```
For detailed script settings, please refer to [doc](https://fairseq.readthedocs.io/en/latest/index.html)<br>

Before training you have to modify [doc](https://github.com/nlpdl/fairseq/blob/fairseq/fairseq/tasks/gmnmt_task.py) lines 74 75 79 80 81 of the file to change the resource reference path.<br>
The 79th and 81st is the redundancy of modifying the model. Just use the files I provided and have no effect on training.
****************************************************************

## Evaluation
```
bash decode_gmnmt.sh
```
Just need to change the pair of path `x`. For detailed script settings, please refer to [doc](https://fairseq.readthedocs.io/en/latest/index.html)<br>

If you want to evaluate its bleu value, you can run
```
bash tiqu_o.sh
```
In the same way, you only need to change the pair of paths `x`.
*******************************************************************
