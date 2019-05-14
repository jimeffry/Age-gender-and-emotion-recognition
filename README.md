# Age and Gender Estimation
This is a Keras implementation of a CNN for estimating age and gender from a face image [1, 2].
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.


## Dependencies
- Python2.7+
- Keras2.0+
- scipy, numpy, Pandas, tqdm, tables, h5py
- OpenCV3

Tested on:
- Ubuntu 16.04, Python 3.5.2, Keras 2.0.3, Tensorflow(-gpu) 1.0.1,  CUDA 8.0, cuDNN 5.0
  - CPU: i7-7700 3.60GHz, GPU: GeForce GTX1080

## Usage
## Face detect
'''
here use opencv-face-detection. or you can use the mtcnn-caffemodel, 

I have packaged the mtcnn face model into a class, the link is [here](https://github.com/jimeffry/face-detect-recognize/tree/master/demo).
'''

### Use pretrained model
Download pretrained model weights for TensorFlow backend:

```
The age, gender and emotion pretrained models locate in the directory PRJ_ROOT/trained_models/
```

Run demo script (requires web cam)

```sh
python test.py
```

### Train a model using the IMDB-WIKI dataset

#### Download the dataset
The dataset is downloaded and extracted to the `data` directory.

```sh
./download.sh
```

#### Create training data
Filter out noise data and serialize images and labels for training into `.mat` file.
Please check [check_dataset.ipynb](check_dataset.ipynb) for the details of the dataset.
```sh
python create_db.py --output data/imdb_db.mat --db imdb --img_size 64
```

```sh
usage: create_db.py [-h] --output OUTPUT [--db DB] [--img_size IMG_SIZE] [--min_score MIN_SCORE]

This script cleans-up noisy labels and creates database for training.

optional arguments:
  -h, --help                 show this help message and exit
  --output OUTPUT, -o OUTPUT path to output database mat file (default: None)
  --db DB                    dataset; wiki or imdb (default: wiki)
  --img_size IMG_SIZE        output image size (default: 32)
  --min_score MIN_SCORE      minimum face_score (default: 1.0)
```

#### Train network
Train the network using the training data created above.

```sh
python train.py --input data/imdb_db.mat
```

Trained weight files are stored as `checkpoints/weights.*.hdf5` for each epoch if the validation loss becomes minimum over previous epochs.

```sh
usage: train.py [-h] --input INPUT [--batch_size BATCH_SIZE]
                [--nb_epochs NB_EPOCHS] [--depth DEPTH] [--width WIDTH]
                [--validation_split VALIDATION_SPLIT]

This script trains the CNN model for age and gender estimation.

optional arguments:
  -h, --help                          show this help message and exit
  --input INPUT, -i INPUT             path to input database mat file (default: None)
  --batch_size BATCH_SIZE             batch size (default: 32)
  --nb_epochs NB_EPOCHS               number of epochs (default: 30)
  --depth DEPTH                       depth of network (should be 10, 16, 22, 28, ...) (default: 16)
  --width WIDTH                       width of network (default: 8)
  --validation_split VALIDATION_SPLIT validation split ratio (default: 0.1)
```

#### Use the trained network

```sh
python demo.py
```

```sh
usage: demo.py [-h] [--weight_file WEIGHT_FILE] [--depth DEPTH] [--width WIDTH]

This script detects faces from web cam input, and estimates age and gender for
the detected faces.

optional arguments:
  -h, --help                show this help message and exit
  --weight_file WEIGHT_FILE path to weight file (e.g. weights.18-4.06.hdf5) (default: None)
  --depth DEPTH             depth of network (default: 16)
  --width WIDTH             width of network (default: 8)

```

Please use the best model among `checkpoints/weights.*.hdf5` for `WEIGHT_FILE` if you use your own trained models.

#### Plot training curves from history file

```sh
python plot_history.py --input models/history_16_8.h5 
```

<img src="https://github.com/yu4u/age-gender-estimation/wiki/images/loss.png" width="400px">



<img src="https://github.com/yu4u/age-gender-estimation/wiki/images/accuracy.png" width="400px">

## Network architecture
In [the original paper](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf) [1, 2], the pretrained VGG network is adopted.
Here the Wide Residual Network (WideResNet) is trained from scratch.
I modified the @asmith26's implementation of the WideResNet; two classification layers (for age and gender estimation) are added on the top of the WideResNet.

Note that while age and gender are independently estimated by different two CNNs in [1, 2], in my implementation, they are simultaneously estimated using a single CNN.


## Results
Trained on imdb, tested on wiki.
![](https://github.com/yu4u/age-gender-estimation/wiki/images/result.png)


## References
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image
without facial landmarks," IJCV, 2016.
