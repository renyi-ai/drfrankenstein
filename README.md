# Dr. Frankenstein - Similarity and Matching of Neural Network Representations

This repo provides a PyTorch implementation of the Dr. Frankenstein model stitching framework used to  perform the experiments described in the paper [Similarity and Matching of Neural Network Representations](https://arxiv.org/abs/2110.14633).

![Task Loss Matching](https://user-images.githubusercontent.com/9435563/144592445-d25cd928-e98a-4f26-8cd6-6d4129bcca16.png)

 * We stitch trained neural networks together to study the similarity of their latent representations. (This technique was first used by [Lenc and Vedaldi, 2015].) Our starting point is the observation that stitching can be done with minor performance loss when the two networks share the same training datasets and network topologies.  
 * We study the relation between this functional similarity and previously studied geometric notions of representational similarity such as CKA and CCA. We observe that their behavior can be counter-intuitive from the perspective of functional similarity. 
 * We experiment with restricting the space of transformations in the stitching layer (to e.g., orthogonal, low rank, sparse) to investigate how information is organised in the latent representations.



## Installation

### Conda
```console
conda env create -f environment.yml
```

### Pip
```console
pip install -r requirements.txt
```
*Note: PyTorch is not included in requirements, you need to install that manually. We used version 1.8 in our experiments.*

### Docker
If you prefer docker, you can also pull a prebuilt container:
```console
cd container
./pull.sh
cd ..
```

After you can run any command inside container like:
```console
cd container
./docker_run.sh -g 0 -c "python python_file.py"
```
* **-g** Which GPU to use for training, list GPU IDs here
* **-c** the command to run inside the container

The config file needs to be set properly. 
As you can see in [docer_run.sh](./container/docker_run.sh) the ```/home/${USER}/cache``` folder is mapped to 
the ```/cache``` folder inside, so it is recommended to store your data in 
```/home/${USER}/cache/data/pytorch``` and ```/home/${USER}/cache/data/celeba``` folders and leave the config file 
with the default settings.

You must create the data folder outside the container like:
```console
mkdir /home/${USER}/cache
```

## Setup

There's a config file which tells the script where it can find or download the datasets to. Please edit `config/default.env`:
```bash
[dataset_root]
pytorch = '/cache/data/pytorch' # path to pytroch datasets such as cifar10
celeba = '/cache/data/celeba'   # path to celeba dataset
```

## Train a neural net

You can train your own networks as below. Some pretrained models are uploaded to *model_weights/* folder.

Example:

```console
python train_model.py -m tiny10 -d cifar10
```

## Stitch neural nets

```console
python stitch_nets.py path/to/model1.pt /path/to/model2.pt layer1 layer2 -d dataset
```
An example stitch with pretrained models.
```console
python stitch_nets.py model_weights/Tiny10/CIFAR10/in0-gn0/110000.pt model_weights/Tiny10/CIFAR10/in10-gn10/110000.pt bn3 bn3 -d cifar10 -i ps_inv
```

#### Settings

* **-h, --help** Get help about parameters
* **--run-name** The name of the subfolder to save to. If not given, it defaults to the current date-time.
* **-e, --epochs** Number of epochs to train. Default: 30
* **-lr, --lr** Learning rate. Default: 1e-3
* **-o, --out-dir** Folder to save networks to. Default: snapshots/
* **-b, --batch-size** Batch size. Default: 128
* **-s, --save-frequency** How often to save the transformation matrix in iterations. This number is multiplied by the
  number of epochs. Default: 10
* **--seed** Seed of the run. Default: 0
* **-wd, --weight-decay** Weight decay to use. Deault: 1e-4
* **--optimizer** Name of the optimizer. Please choose from: adam, sgd. Default: adam
* **--debug** Either to run in debug mode or not. Default: False
* **--flatten** Either to flatten layers around transformation. NOTE: not used in the paper, hardly ever used, it might
  be buggy. Default: False
* **--l1** l1 regularization used on transformation matrix. Default: 0
* **--cka-reg** CKA regularisation used on transformation matrix. Default: 0
* **-r, --low-rank** Maximum rank of matrix. Use max rank by default.
* **-i, --init** Initialisation of transformation matrix. Options:
    * random: random initialisation. Default.
    * perm: random permutation
    * eye: identity matrix
    * ps_inv: pseudo inverse initialisation
    * ones-zeros: weight matrix is all 1, bias is all 0.
* **-m, --mask** Any mask applied on transformation. Options:
    * identity: All values are 1 in mask. Default.
    * semi-match: Based on correlation choose the best pairs.
    * abs-semi-match: Semi-match between absolute correlations.
    * random-permuation: A random permutation matrix.
* **--target-type** The loss to apply at logits. Options:
    * hard: Use true labels. Default.
    * soft_1: Use soft crossentropy loss to model1.
    * soft_2: Use soft crossentropy loss to model2.
    * soft_12: Use soft crossentropy loss to the mean of model1 and model2.
    * soft_1_plus_2: Use soft crossentropy loss to the sum of model1 and model2.
* **--temperature** The temperature to use if target type is a soft label. Default: 1.

You will find the results of your runs under *results/* folder by default, and a pickle file that contains all information about your run. E.g. the bias & weights of the stitching layer, accuracy, crossentropy, etc. 

## Layer information

Print layer information of the architecture, one can stitch between the printed layers

```console
python layer_info.py model_name
```

Handled model_names: lenet, tiny10, nbn_tiny10, nbntiny10, dense, inceptionv1, resnet20_w* 

Example:

```console
python layer_info.py resnet20_w3
```


## Evaluation/Comparison

### Compare stitched representations with Model2's

```console
python compare_frank_m2.py path/to/file.pkl stitch_type measure1 measure2 measure3 ..
```
Stitching types:
 * before - initial state of transformation matrix before training
 * after - trained transformation matrix
 * ps_inv - use pseudo inverse transformation (calculated on validation set)

Example: 
```console
python compare_frank_m2.py results/stitching_result.pkl after cka
```

### Compare trained models

```console
python compare_nets.py path/to/model1.pt /path/to/model2.pt layer1 layer2 dataset method1 method2 method3 ..
```
Example: 

```console
python compare_nets.py model_weights/Tiny10/CIFAR10/in0-gn0/110000.pt model_weights/Tiny10/CIFAR10/in10-gn10/110000.pt bn5 bn5 cifar10 cka l2
```

### Evaluate on ground truth labels

To evaluate a trained network:

```console
python eval_net.py path/to/model.pt
```

Example:

```console
python eval_net.py model_weights/Tiny10/CIFAR10/in0-gn0/110000.pt
```

To evaluate a stitched network:

```console
python eval_stitch.py results/stitching_result.pkl stitch_type
```
Stitching types:
 * before - initial state of transformation matrix before training
 * after - trained transformation matrix
 * ps_inv - use pseudo inverse transformation (calculated on validation set)
