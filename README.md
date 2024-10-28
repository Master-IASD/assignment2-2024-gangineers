[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MaLr9PVg)
# DataLabAssignement2

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.

## Precision and Recall Calculation
To calculate the improved_precision_recall, you can use the following:
```bash
python improved_precision_recall.py path/to/real_images path/to/generated_images --conversion
```
e.g.
```bash
python improved_precision_recall.py data/MNIST/MNIST/raw/train-images-idx3-ubyte samples/ --conversion
```

The `--conversion` flag is used to convert the images to pngs and save them in a new dir called `real_images`.

If the images are already in png format, you can omit the `--conversion` flag

After running it once with the --conversion flag, you can run it again without the flag giving the path to real_images.

```bash
python improved_precision_recall.py real_images/ samples/
```

## FID Calculation
To calculate the FID `real_images` should be first generated.
Then you can use the following:
```bash
python -m pytorch_fid real_images/ samples/
```
