# _Soft Cluster Membership as an Embedding_

This application loads an image dataset, creates an embedding based on probability of cluster membership, adds a positional encoding and saves it as HDF5 files. 

### The details

The program first crops each image to a square, then divides the images into patches. Cluster centroids are then created using the K-means++ method from a subset of patches. Every patch from every image then has its cosine similarity to each centroid calculated. Softmax is then applied to give the probability of a given patch belonging to each of the centroids.

Positional encoding is traditional 2D sinusoidal (with an optional added image edge marker) that is summed with the embedding prior to saving.

It was originally created as a non-trained encoding/embedding for vision transformer experimentation.

Note that the saved HDF5 files contain the encoded images as a `data` variable, and target `labels`. Samples are stored column-wise, as most matrix libraries such as Eigen, Armadillo and Bandicoot also store column-wise.

Images are also saved that show the centroids generated, and an interpretation of each part of the positional embedding.

## Build Notes & Dependencies

The Cmake files are set up to work with Apple Silicon, Apple x86 and Linux. Tested as working on an Apple M2 Pro laptop, an Apple Intel Macbook Pro and Fedora (UNE Turing).

First git clone the repository. Because it is private you will need to use the git CLI (or more painfully, the git SSH approach). Or simply download the zip from the git website, if you just want to test it!

Once cloned, on both platforms you then create and activate a conda environment in which to compile the program. Run the commands below at your terminal, replacing `<ENV_NAME>` with a name of your choosing:

`conda create -n <ENV_NAME> -c conda-forge cmake llvm-openmp cxx-compiler c-compiler opencv hdf5 highfive armadillo`

`conda activate <ENV_NAME>`

Please be patient. For conda to get the data from the repository & solve the environment can take 5-10min - it isn't frozen. So go make a coffee!

## Compile & Run:

From a top-level `build` directory (you'll need to create one), run the following commands in the terminal:

`cmake ..`

`make`

Then set up the required dataset as below. Finally run the compiled program with:

`./kmeans_embed <SOURCE_DIR> <DEST_DIR>  <IMAGE_HEIGHT> <PATCH_HEIGHT> <KMEANS_CLUSTERS> <PATCH_PERCENTAGE_FOR_KMEANS> <IMAGES_PER_HDF5>`

For example:

`./kmeans_embed ../data/imagenette2-160 ../data  160 8 256 0.2 1000`

This would crop source images to 160x160 square, then divide into 8x8 patches, create 256 centroids from 20% of patches in the training files, and save 1000 images per HDF5 file, saving them to "../data".

On an Apple M2 Pro Macbook most processing takes around 2-20 minutes in total for Imagenette (all 10k training files & 4k validation files), depending on the arguments supplied. For the example above, it should take just under 4 minutes. For Imagenet-1K it is around 6 hours.

## Setting up the data:

This is tested against the Imagenette datasets (Full/320/160/Woof/Wang). It should work for anything in the same arrangement: `train` and `val` directories under the main folder, with subdirectories providing the category labels, then each subdirectory containing JPEG/JPG/PNG image files. Examples would be [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) or the Kaggle version of [Imagenet-1K](https://www.kaggle.com/datasets/sautkin/imagenet1k0).

To use Imagenette:

- Download one of the Imagenette datasets from [Github](https://github.com/fastai/imagenette). 
- Unzip & place it under a top-level `data` directory.
- Also create empty `train` and `val` directories under the `data` directory - the output HDF5 files are stored in these.

