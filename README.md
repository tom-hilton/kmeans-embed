# _Vector Quantization with K-Means_

This application loads an image dataset, vector quantizes with K-Means and saves it as HDF5 files. It was originally created as a minimal encoding/embedding to make iterating  novel vision transformer techniques extremely fast The k-dimensional tree approach is used to speed up the quantization process.

Note that the saved HDF5 files contain the encoded images as a `data` variable, and target `labels`. Samples are stored column-wise, as most matrix libraries such as Eigen, Armadillo and Bandicoot also store column-wise.

## Build Notes & Dependencies

The Cmake files are set up to work with Apple Silicon, Apple x86 and Linux. Tested as working on an Apple M2 Pro laptop, an Apple Intel Macbook Pro and Fedora (UNE Turing).

First git clone the repository. Because it is private you will need to use the git CLI (or more painfully, the git SSH approach). Or simply download the zip from the git website, if you just want to test it!

Once cloned, on both platforms you then create and activate a conda environment in which to compile the program. Run the commands below at your terminal, replacing `<ENV_NAME>` with a name of your choosing:

`conda create -n <ENV_NAME> -c conda-forge cmake llvm-openmp cxx-compiler c-compiler opencv hdf5 highfive`

`conda activate <ENV_NAME>`

Please be patient. For conda to get the data from the repository & solve the environment can take 5-10min - it isn't frozen. So go make a coffee!

## Compile & Run:

From a top-level `build` directory (you'll need to create one), run the following commands in the terminal:

`cmake ..`

`make`

Then set up the required dataset as below. Finally run the compiled program with:

`./vector_quant <SOURCE_DIR> <DEST_DIR>  <IMAGE_HEIGHT> <PATCH_HEIGHT> <KMEANS_CLUSTERS> <PATCH_PERCENTAGE_FOR_KMEANS> <IMAGES_PER_HDF5>`

For example:

`./vector_quant ../data/imagenette2-160 ../data  160 6 144 0.05 1000`

On an Apple M2 Pro Macbook this takes around 90 sec total for all 10k training files & 4k validation files.

## Setting up the data:

This is tested against the Imagenette datasets (Full/320/160/Woof/Wang). It should work for anything in the same arrangement: `train` and `val` directories under the main folder, with subdirectories providing the category labels, then each subdirectory containing JPEG image files.

- Download one of the Imagenette datasets from [Github](https://github.com/fastai/imagenette). Unzip & place it under a top-level `data` directory.
- Also create empty `train` and `val` directories under the `data` directory - the output HDF5 files are stored in these.

