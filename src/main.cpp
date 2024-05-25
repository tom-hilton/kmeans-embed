#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <highfive/H5Easy.hpp>
#include <random>
#include <vector>
#include <map>
#include <algorithm>

namespace fs = std::filesystem;
using namespace HighFive;

// Function to extract patches from an image
std::vector<cv::Mat> extractPatches(const cv::Mat& img, int patchSize) {
    std::vector<cv::Mat> patches;
    for (int y = 0; y <= img.rows - patchSize; y += patchSize) {
        for (int x = 0; x <= img.cols - patchSize; x += patchSize) {
            patches.push_back(img(cv::Rect(x, y, patchSize, patchSize)).clone());
        }
    }
    return patches;
}

// Function to randomly select a percentage of patches
std::vector<cv::Mat> selectRandomPatches(const std::vector<cv::Mat>& patches, float percentage) {
    std::vector<cv::Mat> selectedPatches;
    std::sample(patches.begin(), patches.end(), std::back_inserter(selectedPatches),
                static_cast<size_t>(patches.size() * percentage), std::mt19937{std::random_device{}()});
    return selectedPatches;
}

// Function to flatten patches
cv::Mat flattenPatches( const std::vector<cv::Mat>& patches) {
    cv::Mat flatPatches(patches.size(), patches[0].total(), patches[0].type());
    for (int i = 0; i < patches.size(); ++i) {
        cv::Mat patchRow(1, patches[i].total(), patches[i].type());
        patches[i].reshape(3, 1).copyTo(patchRow);
        patchRow.copyTo(flatPatches.row(i));
    }
    return flatPatches;
}

// Function to convert cv::Mat to std::vector for H5Easy
template <typename T>
std::vector<T> matToVector(const cv::Mat& mat) {
    return std::vector<T>(mat.begin<T>(), mat.end<T>());
}

// Function to convert cv::Mat to std::vector<std::vector<T>> for H5Easy
template <typename T>
std::vector<std::vector<T>> matToVector2D(const cv::Mat& mat) {
    std::vector<std::vector<T>> vec2D;
    for (int i = 0; i < mat.rows; ++i) {
        vec2D.push_back(matToVector<T>(mat.row(i)));
    }
    return vec2D;
}

// Function to save HDF5 file using H5Easy
void saveHDF5(const std::string& filename, const cv::Mat& data, const cv::Mat& labels) {
    // Create HDF5 file if it doesn't exist
    H5Easy::File file(filename, H5Easy::File::Overwrite);

    if (data.type() == CV_8U) {
        H5Easy::dump(file, "data", matToVector2D<uint8_t>(data));
    } else if (data.type() == CV_32F) {
        H5Easy::dump(file, "data", matToVector2D<float>(data));
    } else if (data.type() == CV_64F) {
        H5Easy::dump(file, "data", matToVector2D<double>(data));
    } else {
        std::cout << "Unsupported cv::Mat type to save: " << data.type() << std::endl;
        throw std::runtime_error("Unsupported cv::Mat type");
    }
    H5Easy::dump(file, "labels", matToVector2D<float>(labels));
}

// Function to create a tiled PNG from k-means centroids
void saveCentroidsAsImage(const cv::Mat& centers, int patchHeight, const std::string& filename) {
    int numClusters = centers.rows;
    int tileSize = static_cast<int>(std::sqrt(numClusters));
    int imageSize = tileSize * patchHeight;
    cv::Mat image(imageSize, imageSize, CV_32FC3);
    int idx = 0;
    for (int y = 0; y < imageSize; y += patchHeight) {
        for (int x = 0; x < imageSize; x += patchHeight) {
            if (idx < numClusters) {
                cv::Mat patch = centers.row(idx).reshape(3, patchHeight);
                patch.copyTo(image(cv::Rect(x, y, patchHeight, patchHeight)));
                ++idx;
            }
        }
    }
    // Convert to 8-bit unsigned integer
    image.convertTo(image, CV_8U);
    cv::imwrite(filename, image);
}

// Function to crop an image to a random square
cv::Mat randomSquareCrop(const cv::Mat& img, int size) {
    int x = std::rand() % (img.cols - size + 1);
    int y = std::rand() % (img.rows - size + 1);
    return img(cv::Rect(x, y, size, size)).clone();
}

cv::Mat loadFromHDF5(const std::string& filename, const std::string& datasetName) {
    // Open the HDF5 file
    HighFive::File file(filename, HighFive::File::ReadOnly);
    // Access the dataset
    HighFive::DataSet dataset = file.getDataSet(datasetName);
    // Get the dimensions
    std::vector<size_t> dims = dataset.getDimensions();
    // Create a temporary buffer to hold the data
    std::vector<std::vector<float>> buffer(dims[0], std::vector<float>(dims[1]));
    // Read the data into the buffer
    dataset.read(buffer);
    // Create a cv::Mat and copy the data from the buffer
    cv::Mat mat(dims[0], dims[1], CV_32F);
    for (size_t i = 0; i < dims[0]; ++i) {
        float* Mi = mat.ptr<float>(i);
        std::copy(buffer[i].begin(), buffer[i].end(), Mi);
    }
    return mat;
}

// Function to encode patches using vector quantization
void processImagesInBatches(const std::vector<std::string>& imagePaths, const int destImageHeight, const int patchHeight, const cv::Mat& oneHotLabels, const cv::Mat& centers, int batchSize, const std::string& destDir) {
    int batchIndex = 0;
    for (size_t i = 0; i < oneHotLabels.rows; i += batchSize) {
        std::cout << "Processing batch " << batchIndex << "..." << std::endl;
        // Store the one-hot labels for this batch
        std::vector<cv::Mat> batchPatches;
        cv::Mat batchOneHotLabels(std::min(batchSize, static_cast<int>( oneHotLabels.rows - i )), oneHotLabels.cols, CV_32F, cv::Scalar(0));
        oneHotLabels.rowRange(i, std::min(static_cast<int>(i + batchSize), oneHotLabels.rows)).copyTo(batchOneHotLabels);
        for (size_t j = i; j < std::min(static_cast<int>(i + batchSize), oneHotLabels.rows); ++j) {

            // Load image, crop to random square, divide into patches and flatten
            cv::Mat img = cv::imread(imagePaths[j]);
            if (img.empty()) continue;
            cv::Mat croppedImg = randomSquareCrop(img, destImageHeight);
            std::vector<cv::Mat> patches = extractPatches(croppedImg, patchHeight);
            cv::Mat flatPatches = flattenPatches(patches);
            
            // Create a KNN model that uses a KD-tree for finding the closest center
            cv::Mat closestCenters;
            cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
            knn->setAlgorithmType(cv::ml::KNearest::KDTREE);

            // Create labels for the centers
            cv::Mat centerLabels = cv::Mat::zeros(centers.rows, 1, CV_32S);
            for (int k = 0; k < centers.rows; ++k) {
                centerLabels.at<int>(k, 0) = k;
            }

            // Train the KNN model
            knn->train(centers, cv::ml::ROW_SAMPLE, centerLabels);
            cv::Mat flatPatches32F;
            flatPatches.convertTo(flatPatches32F, CV_32F);
            cv::Mat flatPatches32FC1 = flatPatches32F.reshape(1, flatPatches32F.rows);

            // Find and store the closest center for each patch
            knn->findNearest(flatPatches32FC1, 1, closestCenters);
            batchPatches.push_back(closestCenters);
        }

        // Flatten batch patches into a single matrix
        cv::Mat batchPatchesMat;
        cv::hconcat(batchPatches, batchPatchesMat);
        batchPatchesMat.convertTo(batchPatchesMat, CV_32F);
        cv::transpose(batchPatchesMat, batchPatchesMat);

        // Save to file
        saveHDF5(destDir + std::to_string(batchIndex++) + ".h5", batchPatchesMat, batchOneHotLabels);
        std::cout << "Saved batch " << batchIndex-1 << "" << std::endl;

        // Load the saved file and compare against the original data
        cv::Mat loadedOneHotLabel, loadedPatch;
        std::cout << "Checking the saved file..." << std::endl;
        loadedOneHotLabel = loadFromHDF5(destDir + std::to_string(batchIndex-1) + ".h5", "labels");
        loadedPatch = loadFromHDF5(destDir + std::to_string(batchIndex-1) + ".h5", "data");
        double diff = cv::norm(batchOneHotLabels, loadedOneHotLabel, cv::NORM_INF);
        double diff2 = cv::norm(batchPatchesMat, loadedPatch, cv::NORM_INF);
        if (diff == 0 && diff2 == 0) {
            std::cout << "Saved file matches the original data." << std::endl;
        } else {
            std::cout << "Saved file does not match the original data." << std::endl;
            std::cout << "Difference in one-hot labels: " << diff << std::endl;
            std::cout << "Difference in patches: " << diff2 << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <source_dataset_dir> <dest_dir> <dest_image_height> "
                  << "<patch_height> <num_clusters> <percentage_patches> <images_per_batch>" << std::endl;
        return 1;
    }

    std::string sourceDatasetDir = argv[1];
    std::string destDir = argv[2];
    int destImageHeight = std::stoi(argv[3]);
    int patchHeight = std::stoi(argv[4]);
    int numClusters = std::stoi(argv[5]);
    float percentagePatches = std::stof(argv[6]);
    int imagesPerBatch = std::stoi(argv[7]);

    // Track the time taken
    auto start = std::chrono::high_resolution_clock::now();

    // Process train directory - list of image patches, their category labels
    // Also capture a selection of random image patches to create centroids for vector quantization
    std::vector<cv::Mat> allPatches;
    std::vector<std::string> labels;
    std::map<std::string, int> labelMap;
    std::vector<std::string> allImagePaths;
    std::cout << "Processing train directory..." << std::endl;
    for (const auto& entry : fs::directory_iterator(sourceDatasetDir + "/train")) {
        std::string label = entry.path().filename().string();
        if (label == ".DS_Store") continue; // Skip macOS metadata file
        std::cout << "Directory: " << label << std::endl;
        // Assign a unique integer to each label
        labelMap[label] = static_cast<int>(labelMap.size());
        for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
            std::string imgName = imgEntry.path().filename().string();
            if (imgName.find("JPEG") == std::string::npos) continue;
            cv::Mat img = cv::imread(imgEntry.path().string());
            if (img.empty()) continue;
            allImagePaths.push_back(imgEntry.path().string());
            cv::Mat croppedImg = randomSquareCrop(img, destImageHeight);
            std::vector<cv::Mat> patches = extractPatches(croppedImg, patchHeight);
            std::vector<cv::Mat> selectedPatches = selectRandomPatches(patches, percentagePatches);
            cv::Mat flatPatches = flattenPatches(selectedPatches);
            allPatches.push_back(flatPatches);
            labels.push_back(label);
        }
    }
    std::cout << "Train directory processed." << std::endl;

    // Process val directory - list of valid image paths and their category labels
    std::vector<std::string> labelsVal;
    std::vector<std::string> allValImagePaths;
    std::cout << "Processing validation directory..." << std::endl;
    for (const auto& entry : fs::directory_iterator(sourceDatasetDir + "/val")) {
        std::string label = entry.path().filename().string();
        if (label == ".DS_Store") continue; // Skip macOS metadata file
        std::cout << "Directory: " << label << std::endl;
        for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
            std::string imgName = imgEntry.path().filename().string();
            if (imgName.find("JPEG") == std::string::npos) continue;
            cv::Mat img = cv::imread(imgEntry.path().string());
            if (img.empty()) continue;
            allValImagePaths.push_back(imgEntry.path().string());
            labelsVal.push_back(label);
        }
    }
    std::cout << "Validation directory processed." << std::endl;

    // Flatten all patches into one matrix
    cv::Mat allPatchesMat;
    cv::vconcat(allPatches, allPatchesMat);
    allPatchesMat.convertTo(allPatchesMat, CV_32F);

    // Perform k-means clustering
    cv::Mat labelsMat, centers;
    std::cout << "Performing k-means clustering..." << std::endl;
    cv::kmeans(allPatchesMat, numClusters, labelsMat, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01), 3, cv::KMEANS_PP_CENTERS, centers);
    std::cout << "K-means clustering complete." << std::endl;

    // Save k-means centroids
    saveHDF5(destDir + "/centroids.h5", centers, cv::Mat());
    std::cout << "Saved the k-means centroids as HDF5." << std::endl;

    // Save k-means centroids as a tiled PNG image
    saveCentroidsAsImage(centers, patchHeight, destDir + "/centroids.png");
    std::cout << "Saved the k-means centroids as an image." << std::endl;

    // Create one-hot encoding for training labels
    int numClasses = static_cast<int>(labelMap.size());
    std::vector<std::string> labelNames(numClasses);
    cv::Mat oneHotLabels(labels.size(), numClasses, CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < labels.size(); ++i) {
        int classIndex = labelMap[labels[i]];
        oneHotLabels.at<float>(i, classIndex) = 1.0;
        labelNames[classIndex] = labels[i];
    }
    // One-hot encoding for validation labels
    cv::Mat oneHotLabelsVal(labelsVal.size(), numClasses, CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < labelsVal.size(); ++i) {
        int classIndex = labelMap[labelsVal[i]];
        oneHotLabelsVal.at<float>(i, classIndex) = 1.0;
    }
    std::cout << "One-hot encoding of labels created." << std::endl;

    // Save label encodings - index corresponds to one-hot label, value corresponds to label name
    H5Easy::File labelFile(destDir + "/dictionary.h5", H5Easy::File::Overwrite);
    H5Easy::dump(labelFile, "label_names", labelNames);
    std::cout << "Saved the one-hot label dictionary as HDF5." << std::endl;

    // Shuffle training set vectors (so oneHotLabels and allImagePaths still align)
    std::vector<int> indices(allImagePaths.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    cv::Mat shuffledOneHotLabels(labels.size(), numClasses, CV_32F, cv::Scalar(0));
    std::vector<std::string> shuffledImagePaths(allImagePaths.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        oneHotLabels.row(indices[i]).copyTo(shuffledOneHotLabels.row(i));
        shuffledImagePaths[i] = allImagePaths[indices[i]];
    }

    //Now shuffle the validation set
    std::vector<int> indicesVal(allValImagePaths.size());
    std::iota(indicesVal.begin(), indicesVal.end(), 0);
    std::shuffle(indicesVal.begin(), indicesVal.end(), std::mt19937{std::random_device{}()});
    cv::Mat shuffledOneHotLabelsVal(labelsVal.size(), numClasses, CV_32F, cv::Scalar(0));
    std::vector<std::string> shuffledValImagePaths(allValImagePaths.size());
    for (size_t i = 0; i < indicesVal.size(); ++i) {
        oneHotLabelsVal.row(indicesVal[i]).copyTo(shuffledOneHotLabelsVal.row(i));
        shuffledValImagePaths[i] = allValImagePaths[indicesVal[i]];
    }

    std::cout << "Encoding training directory..." << std::endl;
    processImagesInBatches(shuffledImagePaths, destImageHeight, patchHeight, shuffledOneHotLabels, centers, imagesPerBatch, destDir + "/train/");
    std::cout << "Training directory encoded." << std::endl;
    std::cout << "Encoding validation directory..." << std::endl;
    processImagesInBatches(shuffledValImagePaths, destImageHeight, patchHeight, shuffledOneHotLabelsVal, centers, imagesPerBatch, destDir + "/val/");
    std::cout << "Validation directory encoded." << std::endl;

    // Track the time taken in seconds
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
