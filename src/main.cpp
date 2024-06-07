#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <highfive/H5Easy.hpp>
#include <armadillo>
#include <random>
#include <vector>
#include <map>
#include <algorithm>
#include <unordered_set>

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

// Function to convert CV_8UC3 cv::Mat to CV_64FC1
cv::Mat convertToCV64FC1(const cv::Mat& input) {
    assert(input.type() == CV_8UC3);

    cv::Mat outputC3;
    input.convertTo(outputC3, CV_64FC3); // Convert types
    cv::Mat output = outputC3.reshape(1, outputC3.rows); // Flatten to 1 channel
    cv::normalize(output, output, 0, 1, cv::NORM_MINMAX); // Normalize to range 0-1

    return output;
}

// Function to convert CV_64FC1 cv::Mat to CV_8UC3
cv::Mat convertToCV8UC3(const cv::Mat& input) {
    assert(input.type() == CV_64FC1);
    cv::Mat output;
    cv::normalize(input, output, 0, 255, cv::NORM_MINMAX); // Normalize to range 0-255
    output = output.reshape(3, input.rows); // Reshape to 3 channels
    output.convertTo(output, CV_8UC3); // Convert types

    return output;
}

// Function to create a tiled PNG from k-means centroids
void saveCentroidsAsImage(const cv::Mat& centers, int patchHeight, const std::string& filename) {
    int numClusters = centers.rows;
    int tileSize = static_cast<int>(std::sqrt(numClusters));    // Number of tiles per row
    int imageSize = tileSize * patchHeight;                    // Size of the output image
    cv::Mat image(imageSize, imageSize, CV_32FC3);
    int idx = 0;
    // Place each centroid in a tile
    for (int y = 0; y < imageSize; y += patchHeight) {
        for (int x = 0; x < imageSize; x += patchHeight) {
            if (idx < numClusters) {
                cv::Mat patch = centers.row(idx).reshape(3, patchHeight);   // Reshape to 3 channels
                patch.copyTo(image(cv::Rect(x, y, patchHeight, patchHeight))); // Copy to the output image
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
    int x = std::rand() % (img.cols - size + 1);    // Random x-coordinate
    int y = std::rand() % (img.rows - size + 1);    // Random y-coordinate
    return img(cv::Rect(x, y, size, size)).clone(); // Return the cropped image
}

// Function to load a cv::Mat from an HDF5 file
template <typename T>
cv::Mat loadFromHDF5(const std::string& filename, const std::string& datasetName, int dataType) {
    // Open the HDF5 file
    HighFive::File file(filename, HighFive::File::ReadOnly);
    // Access the dataset
    HighFive::DataSet dataset = file.getDataSet(datasetName);
    // Get the dimensions
    std::vector<size_t> dims = dataset.getDimensions();
    // Create a temporary buffer to hold the data
    std::vector<std::vector<T>> buffer(dims[0], std::vector<T>(dims[1]));
    // Read the data into the buffer
    dataset.read(buffer);
    // Create a cv::Mat and copy the data from the buffer
    cv::Mat mat(dims[0], dims[1], dataType);
    for (size_t i = 0; i < dims[0]; ++i) {
        T* Mi = mat.ptr<T>(i);
        std::copy(buffer[i].begin(), buffer[i].end(), Mi);
    }
    return mat;
}

// Overloaded functions to load cv::Mat from HDF5 file with float data type
cv::Mat loadFromHDF5Float(const std::string& filename, const std::string& datasetName) {
    return loadFromHDF5<float>(filename, datasetName, CV_32F);
}

// Overloaded functions to load cv::Mat from HDF5 file with double data type
cv::Mat loadFromHDF5Double(const std::string& filename, const std::string& datasetName) {
    return loadFromHDF5<double>(filename, datasetName, CV_64F);
}

// Normalize the brightness and contrast of each row in an image
cv::Mat adjustBrightnessAndContrast(cv::Mat& img, double delta) {
    cv::Mat adjustedImg = img.clone();
    for (int i = 0; i < adjustedImg.rows; ++i) {    // Iterate over each row
        cv::Mat row = adjustedImg.row(i);
        cv::Scalar mean, stddev;
        cv::meanStdDev(row, mean, stddev);
        row = (row - mean.val[0]) / sqrt(stddev.val[0]*stddev.val[0] + delta); // Normalize
    }
    return adjustedImg;
}

// Convert CV_64FC1 cv::Mat to arma::mat
arma::mat cvMatToArmaMat(const cv::Mat& input) {
    assert(input.type() == CV_64FC1);
    arma::mat output(input.rows, input.cols);

    // Copy each element from cv::Mat to Armadillo matrix
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output(i, j) = input.at<double>(i, j);
        }
    }

    return output;
}

// Convert arma::mat to CV_64FC1 cv::Mat where each row is one patch
cv::Mat armaMatToCvMat(const arma::mat& input) {
    cv::Mat output(input.n_rows, input.n_cols, CV_64FC1);

    // Copy each element from Armadillo matrix to cv::Mat
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.at<double>(i, j) = input(i, j);
        }
    }

    return output;
}

// Apply ZCA whitening to rows of a matrix
cv::Mat applyZCAWhitening(const cv::Mat& input, double epsilon_zca = 0.1) {
    assert(input.type() == CV_64FC1);

    cv::Mat output = input.clone();

    for (int r = 0; r < input.rows; r++) {
        // Convert row to Armadillo matrix
        arma::mat armaRow = cvMatToArmaMat(input.row(r));

        // Compute covariance matrix
        arma::mat covMat = arma::cov(armaRow);

        // Compute eigenvalues and eigenvectors
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, covMat);

        // Compute whitening matrix
        arma::mat whiteningMat = eigvec * arma::diagmat(1.0 / arma::sqrt(eigval + epsilon_zca)) * eigvec.t();

        // Apply whitening
        arma::mat whitenedRow = whiteningMat * armaRow;

        // Convert back to cv::Mat and store in output
        output.row(r) = armaMatToCvMat(whitenedRow);
    }

    return output;
}

// Calculate the cosine similarity between each image and each centroid
cv::Mat calculateCosineSimilarity(const cv::Mat& centroids, const cv::Mat& images) {
    assert(centroids.cols == images.cols);
    assert(centroids.type() == CV_64FC1 && images.type() == CV_64FC1);

    cv::Mat output(images.rows, centroids.rows, CV_64FC1);

    for (int i = 0; i < images.rows; ++i) {     // Iterate over each patch
        for (int j = 0; j < centroids.rows; ++j) {  // Iterate over each centroid
            cv::Mat imageRow = images.row(i);
            cv::Mat centroidRow = centroids.row(j);

            // Cosine similarity is just the dot product for normalized vectors
            double cosineSimilarity = imageRow.dot(centroidRow);

            output.at<double>(i, j) = cosineSimilarity;
        }
    }

    return output;
}

// Apply softmax to a matrix by rows
cv::Mat applySoftmax(const cv::Mat& input) {
    assert(input.type() == CV_64FC1);

    cv::Mat output = cv::Mat::zeros(input.size(), CV_64FC1);

    for (int i = 0; i < input.rows; ++i) {
        cv::Mat row = input.row(i);

        // Subtract the max for numerical stability
        double maxVal;
        cv::minMaxLoc(row, nullptr, &maxVal);
        row -= maxVal;

        // Calculate the exponent
        cv::exp(row, row);

        // Normalize by the sum of the row
        double sum = cv::sum(row)[0];
        row /= sum;

        row.copyTo(output.row(i));
    }

    return output;
}

// Function to generate 2d sinusoidal positional encoding (with edge marker) for each patch
cv::Mat generatePositionalEncoding(int patchCount, int patchPerRow) {
    int dimension = 2; // x and y coordinates
    cv::Mat positionalEncoding = cv::Mat::zeros(patchCount, dimension + 1, CV_64F); // +1 for border marker
    for (int i = 0; i < patchCount; ++i) {      // Iterate over each patch
        for (int j = 0; j < dimension; ++j) {   // Iterate over each dimension
            // Calculate the sinusoidal encoding
            positionalEncoding.at<double>(i, j) = sin((double)i / (pow(10000, (double)(j+1) / dimension)));
        }
        // Add border marker
        int row = i / patchPerRow;
        int col = i % patchPerRow;
        if (row == 0 || row == patchPerRow - 1 || col == 0 || col == patchPerRow - 1) {
            positionalEncoding.at<double>(i, dimension) = 1.0; // border patch
        } else {
            positionalEncoding.at<double>(i, dimension) = 0.0; // non-border patch
        }
    }
    return positionalEncoding;
}

// Save each column of the positional encoding as an image
void savePositionalEncodingAsImages(const cv::Mat& positionalEncoding, int patchPerRow, const std::string& basePath) {
    // Iterate over each positional encoding dimension
    for (int i = 0; i < positionalEncoding.cols; ++i) {
        cv::Mat column = positionalEncoding.col(i).clone();
        cv::Mat image = column.reshape(1, patchPerRow);     // Reshape to 2D image
        
        // Normalize the values to [0, 255] and convert to CV_8U
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8U);

        std::string filename = basePath + "/positionalEncoding" + std::to_string(i) + ".png";
        bool success = cv::imwrite(filename, image);
        if (!success) {
            std::cerr << "Failed to save image: " << filename << std::endl;
        }
    }
}

// Function to encode patches using probability of cluster membership (soft-max cosine distance to k-means centroids)
void sphereImagesInBatches(const std::vector<std::string>& imagePaths, const int destImageHeight, const int patchHeight, const cv::Mat& oneHotLabels, const cv::Mat& centers, int batchSize, const std::string& destDir) {
    int batchIndex = 0;
    cv::Mat positionalEncoding;
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
            flatPatches = convertToCV64FC1(flatPatches);

            // Improve each patch by normalizing brightness/contrast then whitening
            flatPatches = adjustBrightnessAndContrast(flatPatches, 0.04);
            flatPatches = applyZCAWhitening(flatPatches, 0.05);

            // Normalize each row to a vector of unit length
            for (int i = 0; i < flatPatches.rows; ++i) {
                cv::normalize(flatPatches.row(i), flatPatches.row(i), 1, 0, cv::NORM_L2);
            }

            // Calculate distances to centroids and convert to probabilities
            cv::Mat embedding = calculateCosineSimilarity(centers, flatPatches);
            embedding = applySoftmax(embedding);

            // Create positional encoding for the embedding on first image
            int patchPerRow = sqrt(embedding.rows);
            if (positionalEncoding.empty()) {
                positionalEncoding = generatePositionalEncoding(embedding.rows, patchPerRow);
                std::cout << "Saving positional encoding as images..." << std::endl;
                savePositionalEncodingAsImages(positionalEncoding, patchPerRow, destDir + "/..");
            }

            // Concatenate positional encoding to the end of the embedding for every image
            cv::hconcat(embedding, positionalEncoding, embedding);

            batchPatches.push_back(embedding);

            // Create labels for the centers
            cv::Mat centerLabels = cv::Mat::zeros(centers.rows, 1, CV_32S);
            for (int k = 0; k < centers.rows; ++k) {
                centerLabels.at<int>(k, 0) = k;
            }
        }
        cv::Mat batchPatchesMat;
        cv::vconcat(batchPatches, batchPatchesMat);
        cv::transpose(batchOneHotLabels, batchOneHotLabels);

        // Save to file
        saveHDF5(destDir + std::to_string(batchIndex++) + ".h5", batchPatchesMat, batchOneHotLabels);
        std::cout << "Saved batch " << batchIndex-1 << "" << std::endl;

        // Load the saved file and compare against the original data
        cv::Mat loadedOneHotLabel, loadedPatch;
        std::cout << "Checking the saved file..." << std::endl;
        loadedOneHotLabel = loadFromHDF5Float(destDir + std::to_string(batchIndex-1) + ".h5", "labels");
        loadedPatch = loadFromHDF5Double(destDir + std::to_string(batchIndex-1) + ".h5", "data");
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

// Function to find the optimal number of clusters using k-means
int findOptimalClusters(cv::Mat& data, int lower, int upper, int maxIterations) {
    cv::Mat labels, centers;
    double minCompactness = std::numeric_limits<double>::max();
    int optimalClusters = lower;

    auto startTotalTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < maxIterations; ++i) {
        int mid = lower + (upper - lower) / 2;      // Midpoint of the range
        auto startIterationTime = std::chrono::high_resolution_clock::now();
        // Perform k-means clustering
        double compactness = cv::kmeans(data, mid, labels, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01), 3, cv::KMEANS_PP_CENTERS, centers);
        auto endIterationTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedIterationTime = endIterationTime - startIterationTime;
        std::cout << "Iteration " << i << ": " << mid << " clusters, compactness: " << compactness << ", time: " << elapsedIterationTime.count() << " seconds." << std::endl;
        
        // Update the optimal number of clusters
        if (compactness < minCompactness) {
            minCompactness = compactness;
            optimalClusters = mid;
        }

        // Break if the range is too small
        if (upper - lower <= 1) {
            break;
        }

        // Update the range based on the compactness
        if (compactness > minCompactness) {
            upper = mid;
            std::cout << "Searching in lower half..." << std::endl;
        } else {
            lower = mid;
            std::cout << "Searching in upper half..." << std::endl;
        }
    }
    auto endTotalTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTotalTime = endTotalTime - startTotalTime;
    std::cout << "Total time taken: " << elapsedTotalTime.count() << " seconds." << std::endl;

    return optimalClusters;
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

    std::unordered_set<std::string> acceptableFileTypes = {"JPEG", "jpg", "png"};


    // Track the time taken
    auto start = std::chrono::high_resolution_clock::now();

    // Process train directory - list of image patches, their category labels
    // Also capture a selection of random image patches to create centroids
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
            flatPatches = convertToCV64FC1(flatPatches);
            allPatches.push_back(flatPatches);
            labels.push_back(label);
        }
    }
    std::cout << "Train directory processed." << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;

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
            // if (imgName.find("JPEG") == std::string::npos) continue;
            std::string fileExtension = imgName.substr(imgName.find_last_of(".") + 1);
            if (acceptableFileTypes.find(fileExtension) == acceptableFileTypes.end()) continue;
            cv::Mat img = cv::imread(imgEntry.path().string());
            if (img.empty()) continue;
            allValImagePaths.push_back(imgEntry.path().string());
            labelsVal.push_back(label);
        }
    }
    std::cout << "Validation directory processed." << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;


    // Flatten all patches into one matrix
    cv::Mat allPatchesMat;
    cv::vconcat(allPatches, allPatchesMat);

    // Improve each patch by normalizing brightness/contrast then whitening
    std::cout << "Applying brightness and contrast adjustment..." << std::endl;
    allPatchesMat = adjustBrightnessAndContrast(allPatchesMat, 0.04);
    std::cout << "Applying ZCA whitening..." << std::endl;
    allPatchesMat = applyZCAWhitening(allPatchesMat, 0.05);

    // Normalize each row to a vector of unit length
    for (int i = 0; i < allPatchesMat.rows; ++i) {
        cv::normalize(allPatchesMat.row(i), allPatchesMat.row(i), 1, 0, cv::NORM_L2);
    }

    // Perform k-means clustering
    cv::Mat labelsMat, centers;
    cv::Mat allPatchesMat32F;
    allPatchesMat.convertTo(allPatchesMat32F, CV_32F);

    // ********************************************************************************
    // UNCOMMENT TO FIND THE OPTIMAL NUMBER OF CLUSTERS
    // std::cout << "Finding the optimal number of clusters..." << std::endl;
    // int optimalClusters = findOptimalClusters(allPatchesMat32F, 100, 2000, 20);
    // std::cout << "Optimal number of clusters: " << optimalClusters << std::endl;
    // ********************************************************************************

    std::cout << "Performing k-means clustering..." << std::endl;
    auto startKMeans = std::chrono::high_resolution_clock::now();
    cv::kmeans(allPatchesMat32F, numClusters, labelsMat, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), 3, cv::KMEANS_PP_CENTERS, centers);
    auto endKMeans = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedKMeans = endKMeans - startKMeans;
    std::cout << "K-means clustering complete: " << elapsedKMeans.count() << " seconds." << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;
    centers.convertTo(centers, CV_64F);

    // Save k-means centroids
    saveHDF5(destDir + "/centroids.h5", centers, cv::Mat());
    std::cout << "Saved the k-means centroids as HDF5." << std::endl;

    // Convert the centroids back to a colour image for saving
    cv::Mat centersNorm8U = convertToCV8UC3(centers);

    // Save k-means centroids as a tiled PNG image
    saveCentroidsAsImage(centersNorm8U, patchHeight, destDir + "/centroids.png");
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

    std::cout << "==================================================" << std::endl << std::endl;

    std::cout << "Encoding training directory..." << std::endl;
    sphereImagesInBatches(shuffledImagePaths, destImageHeight, patchHeight, shuffledOneHotLabels, centers, imagesPerBatch, destDir + "/train/");
    std::cout << "Training directory encoded." << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;

    std::cout << "Encoding validation directory..." << std::endl;
    sphereImagesInBatches(shuffledValImagePaths, destImageHeight, patchHeight, shuffledOneHotLabelsVal, centers, imagesPerBatch, destDir + "/val/");
    std::cout << "Validation directory encoded." << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;

    // Track the time taken in seconds
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
