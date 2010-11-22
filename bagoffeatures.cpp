#include <stdlib.h>
#include "bagoffeatures.h"

extern "C"
{
    #include "Sift/include/sift.h"
    #include "Sift/include/imgfeatures.h"
    #include "Sift/include/kdtree.h"
    #include "Sift/include/minpq.h"
    #include "Sift/include/utils.h"
    #include "Sift/include/xform.h"
}

#include "Surf/surflib.h"


void copySIFTPts(ImageFeatures &dst, feature* src, const int size, const int length);
void copySURFPts(ImageFeatures &dst, const IpVec src, const int length);
int findDictionaryMatch(float* descriptor, CvMat* dict, int length);
IplImage* preProcessImages(const IplImage* input, int minSize, int maxSize);

BagOfFeatures::BagOfFeatures()
{
    numClasses = 0;
    numFeatures = 0;
    descrSize = 0;
    testObject = NULL;
    validObject = NULL;
    trainObject = NULL;
    data = NULL;
    dictionary = NULL;
    SVMModel = NULL;
    classifierType = -1;
    SVMModel_CV = NULL;
    NBModel_CV = NULL;
}

BagOfFeatures::BagOfFeatures(const int n, DataSet* val)
{
    int i;

    numClasses = n;
    numFeatures = 0;
    descrSize = 0;
    dictionary = NULL;
    SVMModel = NULL;
    SVMModel_CV = NULL;
    NBModel_CV = NULL;
    testObject = new ObjectSet [n];
    validObject = new ObjectSet [n];
    trainObject = new ObjectSet [n];
    data = new DataSet [n];
    classifierType = -1;
    int train, valid, test, label;

    for(i = 0; i < numClasses; i++)
    {
        data[i] = val[i];
        data[i].getDataInfo(train, valid, test, label);
        if(test > 0)
            testObject[i].alloc(test);
        if(valid > 0)
            validObject[i].alloc(valid);
        if(train > 0)
            trainObject[i].alloc(train);
    }
}

BagOfFeatures::~BagOfFeatures()
{
    int i;
    numClasses = 0;
    numFeatures = 0;
    descrSize = 0;
    delete [] testObject;
    delete [] validObject;
    delete [] trainObject;
    delete [] data;
    svm_destroy_model(SVMModel);
    //delete SVMModel_CV;
}

void BagOfFeatures::allocBoF(const int n, DataSet* val)
{
    int i;
    if(data != NULL)
    {
        numClasses = 0;
        delete [] testObject;
        delete [] validObject;
        delete [] trainObject;
        delete [] data;
    }

    numClasses = n;
    numFeatures = 0;
    descrSize = 0;
    classifierType = -1;
    testObject = new ObjectSet [n];
    validObject = new ObjectSet [n];
    trainObject = new ObjectSet [n];
    data = new DataSet [n];

    int train, valid, test, label;

    for(i = 0; i < numClasses; i++)
    {
        data[i] = val[i];
        data[i].getDataInfo(train, valid, test, label);
        if(test > 0)
            testObject[i].alloc(test);
        if(valid > 0)
            validObject[i].alloc(valid);
        if(train > 0)
            trainObject[i].alloc(train);
    }
}


bool BagOfFeatures::extractSIFTFeatures(int lvls = SIFT_INTVLS,
                                        double sigma = SIFT_SIGMA,
                                        double thresh1 = SIFT_CONTR_THR,
                                        int thresh2 = SIFT_CURV_THR,
                                        int dbl = SIFT_IMG_DBL,
                                        int width = SIFT_DESCR_WIDTH,
                                        int bins = SIFT_DESCR_HIST_BINS)
{
    if(numFeatures)
        return false;

	int i, j;
    int train, valid, test, label;
    int count;
    char fileName[256];
    IplImage *dataImage = NULL;
    struct feature *siftFeatures = NULL;

    descrSize = width*width*bins;

    // For each object class
	for(i = 0; i < numClasses; i++)
	{
	    // Get the distribution of data
        data[i].getDataInfo(train, valid, test, label);

        // Extrain the features of the training set
        // For each training image
        for(j = 0; j < train; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j));
            cout << "Loading training image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            IplImage *resized = preProcessImages(dataGray, 80, 200);

            // Extract the sift features from the images
            //Default:  SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
            //          SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
            count = _sift_features(resized, &siftFeatures, lvls, sigma, thresh1, thresh2, dbl, width, bins);
            cout << "Found " << count << " SIFT interest points" << endl;

            // Keep a running total of features for training purposes
            numFeatures += count;

            // Copy the descriptors into the feature set
            copySIFTPts(trainObject[i].featureSet[j], siftFeatures, count, descrSize);

            // Release Memory
            free(siftFeatures);
            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);
        }

        // Extrain the features of the validation set
        // For each validation image
        for(j = 0; j < valid; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j+train));
            cout << "Loading validation image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            IplImage *resized = preProcessImages(dataGray, 80, 200);

            // Extract the sift features from the images
            //Default:  SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
            //          SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
            count = _sift_features(resized, &siftFeatures, lvls, sigma, thresh1, thresh2, dbl, width, bins);
            cout << "Found " << count << " SIFT interest points" << endl;

            // Copy the descriptors into the feature set
            copySIFTPts(validObject[i].featureSet[j], siftFeatures, count, descrSize);

            // Release Memory
            free(siftFeatures);
            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);
        }

        // Extrain the features of the test set
        // For each test image
        for(j = 0; j < test; j++)
        {
            // Get the image from the data list
            strcpy(fileName, data[i].getDataList(j+train+valid));
            cout << "Loading test image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            IplImage *resized = preProcessImages(dataGray, 80, 200);

            // Extract the sift features from the images
            //Default:  SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
            //          SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
            count = _sift_features(resized, &siftFeatures, lvls, sigma, thresh1, thresh2, dbl, width, bins);
            cout << "Found " << count << " SIFT interest points" << endl;

            // Copy the descriptors into the feature set
            copySIFTPts(testObject[i].featureSet[j], siftFeatures, count, descrSize);

            // Release Memory
            free(siftFeatures);
            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);
        }
	}

	return true;
}

bool BagOfFeatures::extractSURFFeatures(bool invariant,
                                        int octaves,
                                        int intervals,
                                        int step,
                                        float thresh)
{
    if(numFeatures > 0)
        return false;

	int i, j;
    int train, valid, test, label;

    char fileName[256];

	IpVec temp;
    IplImage* dataImage = NULL;

    descrSize = 64;

	for(i = 0; i < numClasses; i++)
	{
	    // Get the distribution of data
        data[i].getDataInfo(train, valid, test, label);
	    // Extrain the features of the training set
        // For each training image
        for(j = 0; j < train; j++)
        {
            strcpy(fileName, data[i].getDataList(j));
            cout << "Loading training image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            //Resize the images
            IplImage *resized = preProcessImages(dataGray, 75, 150);

            // Detect the SURF features
            surfDetDes(resized, temp, invariant, octaves, intervals, step, thresh);

            cout << "OpenSURF found: " << temp.size() << " interest points" << endl;
            // Keep track of the feature count
            numFeatures += temp.size();

            /*
            drawIpoints(resized, temp, 3);

            IplImage* display = cvCreateImage(cvSize(resized->width*4, resized->height*4), resized->depth, resized->nChannels);
            cvResize(resized, display, CV_INTER_CUBIC);
            cvShowImage("Extracted SURF", display);
            cvWaitKey(150);
            cvReleaseImage(&display);
            */
            // Copy the SURF feature into the feature object
            copySURFPts(trainObject[i].featureSet[j], temp, descrSize);

            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);
        }

        // Extrain the features of the validation set
        // For each validation image
        for(j = 0; j < valid; j++)
        {
            strcpy(fileName, data[i].getDataList(j+train));
            cout << "Loading validation image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            //Resize the images
            IplImage *resized = preProcessImages(dataGray, 75, 150);

            // Detect the SURF features
            surfDetDes(resized, temp, invariant, octaves, intervals, step, thresh);

            cout << "OpenSURF found: " << temp.size() << " interest points" << endl;


            /*
            drawIpoints(resized, temp, 3);

            IplImage* display = cvCreateImage(cvSize(resized->width*4, resized->height*4), resized->depth, resized->nChannels);
            cvResize(resized, display, CV_INTER_CUBIC);
            cvShowImage("Extracted SURF", display);
            cvWaitKey(150);
            cvReleaseImage(&display);
            */


            // Copy the SURF feature into the feature object
            copySURFPts(validObject[i].featureSet[j], temp, descrSize);

            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);

        }

        // Extrain the features of the test set
        // For each test image
        for(j = 0; j < test; j++)
        {
            strcpy(fileName, data[i].getDataList(j+train+valid));
            cout << "Loading test image: " << fileName << endl;
            dataImage = cvLoadImage(fileName);
            IplImage *dataGray = cvCreateImage(cvSize(dataImage->width, dataImage->height), 8, 1);
            // Convert to grayscale
            cvCvtColor(dataImage, dataGray, CV_BGR2GRAY);

            //Resize the images
            IplImage *resized = preProcessImages(dataGray, 75, 150);

            // Detect the SURF features
            surfDetDes(resized, temp, invariant, octaves, intervals, step, thresh);

            cout << "OpenSURF found: " << temp.size() << " interest points" << endl;

            /*
            drawIpoints(resized, temp, 3);

            IplImage* display = cvCreateImage(cvSize(resized->width*4, resized->height*4), resized->depth, resized->nChannels);
            cvResize(resized, display, CV_INTER_CUBIC);
            cvShowImage("Extracted SURF", display);
            cvWaitKey(150);
            cvReleaseImage(&display);
            */
            // Copy the SURF feature into the feature object
            copySURFPts(testObject[i].featureSet[j], temp, descrSize);

            cvReleaseImage(&dataImage);
            cvReleaseImage(&dataGray);
            cvReleaseImage(&resized);
        }
	}

    return true;
}

bool BagOfFeatures::buildHierarchicalTree(int transpose=0,
                                          char dist='e',
                                          char method='s',
                                          double** distmatrix=NULL)
{
//double** buildMatrixHierarchical(const ImageFeatures *featurePts, const int feature_count,
//			const int image_count, Node*& tree )
    // Check to makes sure that features were found
    if(trainObject ==  NULL || numFeatures == 0 || descrSize == 0)
        return false;

	int i, j;
	int k = 0, l = 0, m;
	int size;
	int totalImages = 0;

	// Initialize the data
	hClusterData = new double* [numFeatures];
	for(i = 0; i < numFeatures; i++)
		hClusterData[i] = new double [descrSize];

    // Allocate mask and set it all to 1 (assume no missing data)
	int ** mask = new int* [numFeatures];
	for(i = 0; i < numFeatures; i++)
		mask[i] = new int [descrSize];
	for(i = 0; i < numFeatures; i++)
		for(j = 0; j< descrSize; j++)
			mask[i][j] = 1;

	// Set the weights equal, all 1
	double * weight = new double [descrSize];
	for(i = 0; i < descrSize; i++)
		weight[i] = 1.0;


	// For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = data[m].getTrainSize();
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // for each feature in that image...
            for(i = 0; i < size; i++)
            {
                // Copy the descriptor into the data array
                for(j = 0; j < descrSize; j++)
                {
                    hClusterData[k][j] = (double)trainObject[m].featureSet[l].descriptors[i][j];
                    //cout << hClusterData[k][j] << " ";
                }
                //cout << endl;
                k++;
            }
        }
    }

	// Centroid Hierarchical Clustering
	// feature_count X DESCRIPTOR_SIZE
	// The feature vectors
	// mask (all 1s)
	// weights (all 1s)
	hTree = treecluster(numFeatures, descrSize, hClusterData, mask, weight, transpose, dist, method, distmatrix);

	// Release the mask
	for(i = 0; i < numFeatures; i++)
		delete [] mask[i];
	delete [] mask;
	// Release the weight
	delete [] weight;

    // Make sure that the tree was allocated
	if(!hTree)
	{
		cout << "Could not allocate the tree: Insufficient memory..." << endl;
		for(i = 0; i < numFeatures; i++)
            delete [] hClusterData[i];
        delete [] hClusterData;
		return false;
	}

    return true;
}

bool BagOfFeatures::cutHierarchicalTree(int numClusters)
{
    if(hClusterData == NULL || hTree == NULL)
        return false;

    if(dictionary != NULL)
        cvReleaseMat(&dictionary);
    int i, j, index;
    float *ptrCenter;

    int *clusterID = new int [numFeatures];
	int *indexCount = new int [numClusters];
	// initialize the count to zero
	for(i = 0; i < numClusters; i++)
		indexCount[i] = 0;

    dictionary = cvCreateMat(numClusters, descrSize, CV_32FC1);

    cvSetZero(dictionary);

	// Cluster the features based on the cluster_count
	cuttree(numFeatures, hTree, numClusters, clusterID);

    // Find the number of features in each cluster
    for(i = 0; i < numFeatures; i++)
    {
        index = clusterID[i];
        indexCount[index]++;
    }

	// Figure out how many clusters per index
	for(i = 0; i < numFeatures; i++)
	{
        index = clusterID[i];
		ptrCenter = (float *)(dictionary->data.ptr + index * dictionary->step);
		//cout << i << "\t";
		for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] += (float)hClusterData[i][j];
            //cout << ptrCenter[j] << "\t";
        }
	}

	for(i = 0; i < numClusters; i++)
	{
        ptrCenter = (float *)(dictionary->data.ptr + i * dictionary->step);
        //cout << i << " \t\t\t" << indexCount[i] << endl << endl;
        float t = indexCount[i];
        for(j = 0; j < descrSize; j++)
        {
            ptrCenter[j] /= (float)indexCount[i];
        }
    }



    int k;
    float *checkData = new float [descrSize];
    float minDist;
    float dist;
    int temp;
    int minIndex;

    for(i = 0; i < numFeatures; i++)
    {
        minDist = 999999.;
        for(j = 0; j < numClusters; j++)
        {
            ptrCenter = (float*)(dictionary->data.ptr + j*dictionary->step);
            for(k = 0; k < descrSize; k++)
            {
                checkData[k] = ptrCenter[k];
            }
            dist = 0;
            for(k = 0; k < descrSize; k++)
            {
                dist += (checkData[k] - hClusterData[i][k])*(checkData[k] - hClusterData[i][k]);
            }
            dist /= descrSize;//sqrt(dist);
            if(dist < minDist)
            {
                minDist = dist;
                minIndex = j;
            }
        }
        temp = clusterID[i];
        if(minIndex != clusterID[i])
            cout << "PROBLEM DURING CLUSTERING" << endl;
    }
    delete [] checkData;


    delete [] clusterID;
	delete [] indexCount;
    return true;
}

bool BagOfFeatures::buildKMeans(int numClusters,
                                CvTermCriteria criteria = cvTermCriteria(
                                        CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,
                                        5,
                                        1.0),
                                int repeat=5)
{
    if(numFeatures == 0 || trainObject == NULL)
        return false;

    if(dictionary != NULL)
        cvReleaseMat(&dictionary);
	int i, j, k = 0, l = 0, m = 0;
	int size, index;
	int totalImages;
	int emptyClusters = 0;
	float* ptrRaw = NULL, *ptrCenter = NULL;
	int* ptrIndex = NULL;
	int* indexCount = NULL;

	// create a matrix that will  contain all the features
	CvMat* feature_mat = cvCreateMat(numFeatures, descrSize, CV_32FC1);
	CvMat* descriptor_clusters = cvCreateMat(numFeatures, 1, CV_32SC1);

	//keep track of how many descriptors there are in each cluster
	indexCount = new int [numClusters];
	// initialize the count to zero
	for(i = 0; i < numClusters; i++)
		indexCount[i] = 0;

    // For each class
    for(m = 0; m < numClasses; m++)
    {
        totalImages = data[m].getTrainSize();
        // For each image in that class...
        for(l = 0; l < totalImages; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // for each feature in that image...
            for(i = 0; i < size; i++)
            {
                ptrRaw = (float *)(feature_mat->data.ptr + k * feature_mat->step);
                // put them in the raw descriptors matrix
                for(j = 0; j < descrSize; j++)
                {
                    ptrRaw[j] = trainObject[m].featureSet[l].descriptors[i][j];
                }
                k++;
            }
        }
    }

	// Cluster the raw matrix with a number of cluster found previously
	cvKMeans2( feature_mat, numClusters, descriptor_clusters, criteria,	repeat);
	// Repeat the clustering by CLUSTER_REPEAT times to get best results
    cout << "Done clustering... \nRegecting empty clusters..." << endl;
	// Figure out how many clusters per index
	for(i = 0; i < numFeatures; i++)
	{
		ptrIndex = (int *)(descriptor_clusters->data.ptr + i * descriptor_clusters->step);
		index = *ptrIndex;
		// increment the number of vectors found in that cluster
		indexCount[index]++;
	}

	// Find how many empty clusters there are
	for(i = 0; i < numClusters; i++)
	{
		if(indexCount[i] == 0)
		{
			emptyClusters++;
		}
	}

	// Descriptor cluster centers: This will look at all the clusters, even the empty
	CvMat* raw_cluster_centers = cvCreateMat(numClusters, descrSize, CV_32FC1);

	for(i = 0; i < numClusters; i++)
	{
		ptrCenter = (float *)(raw_cluster_centers->data.ptr + i * raw_cluster_centers->step);
		for(j = 0; j < descrSize; j++)
		{
			ptrCenter[j] = 0;
		}
	}

	cout << "Total Empty clusters found: " << emptyClusters
		<< " out of " << numClusters << " total clusters" << endl;
	// Calculate the cluster center for the descriptors
	for(i = 0; i < numFeatures; i++)
	{
		ptrRaw = (float *)(feature_mat->data.ptr + i * feature_mat->step);
		// This will give the cluster index number for each descriptor
		ptrIndex = (int *)(descriptor_clusters->data.ptr + i * descriptor_clusters->step);
		index = *ptrIndex;
		ptrCenter = (float *)(raw_cluster_centers->data.ptr + index * raw_cluster_centers->step);
		// Sum up the vectors for each cluster
		for(j = 0; j < descrSize; j++)
		{
			ptrCenter[j] += ptrRaw[j];
		}
	}

	dictionary = cvCreateMat(numClusters - emptyClusters, descrSize, CV_32FC1);
	k = 0;
	// Copy all the non-empty clusters to the cluster_center matrix
	// And output the clusters to the file
	for(i = 0; i < numClusters; i++)
	{
		ptrRaw = (float *)(raw_cluster_centers->data.ptr + i * raw_cluster_centers->step);
		if(indexCount[i] > 0)
		{
			ptrCenter = (float *)(dictionary->data.ptr + k * dictionary->step);
			//cout << i << " \t\t\t" << indexCount[i] << endl << endl;
			for(j = 0; j < descrSize; j++)
			{
				// Calulate the average by dividing by how many in that cluster
				ptrCenter[j] = (ptrRaw[j] / indexCount[i]);
			}
			k++;
		}
	}

	// Release all the matrices allocated
	cvReleaseMat(&feature_mat);
	cvReleaseMat(&descriptor_clusters);
	cvReleaseMat(&raw_cluster_centers);
	// Release the index count
	delete [] indexCount;

	return true;
}

bool BagOfFeatures::buildBofHistograms(bool normalize=true)
{
    if(dictionary == NULL)
        return false;

	int k, l, m, minIndex = 0;
	int count = dictionary->rows;
	int train, valid, test, label, size;

	// If the identity matrix is used for cvMahalanobis, then
	// the distance is equal to Euclidean distance
	//cvSetIdentity(identMat);

     // For each class
    for(m = 0; m < numClasses; m++)
    {
        // Get the information
        data[m].getDataInfo(train, valid, test, label);

        //Training Histograms
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        if(!trainObject[m].histogramSet.alloc(count, label, train))
        {
            trainObject[m].histogramSet.dealloc();
            trainObject[m].histogramSet.alloc(count, label, train);
        }
        // For each training image in that class...
        for(l = 0; l < train; l++)
        {
            size = trainObject[m].featureSet[l].size;
            // For each features in that image
            for(k = 0; k < size; k++)
            {
                // Find the best match, and add it to the bin of the histogram
                minIndex = findDictionaryMatch(trainObject[m].featureSet[l].descriptors[k], dictionary, descrSize);
                // Increment the histogram where the vector belongs
                trainObject[m].histogramSet.addToBin(l, minIndex);
            }
        }
        // Check if user wants to normalize
        if(normalize)
            trainObject[m].histogramSet.normalizeHist();

        // Validation Histograms:
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        if(!validObject[m].histogramSet.alloc(count, label, valid))
        {
            validObject[m].histogramSet.dealloc();
            validObject[m].histogramSet.alloc(count, label, valid);
        }
        // For each training image in that class...
        for(l = 0; l < valid; l++)
        {
            size = validObject[m].featureSet[l].size;
            // For each features in that image
            for(k = 0; k < size; k++)
            {
                // Find the best match, and add it to the bin of the histogram
                minIndex = findDictionaryMatch(validObject[m].featureSet[l].descriptors[k], dictionary, descrSize);
                // Increment the histogram where the vector belongs
                validObject[m].histogramSet.addToBin(l, minIndex);
            }
        }
        // Check if user wants to normalize
        if(normalize)
            validObject[m].histogramSet.normalizeHist();

        // test Histograms:
        // allocate the histogram of size "count", with label, and number of histograms "train"
        // Make sure it hasn't been allocated before
        if(!testObject[m].histogramSet.alloc(count, label, test))
        {
            testObject[m].histogramSet.dealloc();
            testObject[m].histogramSet.alloc(count, label, test);
        }
        // For each training image in that class...
        for(l = 0; l < test; l++)
        {
            size = testObject[m].featureSet[l].size;
            // For each features in that image
            for(k = 0; k < size; k++)
            {
                // Find the best match, and add it to the bin of the histogram
                minIndex = findDictionaryMatch(testObject[m].featureSet[l].descriptors[k], dictionary, descrSize);
                // Increment the histogram where the vector belongs
                testObject[m].histogramSet.addToBin(l, minIndex);
            }
        }
        // Check if user wants to normalize
        if(normalize)
            testObject[m].histogramSet.normalizeHist();
    }

    return true;
}

void BagOfFeatures::trainSVM(int type = NU_SVC,
                            int kernel = RBF,
                            double degree = 0.05,
                            double gamma = 0.25,
                            double coef0 = 0.5,
                            double C = .05,
                            double cache = 300,
                            double eps = 0.000001,
                            double nu = 0.5,
                            int shrinking = 0,
                            int probability = 0,
                            int weight = 0)
{
    if(SVMModel != NULL)
    {
        svm_destroy_model(SVMModel);
        //svm_destroy_param(&SVMParam);
    }




    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;
    int count;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    // Set up the data
    struct svm_problem SVMProblem;
    SVMProblem.l = totalData;
    SVMProblem.y = new double [totalData];
    SVMProblem.x = new struct svm_node* [totalData];
    // Allocate memory
    //for(i = 0; i < totalData; i++)
    //{
    //    SVMProblem.x[i] = new struct svm_node [length+1];
    //}

    // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            count = 0;
            for(k = 0; k < length; k++)
            {
                if(trainObject[i].histogramSet.histogram[j][k] != 0)
                    count++;
            }
            SVMProblem.x[l] = new struct svm_node [count+1];
            count = 0;
            for(k = 0; k < length; k++)
            {
                if(trainObject[i].histogramSet.histogram[j][k] != 0)
                {
                    SVMProblem.x[l][count].index = k;
                    SVMProblem.x[l][count].value = trainObject[i].histogramSet.histogram[j][k];
                    cout << "(" << SVMProblem.x[l][count].index
                        << ", " << SVMProblem.x[l][count].value << ")" << endl;
                    count++;
                }
            }
            SVMProblem.x[l][count].index = -1;
            cout << endl;
            //SVMProblem.x[l][count].value = -1;
            // Copy the histograms
            //for(k = 0; k < length; k++)
            //{
            //    SVMProblem.x[l][k].index = k;
            //    SVMProblem.x[l][k].value = trainObject[i].histogramSet.histogram[j][k];
            //}
            // End of the data
            //SVMProblem.x[l][length].index = -1;
            //SVMProblem.x[l][length].value = -1;
            //Attach the labels

            SVMProblem.y[l] = data[i].getLabel();
            //cout << "Label: " << SVMProblem.y[l] << endl;
        }
    }

    // Types
    SVMParam.svm_type = type;
    SVMParam.kernel_type = kernel;
    // Parameters
    SVMParam.degree = degree;
    SVMParam.gamma = gamma;
    SVMParam.coef0 = coef0;
    SVMParam.C = C;
    // For training only
    SVMParam.cache_size = cache;
    SVMParam.eps = eps;
    SVMParam.nu = nu;
    SVMParam.shrinking = shrinking;
    SVMParam.probability = probability;
    // Don't change the weights
    SVMParam.nr_weight = weight;


    double* target = new double [totalData];
    svm_check_parameter(&SVMProblem, &SVMParam);
    svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
    SVMModel = svm_train(&SVMProblem, &SVMParam);
    delete [] target;

    classifierType = LIBSVM_CLASSIFIER;

}

void BagOfFeatures::trainSVM_CV(int type, int kernel, double degree, double gamma, double coef0,
                        double C, double nu, double p, int termType, int iterations, double eps,
                        char* fileName)
{
    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;

    float *dPtr;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    //CvMat* trainData = cvCreateMat(totalData, dictionary->rows, CV_32FC1);
    //CvMat* dataLabel = cvCreateMat(totalData, 1, CV_32FC1);

    float** trainData = new float* [totalData];
    float* dataLabel = new float [totalData];
    for(i = 0; i < totalData; i++)
        trainData[i] = new float [dictionary->rows];

     // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            //Attach the label to it
            //dataLabel->data.fl[l] = (float)data[i].getLabel();
            //dPtr = (float*)(trainData->data.ptr + l*trainData->step);
            dataLabel[l] = (float)data[i].getLabel();
            // Copy the histograms
            for(k = 0; k < length; k++)
            {
                //dPtr[k] = trainObject[i].histogramSet.histogram[j][k];
                trainData[l][k] = trainObject[i].histogramSet.histogram[j][k];
            }
        }
    }

    SVMParam_CV.svm_type = type;
    SVMParam_CV.kernel_type = kernel;
    SVMParam_CV.degree = degree;
    SVMParam_CV.gamma = gamma;
    SVMParam_CV.coef0 = coef0;
    SVMParam_CV.C = C;
    SVMParam_CV.nu = nu;
    SVMParam_CV.p = p;
    SVMParam_CV.class_weights = NULL;
    SVMParam_CV.term_crit = cvTermCriteria(termType, iterations, eps);

    CvMat *dataHeader = cvCreateMatHeader(totalData, dictionary->rows, CV_32FC1);
	CvMat *labelHeader = cvCreateMatHeader(totalData, 1, CV_32FC1);
    cvInitMatHeader(dataHeader, totalData, dictionary->rows, CV_32FC1, trainData);
	cvInitMatHeader(labelHeader, totalData, 1, CV_32FC1, dataLabel);
    //Train the SVM
    //CvSVM svm(trainData, dataLabel, 0, 0,
    //    CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 0, 0, 0, 2,
     //   0, 0, 0, cvTermCriteria(CV_TERMCRIT_EPS,0, 0.01)));

    //strcpy(classifierFile, fileName);
    if(SVMModel_CV != NULL)
        delete SVMModel_CV;

    SVMModel_CV = new CvSVM;
    SVMModel_CV->train_auto(dataHeader, labelHeader, 0, 0, SVMParam_CV, 10);
    //SVMModel_CV.save(classifierFile);

    cvReleaseMatHeader(&dataHeader);
    cvReleaseMatHeader(&labelHeader);
    for(i = 0; i < totalData; i++)
        delete [] trainData[i];
    delete [] trainData;
    delete [] dataLabel;
    classifierType = CVSVM_CLASSIFIER;

}

void BagOfFeatures::trainNormBayes_CV()
{
    int i, j, k, l = -1;
    int totalData = 0;
    int size, length = dictionary->rows;

    float *dPtr;
    //Get the total number of training data
    for(i = 0; i < numClasses; i++)
        totalData += data[i].getTrainSize();

    CvMat* trainData = cvCreateMat(totalData, dictionary->rows, CV_32FC1);
    CvMat* dataLabel = cvCreateMat(totalData, 1, CV_32FC1);

     // For each class
    for(i = 0; i < numClasses; i++)
    {
        // Get the number of images
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            l++;
            //Attach the label to it
            dataLabel->data.fl[l] = (float)data[i].getLabel();
            dPtr = (float*)(trainData->data.ptr + l*trainData->step);
            // Copy the histograms
            for(k = 0; k < length; k++)
            {
                dPtr[k] = trainObject[i].histogramSet.histogram[j][k];
            }
        }
    }

    //strcpy(classifierFile, fileName);

    if(NBModel_CV != NULL)
        delete NBModel_CV;

    NBModel_CV = new CvNormalBayesClassifier;
    NBModel_CV->train(trainData, dataLabel, 0, 0, false);
    //NBModel_CV.save(classifierFile);

    cvReleaseMat(&trainData);
    cvReleaseMat(&dataLabel);

    classifierType = CVNORM_BAYES_CLASSIFIER;

}


float* BagOfFeatures::resultsTraining()
{
    int i, j, k;
    int size;
    float classification;
    double t;
    float* results = new float [numClasses];
/*
    CvSVM SVMModel_CV;
    if(classifierType == CVSVM_CLASSIFIER)
        SVMModel_CV.load(classifierFile);
    CvNormalBayesClassifier NBModel_CV;
    if(classifierType == CVNORM_BAYES_CLASSIFIER)
        NBModel_CV.load(classifierFile);
*/
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTrainSize();
        for(j = 0; j < size; j++)
        {
            if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* trainData = new struct svm_node [dictionary->rows+1];
                for(k = 0; k < dictionary->rows; k++)
                {
                    trainData[k].index = k;
                    trainData[k].value = trainObject[i].histogramSet.histogram[j][k];
                }
                trainData[k].index = -1;

                classification = svm_predict(SVMModel, trainData);
                t = fabs((double)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] trainData;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = trainObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
            else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = trainObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);

            }

        }
        results[i] /= (float)size;
    }

    return results;
}

float* BagOfFeatures::resultsValidation()
{
    int i, j, k;
    int size;
    float classification;
    float t;

    float* results = new float [numClasses];
/*
    CvSVM SVMModel_CV;
    if(classifierType == CVSVM_CLASSIFIER)
        SVMModel_CV.load(classifierFile);
*/
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getValidSize();
        for(j = 0; j < size; j++)
        {
            if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* validData = new struct svm_node [dictionary->rows+1];
                for(k = 0; k < dictionary->rows; k++)
                {
                    validData[k].index = k;
                    validData[k].value = validObject[i].histogramSet.histogram[j][k];
                }
                validData[k].index = -1;

                classification = svm_predict(SVMModel, validData);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] validData;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = validObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
            else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = validObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);

            }

        }
        results[i] /= (float)size;
    }

    return results;
}

float* BagOfFeatures::resultsTest()
{
    int i, j, k;
    int size;
    float classification;
    float t;
    float* results = new float [numClasses];
/*
    CvSVM SVMModel_CV;
    if(classifierType == CVSVM_CLASSIFIER)
        SVMModel_CV.load(classifierFile);
*/
    for(i = 0; i < numClasses; i++)
    {
        results[i] = 0;
        size = data[i].getTestSize();
        for(j = 0; j < size; j++)
        {
           if(classifierType == LIBSVM_CLASSIFIER)
            {
                struct svm_node* testData = new struct svm_node [dictionary->rows+1];
                double* values;
                for(k = 0; k < dictionary->rows; k++)
                {
                    testData[k].index = k;
                    testData[k].value = testObject[i].histogramSet.histogram[j][k];
                }
                testData[k].index = -1;

                classification = svm_predict(SVMModel, testData);
                //svm_check_probability_model(SVMModel);
                //svm_predict_probability(SVMModel, testData, values);
                //for(k = 0; k < numClasses*(numClasses-1)/2; k++)
                //   classification = values[k];
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                delete [] testData;
                //delete [] values;
            }
            else if(classifierType == CVSVM_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = testObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = SVMModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
             else if(classifierType == CVNORM_BAYES_CLASSIFIER)
            {
                CvMat* sample = cvCreateMat(1, dictionary->rows, CV_32FC1);
                for(k = 0; k < dictionary->rows; k++)
                {
                    sample->data.fl[k] = testObject[i].histogramSet.histogram[j][k];//assign value
                }

                classification = NBModel_CV->predict(sample);
                t = fabs((float)data[i].getLabel()-classification);
                if(t < .5)
                    results[i]++;

                cvReleaseMat(&sample);
            }
        }
        results[i] /= (float)size;
    }

    return results;
}



void copySIFTPts(ImageFeatures &dst, feature* src, const int size, const int length)
{
	int i, j;
	// Check to make sure it hasn't been allocated yet
	if(dst.checkAlloc())
		dst.dealloc();
	// Allocate some memory
	dst.alloc(length, size);
	for(i = 0; i < size; i++)
	{
	    float descript[length];
	    double max = 0.0;
	    // First step is to normalize the sift vector
	    // Find the magnitude of the vector
	    for(j = 0; j < length; j++)
	    {
	        max += (float)(src[i].descr)[j];
	    }
	    double magnitude = sqrt(magnitude);
	    // Normalize by dividing by magnitude
	    for(j = 0; j < length; j++)
	    {
	        descript[j] = (float)(src[i].descr)[j];// / magnitude; /// max;
	        //cout << descript[j] << "\t";
	    }
	    //cout << endl << endl;
	    // Copy the descriptor
		dst.copyDescriptorAt(descript, i);
	}
}

void copySURFPts(ImageFeatures &dst, const IpVec src, const int length)
{
	int i, j, size;
	Ipoint temp;
	size = src.size();
	//Check if the object has been allocated
	//Deallocate it first
	if(dst.checkAlloc())
			dst.dealloc();
	// Allocated with the correct values
	dst.alloc(length, size);
	for(i = 0; i < size; i++)
	{
		temp = src.at(i);
		float mag = 0;
		for(j = 0; j < length; j++)
            mag += temp.descriptor[j]*temp.descriptor[j];
        mag = sqrt(mag);
        for(j = 0; j < length; j++)
            temp.descriptor[j] *= mag;
		// Copy each descriptor into the ImageFeature
		dst.copyDescriptorAt(temp.descriptor, i);

	}
}

int findDictionaryMatch(float* descriptor, CvMat* dict, int length)
{
    int j, k;
    int count = dict->rows;
    int minIndex;

    double minDistance = 99999999999.;
	double tempDistance = 0;

    CvMat* vect1 = cvCreateMat(1, length, CV_32FC1);
	CvMat* vect2 = cvCreateMat(1, length, CV_32FC1);
	CvMat* id = cvCreateMat(length, length, CV_32FC1);
	cvSetIdentity(id);

	float* ptr1 = NULL;
	float* ptr2 = NULL;

    ptr1 = (float *)(vect1->data.ptr);
    // Copy the vector in the vector 1;
    for(j = 0; j < length; j++)
    {
        ptr1[j] = descriptor[j];
    }

    // For each dictionary word
    for(j = 0; j < count; j++)
    {
        cvGetRow(dict, vect2, j);
        tempDistance = cvMahalanobis(vect1, vect2, id);
        /*
        // Get the second vector (word from the list)
        ptr2 = (float*)(vect2->data.ptr);
        for(k = 0; k < length; k++)
        {
            ptr2[k] = CV_MAT_ELEM(*dict, float, j, k);
        }
        tempDistance = 0;
        for(k = 0; k < length; k++)
        {
            tempDistance += (ptr2[k] - ptr1[k])*(ptr2[k] - ptr1[k]);
        }
        // calculate the euclidean distance
        tempDistance /= length; //sqrt(tempDistance);*/

        if(tempDistance < minDistance)
        {
            // get the smallest distance and keep track of the index of the min
            minDistance = tempDistance;
            minIndex = j;
        }
    }

    cvReleaseMat(&vect1);
	cvReleaseMat(&vect2);

    return minIndex;
}

IplImage* preProcessImages(const IplImage* input, int minSize, int maxSize)
{
    int width = input->width;
    int height = input->height;
    int minSide;
    double ratio;

    if(width < height)
        minSide = width;
    else
        minSide = height;

    if(minSide < minSize)
        ratio = (double)minSize / (double)minSide;
    else if(minSide > maxSize)
        ratio = (double)maxSize / (double)minSide;
    else
        ratio = 1.0;

    IplImage* temp = cvCreateImage(cvSize(width*ratio, height*ratio), input->depth, input->nChannels);
    IplImage* output = cvCreateImage(cvSize(width*ratio, height*ratio), input->depth, input->nChannels);
    //Resize based on the ratio
    cvResize(input, temp, CV_INTER_AREA);
    //Equalize the histograms of the images

    //cvEqualizeHist(temp, output);
    cvNormalize(temp, output, 0, 255, CV_MINMAX);

    cvReleaseImage(&temp);

    return output;
}
