#include "imagefeatures.h"
#include "math.h"

// Destructor
ImageFeatures::~ImageFeatures()
{
	int i;
    for(i = 0; i < size; i++)
        delete [] descriptors[i];
    delete [] descriptors;
}

// Constructor
ImageFeatures::ImageFeatures()
{
    descriptors = NULL;
    size = 0;
    length = 0;
}

ImageFeatures::ImageFeatures(const ImageFeatures &cpy)
{
    int i, j;
    size = cpy.size;
    length = cpy.length;
    descriptors = new float* [size];
    for(i = 0; i < size; i++)
    {
        descriptors[i] = new float [length];
        for(j = 0; j < length; j++)
            descriptors[i][j] = cpy.descriptors[i][j];
    }
}


// Default constructor
ImageFeatures::ImageFeatures(int len)
{
    size = 0;
    length = len;
    descriptors = NULL;
}
ImageFeatures::ImageFeatures(int len, int s)
{
    length = len;
    size = s;
    descriptors = NULL;
}

// Allocating the descriptors
void ImageFeatures::alloc(int len, int s)
{
    int i, j;
    length = len;
    size = s;
    descriptors = new float* [size];
    for(i = 0; i < size; i++)
        descriptors[i] = new float[length];
    for(i = 0; i < size; i++)
        for(j = 0; j < length; j++)
            descriptors[i][j] = 0.0;
}

// Deallocate the descriptors
bool ImageFeatures::dealloc()
{

    int i;
    if(descriptors != NULL && size > 0)
    {
        for(i = 0; i < size; i++)
            delete [] descriptors[i];
        delete [] descriptors;
        //descriptors = NULL;
        return true;
    }
    else
        return false;
}

//Check to see if the descriptor was allocated
bool ImageFeatures::checkAlloc()
{
    if(descriptors == NULL)
        return false;
    else
        return true;
}

// Copy the values in
void ImageFeatures::copyDescriptors(const float** input, int count, int len)
{
    int i, j;
    size = count;
    length = len;
    // Allocate the memory if it hasn't been allocated, for the features
    if(descriptors == NULL)
    {
        descriptors = new float* [size];
        for(i = 0; i < size; i++)
            descriptors[i] = new float[length];
    }
    // Copy all the vectors for this image
    for(i = 0; i < size; i++)
    {
        for(j = 0; j < length; j++)
        {
            descriptors[i][j] = input[i][j];
        }
    }
}

bool ImageFeatures::copyDescriptorAt(const float* vector, int location)
{
    int i;
    // Make sure the memory has been allocated or location
    // is within the bounds
    if(descriptors == NULL || location > size-1 || location < 0)
        return false;
    else
    {
        for(i = 0; i < length; i++)
            descriptors[location][i] = vector[i];
        return true;
    }
}

bool ImageFeatures::copyDescriptorAt(const double* vector, int location)
{
    int i;
    // Make sure the memory has been allocated or location
    // is within the bounds
    if(descriptors == NULL || location > size-1 || location < 0)
        return false;
    else
    {
        for(i = 0; i < length; i++)
            descriptors[location][i] = vector[i];
        return true;
    }
}

///////////////////////////////////////////////////////////////////////////
// HISTOGRAMS
///////////////////////////////////////////////////////////////////////////

HistogramFeatures::~HistogramFeatures()
{
    int i;
    for(i = 0; i < totalHist; i++)
        delete [] histogram[i];
    delete [] histogram;
    delete [] label;
    histogram = NULL;
}

HistogramFeatures::HistogramFeatures()
{
    bins = 0;
    label = NULL;
    totalHist = 0;
    histogram = NULL;
}

HistogramFeatures::HistogramFeatures(int n, int l,  int t)
{
    int i, j;
    bins = n;
    totalHist = t;
    label = new float [totalHist];
    histogram = new float* [totalHist];

    for(i = 0; i < totalHist; i++)
    {
        histogram[i] = new float [bins];
        for(j = 0; j < bins; j++)
            histogram[i][j] = 0.0;
        label[i] = l;
    }
}

bool HistogramFeatures::alloc(int n, int l, int t)
{
    int i, j;
    if(histogram == NULL)
    {
        bins = n;
        totalHist = t;
        label = new float [totalHist];
        histogram = new float* [totalHist];
        //Initialize to zero
        for(i = 0; i < totalHist; i++)
        {
            histogram[i] = new float [bins];
            for(j = 0; j < bins; j++)
                histogram[i][j] = 0.0;
            label[i] = l;
        }
        return true;
    }
    else
        return false;
}

bool HistogramFeatures::dealloc()
{
    int i;
    if(histogram != NULL)
    {
        for(i = 0; i < totalHist; i++)
            delete [] histogram[i];
        delete [] histogram;
        delete [] label;
        histogram = NULL;
        bins = 0;
        totalHist = 0;
        label = NULL;
        return true;
    }
    else
        return false;
}

float HistogramFeatures::getValAt(int i, int j)
{
    if(i > -1 && i < totalHist && j > -1 && j < bins)
        return histogram[i][j];
    else
        return -1;
}

bool HistogramFeatures::addToBin(int i, int j)
{
    if(i > -1 && i < totalHist && j > -1 && j < bins)
    {
        histogram[i][j]++;
        return true;
    }
    else
        return false;
}

// Normalize the bins in the histogram from 0 to 1
void HistogramFeatures::normalizeHist()
{
    float magnitude = 0.0;
    int i, j;
    // Find the magnitude of each element
    for(i = 0; i < totalHist; i++)
    {
        magnitude = 0.0;
        for(j = 0; j < bins; j++)
        {
            //magnitude += histogram[i][j]*histogram[i][j];
            magnitude += histogram[i][j];
        }
        //magnitude = sqrt(magnitude);
        // divide by the magnitude
        for(j = 0; j < bins; j++)
            histogram[i][j] /= magnitude;
    }
}

///////////////////////////////////////////////////
// ObjectSet
//////////////////////////////////////////////////

ObjectSet::ObjectSet()
{
    featureSet = NULL;
    setCount = 0;
}

ObjectSet::~ObjectSet()
{
    delete [] featureSet;
}

ObjectSet::ObjectSet(int l)
{
    featureSet = new ImageFeatures [l];
    setCount = l;
}

ObjectSet::ObjectSet(const ObjectSet &cpy)
{
    int i, j, k;
    setCount = cpy.setCount;
    featureSet = new ImageFeatures [setCount];
    for(i = 0; i < setCount; i++)
    {
        featureSet[i].length = cpy.featureSet[i].length;
        featureSet[i].size = cpy.featureSet[i].size;
        for(j = 0; j < cpy.featureSet[i].size; j++)
            for(k = 0; k < cpy.featureSet[i].length; k++)
                featureSet[i].descriptors[j][k] = cpy.featureSet[i].descriptors[j][k];
    }
}

bool ObjectSet::alloc(int l)
{
    if(featureSet == NULL)
    {
        featureSet = new ImageFeatures [l];
        setCount = l;
        return true;
    }
    else
        return false;
}

void ObjectSet::dealloc()
{
    delete [] featureSet;
}


