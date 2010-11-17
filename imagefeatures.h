/***********************************************************

	Image Feature class
	object to manage the features found in images
	Histogram class

************************************************************/
#include <fstream>
#include <iostream>

using namespace std;

class ImageFeatures
{
	public:
        // Destructor
        ~ImageFeatures();
        // Constructor
        ImageFeatures();
        // Copy Constructor
        ImageFeatures(const ImageFeatures &cpy);
        // Default constructor
        ImageFeatures(int len);
        ImageFeatures(int len, int s);

        // Allocating the descriptors
        void alloc(int len, int s);
        // Deallocate the descriptors
        bool dealloc();
        //Check to see if the descriptor was allocated
        bool checkAlloc();

        // Copy the values in
        void copyDescriptors(const float** input, int count, int len);
        bool copyDescriptorAt(const float* vector, int location);
        bool copyDescriptorAt(const double* vector, int location);

        float** descriptors;
        int size;
        int length;

};

class HistogramFeatures
{
	public:

        ~HistogramFeatures();
        HistogramFeatures();
        HistogramFeatures(int n, int l,  int t);

        bool alloc(int n, int l, int t);
        bool dealloc();

        float getValAt(int i, int j);
        bool addToBin(int i, int j);
        // Normalize the bins in the histogram from 0 to 1
        void normalizeHist();

        int bins;
        int totalHist;
        float *label;
        float **histogram;
};

class ObjectSet
{
    public:
        ~ObjectSet();
        ObjectSet();
        ObjectSet(const ObjectSet &cpy);
        ObjectSet(int l);

        bool alloc(int l);
        void dealloc();

        ImageFeatures* featureSet;
        HistogramFeatures histogramSet;
        int setCount;
};

