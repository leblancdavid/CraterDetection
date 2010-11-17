
#include "bagoffeatures.h"
#include "libSVM/svm.h"

using namespace std;

// Main //

int main(int argc, char** argv)
{
    int i, j, n;
    char *classifierFile = "CVSVM.xml";
    // Creating a data set of two classes: Craters and Non-Craters
    DataSet* craterSet = new DataSet [2];

    // Set the sizes of the data
    // Split the 180 crater images into three sets:
        // training 60%, validation 20%, and test 20%
    // Set the label: 1 for CRATERS
    craterSet[0].setDataSize(323, 50, 50, 1, 256);
    //craterSet[0].setDataSize(50, 50, 50, 1, 256);
    // Split the 213 non-crater images into three sets
    // Set the label: -1 for NON-CRATERS
    craterSet[1].setDataSize(334, 50, 50, 0, 256);
    //craterSet[1].setDataSize(50, 50, 50, -1, 256);
    cout << "Loading the Data Sets..." << endl;

    // Load the data
    // Make sure it was able to load
    if(!craterSet[0].loadDataSetAt("Training-Images/craters/crater", ".jpg", 3))
        return 0;
    if(!craterSet[1].loadDataSetAt("Training-Images/non-craters/noncrater", ".jpg", 3))
        return 0;

    cout << "Shuffling the data..." << endl;
    craterSet[0].shuffleDataSet(5000);
    craterSet[1].shuffleDataSet(5000);

    cout << "Saving the shuffled data..." << endl;
    craterSet[0].saveDataSet("Results/Test-Sets/CraterSet01");
    craterSet[1].saveDataSet("Results/Test-Sets/NonCraterSet01");

    ofstream validResultsFile;
    ofstream testResultsFile;
    validResultsFile.open("Results/HCluster/ValidResultsSet01c_SIFT");
    testResultsFile.open("Results/HCluster/TestResultsSet01c_SIFT");

    cout << "Creating the Crater Bag of Features..." << endl;
    // Allocate the Bag of Features
    // Give it the number of classes: 2, and the data set
    BagOfFeatures craterBoF(2, craterSet);

    cout << "Extracting Features..." << endl;
    // Extract SURF features from the data set based on default values
    //craterBoF.extractSURFFeatures(false, 10, 10, 1, 0.00008f);
    craterBoF.extractSIFTFeatures(7, 1.6, 0.04, 10, 1, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS);


    cout << "\n\tNumber of Features extracted for training: " << craterBoF.getNumFeatures() << endl << endl;

    // Extract SIFT Features
    //craterBoF.extractSIFTFeatures(SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
     //                                   SIFT_CURV_THR, SIFT_IMG_DBL,
       //                                 SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS);

    //craterBoF.extractSIFTFeatures(5, 1.6, 0.04, 10, 1, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS);

    cout << "\n\nBuilding Hierarchical Tree..." << endl;
    // Build the hierarchical tree
    craterBoF.buildHierarchicalTree();

    for(i = 0; i < 80; i++)
    {
        n = (i+1)*25;
        cout << "\n\nCutting the tree with " << n << " clusters..."<< endl;
        // Cut the tree
        craterBoF.cutHierarchicalTree(n);

        cout << "Building the Histograms..." << endl;
        // Build the histograms
        craterBoF.buildBofHistograms(true);

        cout << "Training the System using SVM..." << endl;
        //craterBoF.trainSVM(NU_SVC, RBF, .05, .05, .5, .9, 2048, 0.00001, 0.01, 0, 1, 0);
        craterBoF.trainSVM_CV(CvSVM::C_SVC, CvSVM::RBF, 0.6, 0.5, 0.3, 0.6, 0.6, 0.5,
                              CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 500, 0.000001, classifierFile);

        float *resultsTrain = craterBoF.resultsTraining();
        cout << "Results:" << endl;
        cout << "\tTraining:\t" << resultsTrain[0] << "\t" << resultsTrain[1] << endl;

        float *resultsValid = craterBoF.resultsValidation();
        cout << "\tValidation:\t" << resultsValid[0] << "\t" << resultsValid[1] << endl;
        //Save into a file
        validResultsFile << n << "\t" << resultsValid[0] << "\t" << resultsValid[1] << endl;

        float *resultsTest = craterBoF.resultsTest();
        cout << "\tTesting:\t" << resultsTest[0] << "\t" << resultsTest[1] << endl;
        testResultsFile << n << "\t" << resultsTest[0] << "\t" << resultsTest[1] << endl;

        delete [] resultsTrain;
        delete [] resultsValid;
        delete [] resultsTest;

    }

    testResultsFile.close();
    validResultsFile.close();

    cout << "\nEND\n" << endl;
    return 0;

}
