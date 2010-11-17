
class DataSet
{
    public:
        // Constructor
        DataSet();
        // Set the number of train, validation, and test data
        DataSet(int t, int v, int s, int l, int len);
        // Set the total number of images, and the % of each set
        DataSet(int t, double t_p, double v_p, double s_p, int l, int len);
        // Destructor
        ~DataSet();

        DataSet &operator=(const DataSet &rhs);

        // Get the information
        void getDataInfo(int& t, int& v, int& s, int &l);
        int getTrainSize();
        int getValidSize();
        int getTestSize();
        int getLabel();
        int getTotal();
        char* getDataList(int index);

        void setDataSize(int t, int v, int s, int l, int len);

        // Load the image names that are located at a specified point
        bool loadDataSetAt(const char* location, const char* type, int digit);
        // Load the image names from a file
        bool loadDataSetFromFile(const char* fileName);
        // Save the set into a file
        bool saveDataSet(const char* fileName);

        // Shuffles the data arround for validation
        void shuffleDataSet(int times);

    private:
        int train;
        int valid;
        int test;
        int total;
        int label;

        char** dataList;
        int length;
};
