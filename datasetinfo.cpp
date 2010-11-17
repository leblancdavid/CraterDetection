#include "datasetinfo.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

DataSet::DataSet()
{
    train = 0;
    valid = 0;
    test = 0;
    total = 0;
    label = 0;
    length = 0;
    dataList = NULL;
}

DataSet::DataSet(int t, int v, int s, int l, int len=64)
{
    total = t + v + s;
    train = t;
    valid = v;
    test = s;
    label = l;
    length = len;

    int i;
    dataList = new char* [total];
    for(i = 0; i < total; i++)
        dataList[i] = new char [len];
}

DataSet::DataSet(int t, double t_p, double v_p, double s_p, int l, int len=64)
{
    total = t;
    train = total * t_p;
    valid = total * v_p;
    test = total - train - valid;
    label = l;
    length = len;

    int i;
    dataList = new char* [total];
    for(i = 0; i < total; i++)
        dataList[i] = new char [len];
}

DataSet::~DataSet()
{
    int i;
    for(i = 0; i < total; i++)
        delete [] dataList[i];
    delete[] dataList;
}

DataSet& DataSet::operator=(const DataSet &rhs)
{
    int i;
    if(dataList != NULL)
    {
        for(i = 0; i < total; i++)
            delete [] dataList[i];
        delete[] dataList;
    }

    total = rhs.total;
    train = rhs.train;
    valid = rhs.valid;
    test = rhs.test;
    label = rhs.label;
    length = rhs.length;

    dataList = new char* [total];
    for(i = 0; i < total; i++)
    {
        dataList[i] = new char [length];
        strcpy(dataList[i], rhs.dataList[i]);
    }

}

char* DataSet::getDataList(int index)
{
    return dataList[index];
}

void DataSet::getDataInfo(int& t, int& v, int& s, int &l)
{
    t = train;
    v = valid;
    s = test;
    l = label;
}

int DataSet::getTrainSize()
{
    return train;
}

int DataSet::getValidSize()
{
    return valid;
}

int DataSet::getTestSize()
{
    return test;
}

int DataSet::getLabel()
{
    return label;
}

int DataSet::getTotal()
{
    return total;
}

void DataSet::setDataSize(int t, int v, int s, int l, int len=64)
{
    int i;

    if(dataList != NULL)
    {
        for(i = 0; i < total; i++)
            delete [] dataList[i];
        delete [] dataList;
    }

    total = t + v + s;
    train = t;
    valid = v;
    test = s;
    label = l;
    length = len;

    dataList = new char* [total];
    for(i = 0; i < total; i++)
        dataList[i] = new char [len];
}



bool DataSet::loadDataSetAt(const char* location, const char* type, int digit)
{
    if(dataList == NULL)
        return false;

    char temp[digit + 1];
    int i, j, index;

    for(i = 0; i < total; i++)
    {
        temp[digit] = '\0';
        index = i+1;
        for(j = 1; j <= digit; j++)
        {
            temp[digit - j] = '0' + (index % 10);
            index -= index % 10;
            index /= 10;
        }
        strcpy(dataList[i], location);
        strcat(dataList[i], temp);
        strcat(dataList[i], type);
    }

    return true;
}

bool DataSet::loadDataSetFromFile(const char* fileName)
{
    ifstream inFile;
    inFile.open(fileName);
    // Make sure it opens
    if(inFile.bad() || dataList == NULL)
        return false;

    char temp[length];
    int i = 0;
    // Copy names from the file
    while(!inFile.eof() && i < total)
    {
        inFile >> temp;
        strcpy(dataList[i], temp);
        i++;
    }

    inFile.close();
    // If file contains less data, update the total
    if(i < total)
        total = i;

    return true;
}

bool DataSet::saveDataSet(const char* fileName)
{
    int i;
    ofstream outFile;
    outFile.open(fileName);
    if(outFile.bad() || dataList == NULL)
        return false;

    for(i = 0; i < total; i++)
    {
        outFile << dataList[i] << endl;
    }

    outFile.close();
    return true;
}

void DataSet::shuffleDataSet(int times)
{
    int rand1, rand2;
    char temp[length];
    int i;
    for(i = 0; i < times; i++)
    {
        rand1 = rand() % total;
        rand2 = rand() % total;
        strcpy(temp, dataList[rand1]);
        strcpy(dataList[rand1], dataList[rand2]);
        strcpy(dataList[rand2], temp);
    }
}

