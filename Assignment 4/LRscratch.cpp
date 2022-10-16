//Sai Gonuguntla
//Atmin Sheth 


#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

vector<double> logistic(vector<double>train,vector<double>train_surv);
vector<vector<double>> fillMatrix(vector<double> sex);
vector<double> sigmoid(vector<double> sex);
vector<double> multiply(vector<double> weights, vector<vector<double>> m2);
vector<double> predict(vector<double> test, vector<double> weights);
double calcAccuracy(vector<double> pred, vector<double> test);
double calcSensitivity(vector<double> pred, vector<double> test);
double calcSpecificity(vector<double> pred, vector<double> test);

int main(int argc, char** argv)
{
    ifstream inFS;
    string line;
    string string1_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1500;
    vector<string> string1;
    vector<double> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<double> age(MAX_LEN);
    
    cout << "Opening file titanic_project.csv." << endl;
    
    inFS.open("titanic_project.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file titanic_project.csv." << endl;
        return 1;
    }
    
    cout << "Reading line 1" << endl;
    getline(inFS, line);
    
    cout << "heading:" << line << endl;

    int numObservations = 0;
    while(inFS.good())
    {
        //Get each column
        getline(inFS, string1_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        //Store each column
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);

        numObservations++;
    }

    //Resize Vectors
    survived.resize(numObservations);
    sex.resize(numObservations);

    cout << "new length " << survived.size() << endl;
    
    cout << "Closing file titanic_project.csv." << endl;
    inFS.close();
    
    cout << "Number of records: " << numObservations << endl;
    cout << "\n";
    
    //split into test and train
    vector<double> train_surv(survived.begin(), survived.begin() + 800);
    vector<double> test_surv(survived.begin() + 801, survived.end());

    vector<double> train(sex.begin(), sex.begin() + 800);
    vector<double> test(sex.begin() + 801, sex.end());

    chrono::time_point<chrono::system_clock> start, stop;
    start = chrono::system_clock::now();
    
    //calculate weights
    vector<double> weights = logistic(train, train_surv);
    
    stop = chrono::system_clock::now();
    chrono::duration<double> elapsed_sec = (stop - start);

    cout << "Weights/coefficients: ";
    for(int i = 0; i < weights.size(); i++)
        cout << weights[i] << " ";
    
    //make predictions
    vector<double> pred = predict(test, weights);

    cout << "\n";
    
    double accr = calcAccuracy(pred, test_surv);
    double sensv = calcSensitivity(pred, test_surv);
    double spect = calcSpecificity(pred, test_surv);

    cout << "Accuracy: " << accr << endl;
    cout << "Sensitivity: " << sensv << endl;
    cout << "Specificity: " << spect << endl;
    cout << "Run Time: " << elapsed_sec.count() << endl;

    return 0;
}

vector<double> logistic(vector<double>train,vector<double>train_surv)
{
    vector<double> weights {1,1};  //set weights w0 and w1 to 1
    double learningRate = .001;

    vector<vector<double>> dataMatrix = fillMatrix(train); //set the 1st column of matrix to 1s and 2nd column to sex
    
    for(int i = 0; i < 500; i++)
    {
        //multiply the data by the weights to calculate the log likelihood
        vector<double> loglh = multiply(weights, dataMatrix);
        
        //run the values through sigmoid to get vector of probabilities
        vector<double> prob = sigmoid(loglh);

        vector<double> labels = train_surv;
        vector<double> errors(labels.size());

        //calculates the error
        for(int i = 0; i < errors.size(); i++)
            errors[i] = labels[i] - prob[i];

        //updates weights
        for(int i = 0; i < weights.size(); i++)
        {
            double gradient = 0;
            for(int j = 0; j < dataMatrix.size(); j++)
            {
                gradient += dataMatrix[j].at(i) * errors[j];
            }
            weights[i] = weights[i] + learningRate * gradient;
        }
    }
    return weights;
}

vector<vector<double>> fillMatrix(vector<double> sex)
{
    vector<vector<double>> dataMatrix(sex.size());
    
    //Create data matrix with 1s in the 1st col and sex in the 2nd
    for(int i = 0; i < sex.size(); i++)
    {
        vector<double> dataMatrix2 {1, sex[i]};
        dataMatrix[i] = dataMatrix2;
    }
    return dataMatrix;
}

//takes in the train values as input and returns a vector of sigmoid values
vector<double> sigmoid(vector<double> sex)
{
    vector<double> prob(sex.size());
    
    for(int i=0; i < sex.size(); i++)
        prob[i] = 1 / (1 + exp(-sex[i]));   //Sigmoid equation

    return prob;
}

//multiplies the weights by the data matrix or test matrix
vector<double> multiply(vector<double> weights, vector<vector<double>> m2)
{
    vector<double> row;
    double val1 = 0, val2 = 0;
    
    vector<double> logLH(m2.size());

    for(int i = 0; i < m2.size(); i++)
    {
        row = m2[i];
        val1 = weights[0] * row[0];
        val2 = weights[1] * row[1];
        logLH[i] = val1 + val2;
    }
    return logLH;
}

//Calculate probabilities and return predictions
vector<double> predict(vector<double> test, vector<double> weights)
{
    vector<vector<double>> testMatrix = fillMatrix(test);
    vector<double> predicted = multiply(weights, testMatrix);
    
    vector<double> prob(predicted.size());
    vector<double> predictions(prob.size());

    //Calculate the probabilties
    for(int i = 0; i < prob.size(); i++)
        prob[i] = exp(predicted[i]) / (1 + exp(predicted[i]));

    //Based on probabilties predict if survived or not
    for(int i = 0; i < predictions.size(); i++)
    {
        if(prob[i] > .5)
            predictions[i] = 1;
        else
            predictions[i]= 0;
    }
    return predictions;
}

//calculate Accuracy
double calcAccuracy(vector<double> pred, vector<double> test)
{
    double acc = 0;
    int tP = 0, fN = 0, fP = 0, tN = 0;

    //Count the number  true positives, false negatives, etc
    for(int i = 0; i < pred.size(); i++)
    {
        if(pred[i] == 1 && test[i] == 1)
            tP++;
        if(pred[i] == 0 && test[i] == 1)
            fN++;
        if(pred[i] == 1 && test[i] == 0)
            fP++;
        if(pred[i] == 0 && test[i] == 0)
            tN++;
    }
    
    acc = (tP + tN) / (double) (tP + tN + fP + fN);
    return acc;
}

//calculate Sensitivity
double calcSensitivity(vector<double>pred, vector<double>test)
{
    int tP = 0, fN = 0;
    double sens = 0;;

    //Count the number of number of true positives and false negatives
    for(int i = 0; i < pred.size(); i++)
    {
        if(pred[i] == 1 && test[i] == 1)
            tP++;
        if(pred[i] == 0 && test[i] == 1)
            fN++;
    }
    
    sens = tP / (double)(tP + fN);
    return sens;
}

//calculate Specificity
double calcSpecificity(vector<double> pred, vector<double> test)
{
    int tN = 0, fP = 0;
    double spec = 0;
    
    //Count the number of number of true negatives and false positives
    for(int i = 0; i < pred.size(); i++)
    {
        if(pred[i] == 0 && test[i] == 0)
            tN++;
        if(pred[i] == 1 && test[i] == 0)
            fP++;
    }
    
    spec = tN / (double)(tN + fP);
    return spec;
}

