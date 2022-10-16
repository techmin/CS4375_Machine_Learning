//
// naive_bayes.cpp
//ML-AS2
// created by Atmin Sheth and Sai Gonuguntla  on 9/29/2022
///
#define _USE_MATH_DEFINES 
#include <iostream>
#include <fstream>
#include <vector>
#include<cmath>
#include<chrono>
#include <tuple>
#include <string>
using namespace std;
using namespace std::chrono;
double mean(vector<double> v);
double sum(vector<double> v);
double stDev(vector<double>& data);
double variance(vector<double> data);
double cond_prob(vector<double> dataset, double val, vector<double> dataT, double target, double probTarget);
double quant_prob(double v, double mean, double var);
vector<double> prior(vector<double> survive);
vector<vector<double>> predict(
   const tuple<vector<double>, vector<vector<double>>, vector<double[]>, vector<double>> nb,
   const vector<vector<double>> test
);
vector<double> calc_raw_prob(vector<double> prior, vector<double> testAtr,
  tuple<vector<vector<double>>, vector<double>, vector<vector<double>>> LH
);
int main(int argc, const char* argv[])
{
  ifstream inFS;
  string line;
  string string1_in, pclass_in, survived_in, sex_in, age_in;
  const int MAX_LEN = 1500;
  vector<string> string1;
  vector<double> pclass(MAX_LEN);
  vector<double> survived(MAX_LEN);
  vector<double> sex(MAX_LEN);
  vector<double> age(MAX_LEN);

  vector<double> weight(1, 1);
  vector<double> label(MAX_LEN);


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
  while (inFS.good()) {
    getline(inFS, string1_in, ',');
    getline(inFS, pclass_in, ',');
    getline(inFS, survived_in, ',');
    getline(inFS, sex_in, ',');
    getline(inFS, age_in, '\n');


    //  string1.at(numObservations) = stof(string1_in);
    pclass.at(numObservations) = stof(pclass_in);
    survived.at(numObservations) = stof(survived_in);
    sex.at(numObservations) = stof(sex_in);
    age.at(numObservations) = stof(age_in);

    numObservations++;
  }

  //string1.resize(numObservations);
  vector<double> test_surv(survived.begin() + 801, survived.end());
  vector<double> test_pc(pclass.begin() + 801, pclass.end());
  vector<double> test_age(age.begin() + 801, age.end());
  vector<double> test_sex(sex.begin() + 801, sex.end());
  pclass.resize(800);
  survived.resize(800);
  sex.resize(800);
  age.resize(800);
  cout << "new length " << survived.size() << endl;
  //cout<< "no. of columns" << survived[0].size();

  cout << "Closing file titanic_project.csv." << endl;
  inFS.close();

  cout << "Number of records: " << numObservations << endl;

  auto start = high_resolution_clock::now();
 
  vector<double> prior_sur =  prior(survived);
  double prob_Survive = prior_sur[1];
  double prob_Nsurvive = prior_sur[0];
    
    //sex  
  
double FS = cond_prob(sex, 0, survived, 1, prob_Survive);
double MS = cond_prob(sex, 1, survived, 1, prob_Survive);
double fNS= cond_prob(sex, 0, survived, 0, prob_Nsurvive);
double mNS = cond_prob(sex, 1, survived, 0, prob_Nsurvive);

vector<vector<double>> sex_prob = { {fNS,FS},{mNS,MS} };

//pclass

double csurvied[3];
double cNotsurv[3];
for (int i = 0; i < 3; i++)
{
  csurvied[i] = cond_prob(pclass, i + 1, survived, 1, prob_Survive);
  cNotsurv[i] = cond_prob(pclass, i + 1, survived, 0, prob_Nsurvive);
}
vector<double[3]> pclass_prob = { cNotsurv, cNotsurv };
//age
vector<double> survA;
vector<double> notSurvA;

for (int i = 0; i < age.size(); i++)
{
  if (survived.at(i) == 1)
    survA.push_back(age.at(i));
  else
    notSurvA.push_back(age.at(i));
}
//vector<tuple<double,double>> age_prob={make_tuple()}
double meanAgeS = mean(survA);
double varAgeS = variance(survA);
double meanAgeNS = mean(notSurvA);
double varAgeNS = variance(notSurvA);
vector<double> mean = { meanAgeNS,meanAgeS };
vector<double> variance = { varAgeNS,varAgeS };
vector<vector<double>> age_prob = { mean,variance };
  //formating

  auto stop = high_resolution_clock::now();

  std::chrono::duration<double> elapsed_sec = stop - start;
  cout << "Time:" << elapsed_sec.count() << endl;
  cout << "Call::" << endl << "naiveBayes.default(x=x,y=y,lapace-lapace)" << endl;
  cout << "A-porior probilities: " << endl << "Y" << endl << "0     1" << endl << prior_sur[0] << "     "
    << prior_sur[1] << endl;
  //sex
  cout <<"sex "<<endl << "y  Female   male" << endl;
  cout << "0    " << fNS << "    " << mNS << endl;
  cout << "1    " << FS << "    " << MS << endl;

//pclass
  cout << "p class" << endl << "y    1      2      3" << endl;
  cout << "1     ";
  for (int i = 0; i < 3; i++)
  {
    cout << csurvied[i] << " ";
  }
  cout << endl << "0   ";
  for (int i = 0; i < 3; i++)
  {
    cout << cNotsurv[i] << " ";
  }
  cout << endl;
  //age
  cout << "age " << endl << "y " << "mean    variance" << endl;
  cout << "0  " << meanAgeNS << "  " << varAgeNS << endl;
  cout << "1  " << meanAgeS << "  " << varAgeS << endl;



  //do the testing 
//  auto nb = train_tuple(prior_sur, sex_prob, pclass_prob, age_prob);
//      prior         sex                       pclass              age
  tuple<vector<double>, vector<vector<double>>, vector<double[]>, vector<double>> nb;
  vector < vector<double>> newdata = { test_sex,test_pc,test_age };
  vector<double> pred = predict(nb,
    newdata);
  cout << "\nProgram terminated.";

  return 0;

}

double sum(vector<double> v)
{
  double s = 0;
  for (double d : v)
    s += d;
  return s;
}

double mean(vector<double> v)
{
  return sum(v) / v.size();
}

double variance(vector<double> data)
{
  double m = mean(data);
  double sqr = 0;
  for (int i = 0; i < data.size(); i++)
  {
    sqr += pow(data.at(i) - m, 2);
  }
  return  sqr /data.size();
  
}
double stDev(vector<double>& data)
{
  return sqrt(variance(data));
}
double quant_prob(double v, double mean, double var)
{
  return (1 / sqrt(2 *M_PI * var)) * exp(-((pow(v - mean, 2) / (2 * var))));
}
vector<double> prior(vector<double> survive)
{
  double s1 = 0;
  for (double d : survive)
  {
    if (d == 1)
      s1++;
  }
  double prob_Survive = s1 / (survive.size());
  return { 1 - prob_Survive,prob_Survive };
}
/*
getting the each value in the test data testAtr
and pulling the corresponding prob from LH
*/
vector<double> calc_raw_prob(vector<double> prior, vector<double> testAtr,
  tuple<vector<vector<double>>,vector<double[]>,vector<vector<double>>> LH
  )
{
  //extract the LH
  vector<vector<double>> sex = get<0>(LH);
  vector<double[]> pclass = get<1>(LH);
  vector<double> mean = get<2>(LH)[0];
  vector<double> var = get<2>(LH)[1];
  vector<double> res;
  //val of double 
  double MF = testAtr[0];
  double classes = testAtr[1];
  double age = testAtr[2];
  double denom = 0;
  for (int i = 0; i < prior.size(); i++)
  {
    double n = sex[(int)MF][i]*pclass[(int)classes][i];
    n *= quant_prob(age, mean[i], var[i]) * prior[i];
    denom += n;
    res.push_back(n);
  }

  for (int i = 0; i < res.size(); i++)
  {
    res[i] = res[i] / denom;
  }
  return res; 
}

vector<vector<double>> predict(
  const  tuple<vector<double>, vector<vector<double>>, vector<double[]>, vector<vector<double>>> nb,
   const vector< vector<double>> test
  )
{
  //extraction
  vector<double> prior = get<0>(nb);
    //test
  vector<double> sex = test[0];
  vector<double> pclass = test[1];
  vector<double> age = test[2];

  vector<vector<double>> res;
  vector<double> atr;
  for (int i = 0; i < sex.size(); i++)
  {
    for (int j = 0; j > test.size(); j++)
    {
      atr.push_back(test[j][i]);
    }
    /*
    atr.push_back(sex[i]);
    atr.push_back(pclass[i]);
    atr.push_back(age[i]);
    */
    res.push_back(calc_raw_prob(prior, atr, {
      get<1>(nb), get<2>(nb),get<3>(nb)
      }));

  }

  return res;

}

double cond_prob(vector<double> dataset, double val,vector<double> dataT, double target, double probTarget)
{
  double tot = 0;
  double count = 0;
  for (int i = 0; i < dataset.size(); i++)
  {
    if (dataset.at(i) == val)
    {
      tot++;
    //  cout << dataT.at(i) << endl;
      
      if (dataT.at(i) == target)
      {
        count++;
      }

    }
  }
  double p = count/dataset.size() ;
  return( p/probTarget);
}
/*
tuple<vector<double>, vector<vector<double>>, vector<double[3]>, vector<vector<double>>> train_tuple(
  vector<double> prior, vector<vector<double>> sex, vector<double[3]> pclass, vector<vector<double>> age)
{
  return {
    prior,sex,pclass,age
  };
}*/