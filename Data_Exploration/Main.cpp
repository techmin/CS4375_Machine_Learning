#include "iostream"
#include "string"
#include "fstream"
#include "vector"
#include "math.h"

using namespace std;
int read();
double sum(vector<double> v);
double mean(vector<double> v);
double median(vector<double> v);
double range(vector<double> v);
double cov(vector<double> rm, vector<double> medv);
double corr(vector<double> rm, vector<double> medv);
int main(int argc,char** argv)
{
  read();
  return 0;
}

int  read()
{
  ifstream inFS;
  ofstream file_write("test.txt");
  string line;
  string rm_in, medv_in;
  const int MAX_LEN = 100;
  vector<double> rm(MAX_LEN);
  vector<double> medv(MAX_LEN);

  cout << "opening the file" << endl;

  inFS.open("Boston.csv");
  if (!inFS.is_open()) {
    cout << "Could not open file Boston.csv" << endl;
    return 1;
  }

  cout << "Reading line 1" << endl;
  getline(inFS, line);

  cout << "heading: " << line << endl;

  int numObservation = 0;
  while (inFS.good())
  {
    getline(inFS, rm_in, ',');
    getline(inFS, medv_in, '\n');

    rm.push_back(stof(rm_in));
    medv.push_back(stof(medv_in));
    
    numObservation++;

  }
  rm.resize(numObservation);
  medv.resize(numObservation);
  cout << "new length " << rm.size() << endl;

  cout << "closing " << endl;
  inFS.close();

  cout << "Number of records:" << numObservation << endl;

  cout << "\nstats for rnm " << endl;
 
  cout << "sum of rm " << sum(rm) << endl;
  cout << "sum of medv" << sum(medv) << endl;
  cout << "mean of rm " << mean(rm) << endl;
  cout << "mean of medv" << mean(medv) << endl;
  cout << "range of rm " << range(rm) << endl;
  cout << "range of medv " << range(medv) << endl;
  cout << "median of rm " << median(rm) << endl;
  cout << "median of medv " << median(medv) << endl;
  cout << "covarience " << cov(rm, medv) << endl;
  cout << "correlation " << corr(rm, medv) << endl;
  
  cout << "program ended ";
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

double range(vector<double> v)
{
  double min = LONG_MAX;
  double max = LONG_MIN;
  for (double d : v)
  {
    if (d < min)
      min = d;
    if(d>max)
      max = d;
  }
  return max - min;
}

double median(vector<double> v)
{
  if (v.size() % 2 == 0)
  {
    return (v.at(v.size() / 2) + v.at((v.size() / 2) - 2)) / 2;
  }
  else
      return v.at(v.size() / 2);
}

double cov(vector<double> rm, vector<double> medv)
{
  double sumr = sum(rm);
  double sumv = sum(medv);
  double x = 0;
  for (int i = 0; i < rm.size(); i++)
  {
    x += (rm.at(i) - sumr) * (medv.at(i) - sumv);
  }
  return x / rm.size()-1;
}

double corr(vector<double> rm, vector<double> medv)
{
  double sumr = sum(rm);
  double sumv = sum(medv);
  
  double sqr = 0;
  double sqv = 0;
  for (int i = 0; i < rm.size(); i++)
  {
    sqr += pow(rm.at(i) - sumr, 2);
    sqv += pow(medv.at(i) - sumv, 2);
  }
  sqr = sqr / rm.size();
  sqv = sqv / medv.size();

  double sdr = sqrt(sqr);
  double sdv = sqrt(sqv);
  return cov(rm, medv) / (sdr * sdv);
}
