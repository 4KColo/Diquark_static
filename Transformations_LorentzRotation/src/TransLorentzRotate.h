#ifndef TRANSLORENTZROTATE_H        // check if DISREC_H has been defined, if not, then define
#define TRANSLORENTZROTATE_H

#include <vector>

using namespace std;

vector<double> lorentz_transform(vector<double> momentum_4, vector<double> velocity_3);
vector<double> rotation_transform(vector<double> vector_3, double theta, double phi);
vector<double> angle_find(vector<double> vector_3);


// define these functions in .h file so that cython can call; for what these functions really do, the compiler will refer to cpp(cxx) file
#endif
