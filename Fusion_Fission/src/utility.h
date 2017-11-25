#ifndef UTILITY_H
#define UTILITY_H

const double fix_alpha_s = 0.3;
const double Nc = 3., TF = 0.5, CF = 4./3.;
const double rho_c = 0.5;
const double M = 1.29; //[GeV]
const double a_B = 3./fix_alpha_s/M;
const double E1S = fix_alpha_s*fix_alpha_s*M/9.;
const double M1S = 2.*M - E1S;
const double InverseFermiToGeV = 0.197327;

#endif
