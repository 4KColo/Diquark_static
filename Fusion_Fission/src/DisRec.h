#ifndef DISREC_H        // check if DISREC_H has been defined, if not, then define
#define DISREC_H

#include "utility.h"
#include <vector>

const double MS_1S_prefactor = pow(2., 9) * pow(M_PI, 2)
							 * pow(a_B, 5) * pow(2.+rho_c, 2);    // matrix element squared
const double rho_c_sq = pow(rho_c, 2);
const double MS_1S_scale = fix_alpha_s*M/(2.*Nc);   // for calculation of "eta"
const double TwoPi = 2.*M_PI;

const double X_1S_prefactor = fix_alpha_s/3.0 * pow(2.0,10)
							* pow(M_PI, 2) * rho_c*pow(2.0+rho_c, 2)
							* pow(E1S, 4) / M;      // cross section
const double small_number = 1e-4;
const double reco_1S_prefactor = fix_alpha_s / 3.;


double M2_1S_to_Psi_p(double p);
double X_1S_dis(double q);
double X_1S_dis_syn(double q);
double fac1(double z);
double dRdq_1S(double q, void * params_);
double dRdq_1S_small_v(double q, void * params_);
double R_1S_decay(double vabs, double T);
double decay_sample_1S_dRdq(double v, double T, double maximum);
double decay_sample_1S_costheta_q(double q, double v, double T);
double decay_sample_1S_final_p(double q);
double find_max(double(*f)(double x, void * params), void * params, 
				double xL, double xR);
double qdRdq_1S_u(double u, void * params);
double RtimesV_1S_reco(double v, double T, double p);
double dist_position(double r);
double BEplus1(double x);
double reco_sample_1S_q(double p);
double reco_sample_1S_costheta(double v, double T, double q);


// define these functions in .h file so that cython can call; for what these functions really do, the compiler will refer to cpp(cxx) file
#endif
