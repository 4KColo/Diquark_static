#include <iostream>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include "DisRec.h"
#include <random>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist_u(0., 4.);
std::uniform_real_distribution<double> rejection(0., 1.);
std::uniform_real_distribution<double> y_cdf(0., 1.);
std::uniform_real_distribution<double> reco_uniform(-1., 1.);
std::uniform_real_distribution<double> reco_reject(0., 1.);


// static and moving frame statistical
double BE(double x){
	return 1./(std::exp(x)-1.);
}

double BEplus1(double x){
    return 1./(1. - std::exp(-x));
}

double fac1(double z){
	return std::log(1. - std::exp(-z));
}


// find maximum of a function
// only works for positive-function with one local maximum within [xL, xH]
double find_max(double(*f)(double x, void * params), void * params, double xL_, double xR_){
	double dfL, dfR, dfM, xM, xL = xL_, xR = xR_, dx, fM;
	dx = (xR-xL)/20.;
	dfL = f(xL+dx, params) - f(xL-dx, params); 
	dfR = f(xR+dx, params) - f(xR-dx, params);
	do{
		xM = (xL+xR)/2.;
		dfM = f(xM+dx, params) - f(xM-dx, params);
		fM = f(xM, params);
		if (dfL*dfM < 0) {xR = xM; dfR = dfM;}
		if (dfM*dfR < 0) {xL = xM; dfL = dfM;}
		dx = (xR-xL)/20.;
	}while ( std::abs(dfM/dx/fM) > 1e-3 );
	return fM;
}


//------------------------------- Diquark dissociation --------------------------------
// Matrix element of |<1S|Psi(p)>|^2, (Coulomb interaction)
double M2_1S_to_Psi_p(double p){
	double eta = MS_1S_scale/(p+small_number);
	double x = a_B * p;
	double Numerator = MS_1S_prefactor*eta*(rho_c_sq+x*x)*std::exp(4.0*eta*std::atan(x));
	double Denominator = pow(1.0+x*x, 6) * ( std::exp(TwoPi*eta) - 1.0 );
	return Numerator/Denominator;
}

// Two formula (with the same reuslts) of the Cross-section of 1S -> Psi(p)
// first one
double X_1S_dis(double q){
	if (q <= E1S) return 0.;
	double t1 = std::sqrt(q/E1S - 1.);
	double x = rho_c/t1;
	return X_1S_prefactor/pow(q, 5) * (t1*t1 + rho_c_sq)
		 * std::exp(4.*x*std::atan(t1)) / (std::exp(TwoPi*x) - 1.);
}

//second one
double X_1S_dis_syn(double q){
	if (q <= E1S) return 0.;
	double p = std::sqrt(M*(q-E1S));
	return 2./3.*fix_alpha_s*M*q*p*M2_1S_to_Psi_p(p);
}

// differential decay rate dR/dq, q is the momentum of incident gluon
// and its approximation form in the limit of small v
double dRdq_1S(double q, void * params_) {
    double * params = static_cast<double *>(params_);
    double k1 = params[0]; //k1 is gamma*(1.+v)/T;
    double k2 = params[1]; //k2 is gamma*(1.-v)/T;
    return q*X_1S_dis(q)*(fac1(q*k1) - fac1(q*k2));
}

double dRdq_1S_small_v(double q, void * params_) {
    double * params = static_cast<double *>(params_);
    double T = params[0];
    return q*q*X_1S_dis(q)*BE(q/T);
}

double qdRdq_1S_u(double u, void * params_){
    double * params = static_cast<double *>(params_);
    double v = params[0];
    double T = params[1];
    v = std::max(v, small_number);
    double gamma = 1./sqrt(1.-v*v);
    double new_params[2];
    new_params[0] = gamma*(1.+v)/T;
    new_params[1] = gamma*(1.-v)/T;
    double q = E1S*std::exp(u);
    return q*dRdq_1S(q, new_params);
}

// sample dR/dq (rejection) and dR/dq/dcostheta_q (inverse CDF)
double decay_sample_1S_dRdq(double v, double T, double maximum){
	// uniformly sampling u in [0,4]
	// q = E1S*exp(u)
	// dRdu = dRdq(q(u))*dq/du = dRdq(q(u))*q(u)
	double u, result, params[2];
	params[0] = v; params[1] = T;
	do{
		u = dist_u(gen);
		result = qdRdq_1S_u(u, params)/maximum;
	} while( rejection(gen) > result );
	return E1S*std::exp(u);
}

double decay_sample_1S_costheta_q(double q, double v, double T){
	v = std::max(v, small_number);
    double coeff = q/T/std::sqrt(1. - v*v);     // parameter B
	double low = fac1(coeff*(1.-v));
	double norm = fac1(coeff*(1.+v))-low;
	double y_fac1 = y_cdf(gen)*norm + low;
	return ( -std::log(1. - std::exp(y_fac1))/coeff - 1. )/v;
}

// return the final relative momentum between Q-Q pair
double decay_sample_1S_final_p(double q){
    return std::sqrt((q-E1S)*M);
}


// this function intergate dR/dq to get R_decay
// this is used for tabulation, not for monte-carlo simulation.
// there is a interpolation function that interpolates the table for MC-simulation.
double R_1S_decay(double vabs, double T){
	if (vabs < small_number){
		double result, error, qmin=E1S, qmax=E1S*100.;
		gsl_integration_workspace *w = gsl_integration_workspace_alloc(5000);
		double * params = new double[1];
		params[0] = T;
        gsl_function F;
		F.function = dRdq_1S_small_v;
		F.params = params;
		gsl_integration_qag(&F, qmin, qmax, 0, 1e-6, 1000, 6, w, &result, &error);
        delete [] params;
		gsl_integration_workspace_free(w);
		return result / (TwoPi*M_PI);
	}
	else{
		double gamma = 1./std::sqrt(1. - vabs*vabs);
		double k1 = gamma*(1.+vabs)/T;
		double k2 = gamma*(1.-vabs)/T;
		double result, error, qmin=E1S, qmax=E1S*100.;
		gsl_integration_workspace *w = gsl_integration_workspace_alloc(5000);
		double * params = new double[2];
		params[0] = k1;
		params[1] = k2;
        gsl_function F;
		F.function = dRdq_1S;
		F.params = params;
		gsl_integration_qag(&F, qmin, qmax, 0, 1e-6, 1000, 6, w, &result, &error);
        delete [] params;
		gsl_integration_workspace_free(w);
		return result * T / (TwoPi*TwoPi*vabs*gamma*gamma);
	}
}
//-------------------------- end of diquark dissociation --------------------------------



// ------------------------- Quark-quark recombination ------------------------------
// here we define rate * vol = v_rel * cross section in GeV*fm^3; p is the QQbar relative momentum
// rate * vol for a specific color, no summation over colors
double RtimesV_1S_reco(double v, double T, double p){
	double q = p*p/M + E1S;
    double r = reco_1S_prefactor * pow(q,3) *M2_1S_to_Psi_p(p)* pow(InverseFermiToGeV,3);
// convert GeV^-2 to GeV * fm^3
    if (v < small_number){
		double enhencement = BEplus1(q/T);
		return 2.* r * enhencement;
	}
	else{
		double gamma = 1./std::sqrt(1.-v*v);
		double k1 = gamma*(1.+v)/T, k2 = gamma*(1.-v)/T;
		double enhencement = 2. + T/(gamma*q*v)*( fac1(q*k1) - fac1(q*k2) );
		return r * enhencement;
	}
}

// no factor of 2 here, choose to add factor 2 when adding theta function
double dist_position(double r){
    double sigma = a_B * InverseFermiToGeV;
    return std::exp( -r*r/(2.0*sigma*sigma) )/( pow(TwoPi*sigma*sigma, 1.5) );
}


// now sampling
double reco_sample_1S_q(double p){
    double q = p*p/M + E1S;
    return q;
}

// input the above q into the costheta sampling
double reco_sample_1S_costheta(double v, double T, double q){
    double gamma = 1./std::sqrt(1.-v*v);
    double y1 = q*gamma*(1.-v)/T;
    double max_value = BEplus1(y1);
    double x_try, y_try, result;
    do {
        x_try = reco_uniform(gen);
        y_try = q*gamma*(1.+ x_try*v)/T;
        result = BEplus1(y_try);
    } while (reco_reject(gen)*max_value > result);
    return x_try;
}

// ---------------------------- end of recombination ------------------------------




