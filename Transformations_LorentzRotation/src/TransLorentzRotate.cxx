#include <iostream>
#include <cmath>
#include "TransLorentzRotate.h"
#include <vector>


// ---------------------------- lorentz transformation ----------------------------
std::vector<double> lorentz_transform(std::vector<double> momentum_4, std::vector<double> velocity_3){
    double vx = velocity_3[0];
    double vy = velocity_3[1];
    double vz = velocity_3[2];
    double v_sq = vx*vx + vy*vy + vz*vz;
    
    if (v_sq == 0.0){
        return momentum_4;
    }
    else{
        double E = momentum_4[0];
        double px = momentum_4[1];
        double py = momentum_4[2];
        double pz = momentum_4[3];
        
        double gamma = 1./std::sqrt(1.-v_sq);
        double vdotp = vx*px + vy*py + vz*pz;
        
        std::vector<double> p4_new(4);
        p4_new[0] = gamma*(E - vdotp);
        p4_new[1] = -gamma*vx*E + px + (gamma-1.)*vx*vdotp/v_sq;
        p4_new[2] = -gamma*vy*E + py + (gamma-1.)*vy*vdotp/v_sq;
        p4_new[3] = -gamma*vz*E + pz + (gamma-1.)*vz*vdotp/v_sq;
        return p4_new;
    }
}

// ------------------------ end of lorentz transformation ----------------------------



// ---------------------------- rotation transformation ------------------------------
std::vector<double> rotation_transform(std::vector<double> vector_3, double theta, double phi){
    double vx = vector_3[0];
    double vy = vector_3[1];
    double vz = vector_3[2];
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    
    std::vector<double> v3_new(3);
    v3_new[0] = cos_theta*cos_phi*vx - sin_phi*vy + sin_theta*cos_phi*vz;
    v3_new[1] = cos_theta*sin_phi*vx + cos_phi*vy + sin_theta*sin_phi*vz;
    v3_new[2] = -sin_theta*vx + cos_theta*vz;
    return v3_new;
}


// ------------------------ end of rotation transformation ---------------------------



// ------------------------ find theta and phi of a vector ---------------------------
std::vector<double> angle_find(std::vector<double> vector_3){
    double vx = vector_3[0];
    double vy = vector_3[1];
    double vz = vector_3[2];
    double vabs = std::sqrt(vx*vx + vy*vy + vz*vz);
    
    std::vector<double> theta_phi(2);
    if (vabs==0.){
        theta_phi[0] = 0.;
    }
    else{
        theta_phi[0] = std::acos(vz/vabs);
    }
    if (vx > 0.){
        theta_phi[1] = std::atan(vy/vx);
    }
    else if (vx == 0.){
        if (vy > 0.){
            theta_phi[1] = 0.5*M_PI;
        }
        else if (vy == 0.){
            theta_phi[1] = 0.;
        }
        else{
            theta_phi[1] = -0.5*M_PI;
        }
    }
    else{
        theta_phi[1] = std::atan(vy/vx) + M_PI;
    }
    return theta_phi;
}

// ---------------------- end of find theta and phi of a vector ----------------------

