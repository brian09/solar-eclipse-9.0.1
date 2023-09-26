#include <Eigen/Dense>
#include <iostream>
#include "solar.h"
#include <fstream>
#include "solar_mle_setup.h"
#include <cmath>
#include <string>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>
using namespace std;
#define MAX_ITERATIONS 500
#define MAX_DELTA_ERROR 1e-07
#define MAX_LOGLIK_ERROR 1e-08
#define DP 1.0E-4
#define STOP_CRITERIA 1.0E-3
#define T 1.0

extern "C" void cdfchi_ (int*, double*, double*, double*, double*,
                         int*, double*);
static double chicdf(double chi, double df){
    double p, q, bound;
    int status = 0;
    int which = 1;
    
    
    cdfchi_ (&which, &p, &q, &chi, &df, &status, &bound);
    
    return q/2.0;
}
static inline double calculate_constraint(const double x){
    //return exp(x)/(1 + exp(x));
    return x*x/(1.0 + x*x);

}
static double reverse_constraint(double x){
    return sqrt(x/(1-x));
    
    
}
static inline double calculate_dconstraint(const double x){

    return 2*x/pow(1+x*x, 2);
    //const double e_x = exp(x);
	//return e_x*pow(e_x + 1, -2);
	
}


static inline double calculate_ddconstraint(const double x){

    return -2*(3*x*x - 1)/pow((x*x + 1), 3);
	//const double e_x = exp(x);
	//return -(e_x-1.0)*e_x*pow(e_x+1,-3);
	
}

/*
static double sd_error(const double theta_g, const double theta_e, const double se_g , const double se_e){
    const double g_error = se_g*0.5*pow(theta_g + theta_e, -0.5);
    const double e_error = se_e*0.5*pow(theta_g + theta_e, -0.5);
    return sqrt(g_error*g_error + e_error*e_error);
}
*/

/*
static double rho_error(const double covar, const double theta_one, const double theta_two,\
    const double se_covar, const double se_theta_one, const double se_theta_two){
    const double covar_error = se_covar*pow(theta_one*theta_two, -0.5);
    const double theta_one_error = -0.5*se_theta_one*covar*theta_two*pow(theta_one*theta_two, -1.5);
    const double theta_two_error = -0.5*se_theta_two*covar*theta_one*pow(theta_one*theta_two, -1.5);
    return sqrt(covar_error*covar_error + theta_one_error*theta_one_error + theta_two_error*theta_two_error);
}

static double h2r_error(const double theta_g, const double var, const double se_g, const double se_var){
//    const double g_error = se_g*theta_e*pow(theta_g + theta_e, -2.0);
//    const double e_error = se_e*theta_g*pow(theta_g + theta_e, -2.0);
    const double g_error = se_g/var;
    const double sd_error = -se_var*theta_g*pow(var, -2.0);
    return sqrt(g_error*g_error + sd_error*sd_error);
}

static double sd_error(const double var, const double se_var){
    return 0.5*se_var*pow(var, -0.5);
}

static double rhog_error(const double covar, const double theta_one, const double theta_two,\
    const double se_covar, const double se_theta_one, const double se_theta_two){
    const double covar_error = se_covar*pow(theta_one*theta_two, -0.5);
    const double theta_one_error = -0.5*se_theta_one*covar*theta_two*pow(theta_one*theta_two, -1.5);
    const double theta_two_error = -0.5*se_theta_two*covar*theta_one*pow(theta_one*theta_two, -1.5);
    return sqrt(covar_error*covar_error + theta_one_error*theta_one_error + theta_two_error*theta_two_error);
}

static double rhoe_error(const double covar, const double theta_one, const double theta_two,\
    const double var_one, const double var_two, const double se_covar, const double se_theta_one,\
    const double se_theta_two, const double se_var_one,const double se_var_two){
    const double covar_error = se_covar*pow((var_one-theta_one)*(var_two-theta_two), -0.5);
    const double var_one_error = -0.5*se_var_one*(var_two-theta_two)*covar*pow((var_two-theta_two)*(var_one-theta_one), -1.5);
    const double var_two_error = -0.5*se_var_two*(var_one-theta_one)*covar*pow((var_two-theta_two)*(var_one-theta_one), -1.5);
    const double theta_one_error = 0.5*se_theta_one*(var_two-theta_two)*covar*pow((var_two-theta_two)*(var_one-theta_one), -1.5);
    const double theta_two_error = 0.5*se_theta_two*(var_one-theta_one)*covar*pow((var_two-theta_two)*(var_one-theta_one), -1.5);    
    return sqrt(covar_error*covar_error + theta_one_error*theta_one_error + theta_two_error*theta_two_error +\
             var_one_error*var_one_error + var_two_error*var_two_error);
}
*/

static double h2r_error(double theta_g, double theta_e, double se_theta_g, double se_theta_e){
    const double theta_g_error = se_theta_g*theta_e*pow(theta_e + theta_g, -2);
    const double theta_e_error = -se_theta_e*theta_g*pow(theta_e + theta_g, -2);
    return sqrt(theta_e_error*theta_e_error + theta_g_error*theta_g_error);
}

static double sd_error(double theta_g, double theta_e, double se_theta_g, double se_theta_e){
    const double theta_g_error = 0.5*se_theta_g*pow(theta_e + theta_g, -0.5);
    const double theta_e_error = 0.5*se_theta_e*pow(theta_e + theta_g, -0.5);
    return sqrt(theta_g_error*theta_g_error + theta_e_error*theta_e_error);

}
static double rho_error(double t, double se_t){
    return se_t*2.0/(M_PI*(t*t + 1));
}

static void compute_blockwise_diagonal_inversions(Eigen::VectorXd & A, Eigen::VectorXd & BC, \
                                                     Eigen::VectorXd & D){
/*
   Eigen::VectorXd denom = A.cwiseProduct(D) - BC.cwiseAbs2();
    denom = denom.cwiseInverse();
    BC = -BC.cwiseProduct(denom);
    Eigen::VectorXd old_A = A;
    A = D.cwiseProduct(denom);
    D = old_A.cwiseProduct(denom);*/
    Eigen::VectorXd A_inverse = A.cwiseInverse();
    Eigen::VectorXd D_inverse = D.cwiseInverse();
    
    Eigen::VectorXd E = (D-BC.cwiseAbs2().cwiseProduct(A_inverse)).cwiseInverse();
        
    Eigen::VectorXd new_A = A_inverse + A_inverse.cwiseAbs2().cwiseProduct(BC.cwiseAbs2().cwiseProduct(E));
    Eigen::VectorXd new_BC = -A_inverse.cwiseProduct(BC.cwiseProduct(E));
    Eigen::VectorXd new_D = E;
    A = new_A;
    BC = new_BC;
    D = new_D;
}
static double calculate_rho(const double x){
   // return x;
    return atan(x)*2.0/M_PI;
}
static double reverse_rho(double x){
        return tan(x*M_PI/2.0);
}
static double calculate_rho_dconstraint(const double x){
    return 2.0/((1.0 + x*x)*M_PI);
   
}
static double calculate_rho_ddconstraint(const double x){
    return -4.0*x/(pow((1.0 + x*x), 2)*M_PI);
   
}
static Eigen::VectorXd combine_parameters_and_beta(Eigen::VectorXd parameters, Eigen::VectorXd beta){
	Eigen::VectorXd all_parameters(parameters.rows() + beta.rows());
	for(int i = 0; i < parameters.rows(); i++){
		all_parameters(i) = parameters(i);
	}

	for(int i = 0; i < beta.rows(); i++){
		all_parameters(parameters.rows() + i) = beta(i);
	}
	return all_parameters;
}
/*
static double calculate_loglikelihood_param(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd parameters, const bool use_constraints = true){
                   
    Eigen::VectorXd omega_one_one;// = omega_1_1 = sd_one*sd_one*(h2r_one*lambda + (1.0-h2r_one)*Eigen::VectorXd::Ones(lambda.rows()));
    Eigen::VectorXd omega_one_two;
     Eigen::VectorXd omega_two_two;
     const double h2r_one = (use_constraints) ? calculate_constraint(parameters(0)) : parameters(0);
     const double h2r_two = (use_constraints) ? calculate_constraint(parameters(1)) : parameters(1);
     const double rhog = (use_constraints) ? calculate_rho(parameters(4)) : parameters(4);
     const  double rhoe =  (use_constraints) ? calculate_rho(parameters(5)) : parameters(5);
     const double sd_one = fabs(parameters(2));
     const  double sd_two = fabs(parameters(3));
      omega_one_one =  sd_one*sd_one*(h2r_one*lambda + (1.0-h2r_one)*Eigen::VectorXd::Ones(lambda.rows()));
      omega_one_two = sd_one*sd_two*(lambda*sqrt(h2r_one)*sqrt(h2r_two)*rhog + rhoe*sqrt(1.0-h2r_one)*sqrt(1.0-h2r_two)*Eigen::VectorXd::Ones(lambda.rows()));
      omega_two_two = sd_two*sd_two*(h2r_two*lambda + (1.0-h2r_two)*Eigen::VectorXd::Ones(lambda.rows()));
    

    Eigen::VectorXd beta_one(covariate_matrix.cols());
    Eigen::VectorXd beta_two(covariate_matrix.cols());
    for(int i = 0 ; i < covariate_matrix.cols(); i++){
        beta_one(i) = parameters(i + 6);
        beta_two(i) = parameters(i + covariate_matrix.cols() + 6);
    }
    Eigen::VectorXd residual_one = Y_one - covariate_matrix*beta_one;//parameters(0);
    Eigen::VectorXd residual_two = Y_two - covariate_matrix*beta_two;//parameters(1);

     
    
    double part_one = 0.0;

   Eigen::VectorXd omega_det = omega_one_one.cwiseProduct(omega_two_two) - omega_one_two.cwiseAbs2();
    for(int i = 0; i < lambda.rows() ; i++){
        part_one += log(fabs(omega_det(i)));
    }

//    compute_blockwise_diagonal_inversions(omega_one_one, omega_one_two, omega_two_two);
   
    omega_det = omega_det.cwiseInverse();
    const double part_two = residual_one.dot(residual_one.cwiseProduct(omega_two_two.cwiseProduct(omega_det))) + residual_two.dot(residual_two.cwiseProduct(omega_one_one.cwiseProduct(omega_det))) - 2.0*residual_one.dot(omega_one_two.cwiseProduct(residual_two).cwiseProduct(omega_det));
    //if(use_constraints)
        

    return -0.5*(part_one + part_two);
}*/

static int display_totals  = 0;
static double calculate_loglikelihood_param(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd parameters, const int use_constrain_function = 1){
                   
    Eigen::VectorXd omega_one_one;// = omega_1_1 = sd_one*sd_one*(h2r_one*lambda + (1.0-h2r_one)*Eigen::VectorXd::Ones(lambda.rows()));
    Eigen::VectorXd omega_one_two;
    Eigen::VectorXd omega_two_two;
    if(parameters(0) < 0.0) parameters(0) = abs(parameters(0));
    if(parameters(1) < 0.0) parameters(1) = abs(parameters(1));
    if(parameters(2) < 0.0) parameters(2) = abs(parameters(2));
    if(parameters(3) < 0.0) parameters(3) = abs(parameters(3));
     /*
     const double h2r_one = parameters(0)/(parameters(0) + parameters(2));//(use_constraints) ? calculate_constraint(parameters(0)) : parameters(0);
     const double h2r_two = parameters(1)/(parameters(1) + parameters(3));;//(use_constraints) ? calculate_constraint(parameters(1)) : parameters(1);
     const double sd_one = sqrt(parameters(0) + parameters(2));
     const double sd_two = sqrt(parameters(1) + parameters(3));
     const double rhog = parameters(4)/(sqrt(parameters(0)*parameters(1)));//(use_constraints) ? calculate_rho(parameters(4)) : parameters(4);
     const double rhoe = parameters(5)/(sqrt(parameters(2)*parameters(3)));
*/

     double h2r_one = (use_constrain_function == 1) ? parameters(0)/(parameters(0) + parameters(2))  : parameters(0);//(use_constraints) ? calculate_constraint(parameters(0)) : parameters(0);
     const double h2r_two = (use_constrain_function == 1) ?  parameters(1)/(parameters(1) + parameters(3)) : parameters(1);//(use_constraints) ? calculate_constraint(parameters(1)) : parameters(1);
     const double sd_one = (use_constrain_function == 1) ? sqrt(parameters(0) + parameters(2)) : parameters(2);
     const double sd_two = (use_constrain_function == 1) ? sqrt(parameters(1) + parameters(3)) : parameters(3);
     const double rhog = (use_constrain_function == 1) ? calculate_rho(parameters(4)) : parameters(4);//parameters(4)/(sqrt(parameters(0)*parameters(1)));//(use_constraints) ? calculate_rho(parameters(4)) : parameters(4);
     const double rhoe = (use_constrain_function == 1) ? calculate_rho(parameters(5)) : parameters(5);//parameters(5)/(sqrt((parameters(2)-parameters(0))*(parameters(3)-parameters(1))));     

      omega_one_one =  sd_one*sd_one*(h2r_one*lambda + (1.0-h2r_one)*Eigen::VectorXd::Ones(lambda.rows()));
      omega_one_two = sd_one*sd_two*(lambda*sqrt(h2r_one)*sqrt(h2r_two)*rhog + rhoe*sqrt(1.0-h2r_one)*sqrt(1.0-h2r_two)*Eigen::VectorXd::Ones(lambda.rows()));
      omega_two_two = sd_two*sd_two*(h2r_two*lambda + (1.0-h2r_two)*Eigen::VectorXd::Ones(lambda.rows()));
    

    Eigen::VectorXd beta_one(covariate_matrix.cols());
    Eigen::VectorXd beta_two(covariate_matrix.cols());
    for(int i = 0 ; i < covariate_matrix.cols(); i++){
        beta_one(i) = parameters(i + 6);
        beta_two(i) = parameters(i + covariate_matrix.cols() + 6);
    }
    Eigen::VectorXd residual_one = Y_one - covariate_matrix*beta_one;//parameters(0);
    Eigen::VectorXd residual_two = Y_two - covariate_matrix*beta_two;//parameters(1);

     
    
    double part_one = 0.0;

   Eigen::VectorXd omega_det = omega_one_one.cwiseProduct(omega_two_two) - omega_one_two.cwiseAbs2();
    for(int i = 0; i < lambda.rows() ; i++){
        part_one += log(fabs(omega_det(i)));
    }

//    compute_blockwise_diagonal_inversions(omega_one_one, omega_one_two, omega_two_two);
   
    omega_det = omega_det.cwiseInverse();
    const double part_two = residual_one.dot(residual_one.cwiseProduct(omega_two_two.cwiseProduct(omega_det)));
    const double part_three = residual_two.dot(residual_two.cwiseProduct(omega_one_one.cwiseProduct(omega_det)));
    const double part_four = residual_one.dot(omega_one_two.cwiseProduct(residual_two).cwiseProduct(omega_det));
    const double total = part_two + part_three - 2.0*part_four;// residual_one.dot(residual_one.cwiseProduct(omega_two_two.cwiseProduct(omega_det))) + residual_two.dot(residual_two.cwiseProduct(omega_one_one.cwiseProduct(omega_det))) - 2.0*residual_one.dot(omega_one_two.cwiseProduct(residual_two).cwiseProduct(omega_det));
    //if(use_constraints)
 /*   if(display_totals == 1){
        cout << "p1: " << part_two << endl;
        cout << "p2: " << part_three << endl;
        cout << "p3: " << part_four << endl;
        cout << "total: " << total << endl;
        cout << "var part: " << part_one << endl;
    }*/

    return -0.5*(part_one + total);
}



static double calculate_gradient(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two,\
                        Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, \
                        Eigen::VectorXd all_parameters, const int index, int use_constrain_function = 1){

    double h;
 //  is_hessian = true;
  // if(is_hessian){
        h = pow(DP, 0.66667)*max(abs(all_parameters(index)), 1.0);
 //  }else{
  //      h = DP*max(abs(all_parameters(index)), 1.0);
  //  }
    Eigen::VectorXd positive_parameters = all_parameters;
    positive_parameters(index) += h;
    Eigen::VectorXd negative_parameters =all_parameters;
    negative_parameters(index) -= h;
    return (calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
     lambda,  positive_parameters, use_constrain_function) - calculate_loglikelihood_param(Y_one,  \
     Y_two,  covariate_matrix, lambda,  negative_parameters, use_constrain_function))/(2.0*h);
}
static double calculate_hessian(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two,\
                        Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, \
                        Eigen::VectorXd all_parameters, const int index_one,\
                        const int index_two, int use_constrain_function = 1){
    Eigen::VectorXd positive_parameters = all_parameters;
    double h = DP*max(abs(all_parameters(index_two)), 1.0);;
    positive_parameters(index_two) += h;
    
    return ((calculate_gradient(Y_one, Y_two, covariate_matrix, \
        lambda, positive_parameters,  index_one, use_constrain_function)) - calculate_gradient(Y_one, Y_two, covariate_matrix, \
        lambda, all_parameters,  index_one, use_constrain_function))/(h);
}
static Eigen::VectorXd calculate_gradient_vector(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two,\
                        Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, \
                        Eigen::VectorXd all_parameters){
    Eigen::VectorXd gradient(all_parameters.rows());

    for(int i = 0 ; i < all_parameters.rows(); i++){
        gradient(i) = calculate_gradient(Y_one, Y_two,\
                        covariate_matrix, lambda, \
                        all_parameters, i, 1);
    }
    return gradient;
}

static Eigen::VectorXd calculate_standard_error(Eigen::VectorXd trait_one, Eigen::VectorXd trait_two, Eigen::MatrixXd covariate_matrix, \
                                                Eigen::VectorXd lambda, Eigen::VectorXd parameters){

        const double h2r_one = parameters(0)/(parameters(0) + parameters(2));
        const double h2r_two = parameters(1)/(parameters(1) + parameters(3));
        const double sd_one = sqrt(parameters(0) + parameters(2));
        const double sd_two = sqrt(parameters(1) + parameters(3));
        const double rhog = calculate_rho(parameters(4));
        const double rhoe = calculate_rho(parameters(5));///sqrt((final_parameters(2)-final_parameters(0))*(final_parameters(3)-final_parameters(1)));
        Eigen::VectorXd new_parameters = parameters;
        new_parameters(0) = h2r_one;
        new_parameters(1) = h2r_two;
        new_parameters(2) = sd_one;
        new_parameters(3) = sd_two;
        new_parameters(4) = rhog;
        new_parameters(5) = rhoe;
        Eigen::MatrixXd hessian(parameters.rows(), parameters.rows());
    for(int index = 0; index < parameters.rows() ; index++){

        hessian(index, index) = calculate_hessian(trait_one, trait_two,\
                        covariate_matrix, lambda, \
                        new_parameters, index, \
                         index, 0);

    }
  

    for(int i = 0 ; i < parameters.rows(); i++){

        for(int j = i  + 1; j < parameters.rows() ; j++){

         hessian(i, j) = hessian(j, i) = calculate_hessian(trait_one, trait_two,\
                        covariate_matrix, lambda, \
                        new_parameters, i, j, 0);
        }
       
    }

    Eigen::VectorXd standard_errors = (-hessian).inverse().diagonal().cwiseAbs().cwiseSqrt();

    return standard_errors;             

}

static void genetic_correlation_calculate_hessian_and_gradient(Eigen::MatrixXd & hessian, Eigen::VectorXd & gradient, Eigen::VectorXd Y_one, Eigen::VectorXd Y_two,\
                                                                Eigen::MatrixXd covariate_matrix,  Eigen::VectorXd lambda, Eigen::VectorXd all_parameters){
                                                                




  	for(int index = 0; index < all_parameters.rows() ; index++){

        gradient(index) = calculate_gradient(Y_one, Y_two,\
                        covariate_matrix, lambda, \
                        all_parameters, index);

        hessian(index, index) = calculate_hessian(Y_one, Y_two,\
                        covariate_matrix, lambda, \
                        all_parameters, index, \
                         index);

    }
  

    for(int i = 0 ; i < all_parameters.rows(); i++){

        for(int j = i  + 1; j < all_parameters.rows() ; j++){

         hessian(i, j) = hessian(j, i) = calculate_hessian(Y_one, Y_two,\
                        covariate_matrix, lambda, \
                        all_parameters, i, j);
        }
       
    }                                                                  
                                                                
} 
/*
static void genetic_correlation_calculate_hessian_and_gradient(Eigen::MatrixXd & hessian, Eigen::VectorXd & gradient, Eigen::VectorXd Y_one, Eigen::VectorXd Y_two,\
                                                                Eigen::MatrixXd covariate_matrix,  Eigen::VectorXd lambda, Eigen::VectorXd all_parameters){
                                                                



    const double loglik = calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
     lambda,  all_parameters); 
    for(int index = 0; index < all_parameters.rows() ; index++){
        double h = pow(DP, .66667)*max(abs(all_parameters(index)), 1.0);
        Eigen::VectorXd positive_parameters = all_parameters;
        positive_parameters(index) += h;
        Eigen::VectorXd negative_parameters = all_parameters;
        negative_parameters(index) -= h;
        const double positive_loglik = calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
        lambda,  positive_parameters);
        const double  negative_loglik = calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
        lambda,  negative_parameters);        
        gradient(index) = (positive_loglik - negative_loglik)/(2.0*h);

        hessian(index, index) = (positive_loglik -2.0*loglik + negative_loglik)/(h*h);

    }
  

    for(int i = 0 ; i < all_parameters.rows(); i++){
        double h_one = pow(DP, .66667)*max(abs(all_parameters(i)), 1.0);
    
        for(int j = i  + 1; j < all_parameters.rows() ; j++){
            double h_two = pow(DP, .66667)*max(abs(all_parameters(j)), 1.0);
            Eigen::VectorXd positive_parameters = all_parameters;
            double h = min(h_one, h_two);
            positive_parameters(i) += h;//h_one;
            positive_parameters(j) += h;//h_two;
            Eigen::VectorXd negative_parameters = all_parameters;
            negative_parameters(i) -= h;//h_one;    
            negative_parameters(j) -= h;//h_two;    
            const double positive_loglik = calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
            lambda,  positive_parameters);
            const double  negative_loglik = calculate_loglikelihood_param(Y_one,  Y_two,  covariate_matrix, \
            lambda,  negative_parameters);  
            hessian(i, j) = hessian(j, i) = (positive_loglik -2.0*loglik + negative_loglik)/(2.0*h*h - (hessian(i, i) + hessian(j, j))/2.0);
        }
       
    }                                                                  
                                                                
} */


static void calculate_mean_and_sd(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd & parameters, Eigen::VectorXd & beta){
    Eigen::VectorXd omega_one = lambda*calculate_constraint(parameters(0)) + (1.0 - calculate_constraint(parameters(0)))*Eigen::VectorXd::Ones(Y_one.rows());
    Eigen::VectorXd omega_two = lambda*calculate_constraint(parameters(1)) + (1.0 - calculate_constraint(parameters(1)))*Eigen::VectorXd::Ones(Y_one.rows());

    omega_one = omega_one.cwiseInverse();
    
    Eigen::VectorXd beta_one = (covariate_matrix.transpose()*omega_one.asDiagonal()*covariate_matrix).inverse()*covariate_matrix.transpose()*omega_one.asDiagonal()*Y_one;

    parameters(2) = sqrt((Y_one - covariate_matrix*beta_one).cwiseAbs2().dot(omega_one)/Y_one.rows());
    
    omega_two = omega_two.cwiseInverse();

    Eigen::VectorXd beta_two = (covariate_matrix.transpose()*omega_two.asDiagonal()*covariate_matrix).inverse()*covariate_matrix.transpose()*omega_two.asDiagonal()*Y_two;
 
  
    parameters(3) = sqrt((Y_two - covariate_matrix*beta_two).cwiseAbs2().dot(omega_two)/Y_two.rows());
    for(int i = 0; i < covariate_matrix.cols(); i++){
        beta(i) = beta_one(i);
        beta(i + covariate_matrix.cols()) = beta_two(i);
    }    

}
static Eigen::VectorXd estimate_theta(Eigen::VectorXd residual_squared, Eigen::MatrixXd aux){
    Eigen::VectorXd theta = (aux.transpose()*aux).inverse()*aux.transpose()*residual_squared;
    if(theta(0) < 0.0) theta(0) = 0.0;
    if(theta(1) < 0.0) theta(1) = 0.0;

    if(theta(0) == 0.0 && theta(1) == 0.0){
        theta(0) = residual_squared.sum()/aux.rows();
        return theta;
    }
    Eigen::VectorXd omega = (aux*theta).cwiseAbs2().cwiseInverse();
    Eigen::MatrixXd aux_omega = aux;
    aux_omega.col(0) = aux_omega.col(0).cwiseProduct(omega);
    aux_omega.col(1) = aux_omega.col(1).cwiseProduct(omega);

    theta = (aux_omega.transpose()*aux).inverse()*aux_omega.transpose()*residual_squared;
    if(theta(0) < 0.0) theta(0) = 0.0;
    if(theta(1) < 0.0) theta(1) = 0.0;
    if((theta(0) == 0.0 && theta(1) == 0.0) || theta(0) != theta(0) || theta(1) != theta(1)){
        theta(0) = residual_squared.dot(omega)/omega.sum();
        return theta;
    }
    return theta;    

}
static Eigen::VectorXd estimate_theta_with_omega(Eigen::VectorXd residual_squared, Eigen::MatrixXd aux, Eigen::MatrixXd omega){
    omega = omega.cwiseAbs2();
    Eigen::VectorXd theta = (aux.transpose()*omega*aux).inverse()*aux.transpose()*omega*residual_squared;
    if(theta(0) < 0.0) theta(0) = 0.0;
    if(theta(1) < 0.0) theta(1) = 0.0;

    if(theta(0) == 0.0 && theta(1) == 0.0){
        theta(0) = residual_squared.dot(omega.diagonal())/(omega.diagonal().sum());
        return theta;
    }
    /*
    Eigen::VectorXd omega = (aux*theta).cwiseAbs2().cwiseInverse();
    Eigen::MatrixXd aux_omega = aux;
    aux_omega.col(0) = aux_omega.col(0).cwiseProduct(omega);
    aux_omega.col(1) = aux_omega.col(1).cwiseProduct(omega);

    theta = (aux_omega.transpose()*aux).inverse()*aux_omega.transpose()*residual_squared;
    if(theta(0) < 0.0) theta(0) = 0.0;
    if(theta(1) < 0.0) theta(1) = 0.0;
    if((theta(0) == 0.0 && theta(1) == 0.0) || theta(0) != theta(0) || theta(1) != theta(1)){
        theta(0) = residual_squared.sum()/omega.rows();
        return theta;
    }*/
    return theta;    

}
/*
static void calculate_initial_parameters(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd & parameters, Eigen::VectorXd & beta){


    Eigen::MatrixXd aux = Eigen::MatrixXd::Ones(lambda.rows(), 2);
    aux.col(1) = lambda;

    
    Eigen::VectorXd beta_one = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_one;
    Eigen::VectorXd residual_one = Y_one - covariate_matrix*beta_one;
    Eigen::VectorXd theta_one = estimate_theta(residual_one.cwiseAbs2(), aux);
    
    parameters(2) = sqrt(theta_one.sum());

    parameters(0) = reverse_constraint(theta_one(1)/theta_one.sum());


    Eigen::VectorXd beta_two = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_two;
    Eigen::VectorXd residual_two = Y_two - covariate_matrix*beta_two;
    Eigen::VectorXd theta_two = estimate_theta(residual_two.cwiseAbs2(), aux);

    parameters(3) = sqrt(theta_two.sum());

    parameters(1) = reverse_constraint(theta_two(1)/theta_two.sum());
    /*
    double genetic_covar  = 0.0;
    for(int i = 0 ; i < lambda.rows();i++){
        double l = lambda(i);
        if(l != 0.0){
            genetic_covar += residual_one(i)*residual_two(i)/l;
        }
    }
    Eigen::VectorXd covar_residual = residual_one.cwiseProduct(residual_two);
    Eigen::VectorXd covar_theta = (aux.transpose()*aux).inverse()*aux.transpose()*covar_residual;
   // Eigen::MatrixXd omega = (aux*covar_theta).cwiseInverse().cwiseAbs2().asDiagonal();
    double rhog = covar_theta(1)/(parameters(2)*parameters(3)*sqrt(calculate_constraint(parameters(0))*calculate_constraint(parameters(1))));
    double rhoe = covar_theta(0)/(parameters(2)*parameters(3)*sqrt((1.0-calculate_constraint(parameters(0)))*(1.0-calculate_constraint(parameters(1)))));
    //double rhog = genetic_covar/(lambda.rows()*parameters(2)*parameters(3)*(calculate_constraint(parameters(0))*calculate_constraint(parameters(1))));
    cout << "rhog: " << rhog << endl;

    //double rhoe = residual_one.dot(residual_two)/(lambda.rows()*parameters(2)*parameters(3)*(1.0-calculate_constraint(parameters(0)))*(1.0-calculate_constraint(parameters(1))));
    cout << "rhoe: " << rhoe << endl;
    parameters(4) = reverse_rho(rhog);
    parameters(5) = reverse_rho(rhoe);
    for(int i = 0; i < covariate_matrix.cols(); i++){
        beta(i) = beta_one(i);
        beta(i + covariate_matrix.cols()) = beta_two(i);
    }    

}*/
static void calculate_parameters_fast(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd & parameters, Eigen::VectorXd & beta, Eigen::VectorXd & T_wald){


    Eigen::MatrixXd aux = Eigen::MatrixXd::Ones(lambda.rows(), 2);
    aux.col(1) = lambda;

    
    Eigen::VectorXd beta_one = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_one;
    Eigen::VectorXd residual_one = Y_one - covariate_matrix*beta_one;
    Eigen::VectorXd theta_one = estimate_theta(residual_one.cwiseAbs2(), aux);
    
    parameters(2) = sqrt(theta_one(0) + theta_one(1));

    parameters(0) = theta_one(1)/(theta_one(0) + theta_one(1));//reverse_constraint(theta_one(1)/theta_one.sum());


    Eigen::VectorXd beta_two = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_two;
    Eigen::VectorXd residual_two = Y_two - covariate_matrix*beta_two;
    Eigen::VectorXd theta_two = estimate_theta(residual_two.cwiseAbs2(), aux);

    parameters(3) = sqrt(theta_two(0) + theta_two(1));

    parameters(1) = theta_two(1)/(theta_two(0) + theta_two(1));

//reverse_constraint(theta_two(1)/theta_two.sum());
    /*
    double genetic_covar  = 0.0;
    for(int i = 0 ; i < lambda.rows();i++){
        double l = lambda(i);
        if(l != 0.0){
            genetic_covar += residual_one(i)*residual_two(i)/l;
        }
    }*/
   // if(constrain_parameter == 0 ){
        Eigen::VectorXd covar_residual = residual_one.cwiseProduct(residual_two);
        Eigen::VectorXd res_theta = theta_one.cwiseProduct(theta_two).cwiseSqrt();
        Eigen::MatrixXd covar_omega = (aux*res_theta).cwiseInverse().cwiseAbs2().asDiagonal();
        Eigen::MatrixXd aux_inverse = (aux.transpose()*covar_omega*aux).inverse();
        Eigen::VectorXd covar_theta = aux_inverse*aux.transpose()*covar_omega*covar_residual;


    double rhog = covar_theta(1)/sqrt(theta_one(1)*theta_two(1));
    
    if(rhog > 1.0){
        covar_theta(1) = sqrt(theta_one(1)*theta_two(1));
        rhog = 1.0;
    }else if (rhog < -1.0){
        covar_theta(1) = -sqrt(theta_one(1)*theta_two(1));
        rhog = -1.0;
    }

    
  
    double rhoe = covar_theta(0)/sqrt(theta_one(0)*theta_two(0));
    
    if(rhoe > 1.0){
        covar_theta(0) = sqrt(theta_one(0)*theta_two(0));
        rhoe = 1.0;
    }else if (rhoe < -1.0){
        covar_theta(0) = -sqrt(theta_one(0)*theta_two(0));
        rhoe = -1.0;
    }
    
    
    T_wald(0) = pow(covar_theta(0), 2)/aux_inverse(0, 0);
    T_wald(1) = pow(covar_theta(1), 2)/aux_inverse(1, 1);

 /*

        cout << "rhog : " << rhog << endl;
        cout << "rhoe : "  << rhoe << endl;
        cout << "rhop : " << (covar_theta(0) + covar_theta(1))/sqrt((theta_one(0) + theta_one(1))*(theta_two(0)+theta_two(1))) << endl;
        cout << "covar/var: " << covar_theta(0)/(covar_theta(0) + covar_theta(1)) << " " << covar_theta(1)/(covar_theta(0) + covar_theta(1)) << endl;
        cout << endl;

        Eigen::VectorXd covar_omega_one_two = (aux*covar_theta)*pow((theta_one(0) + theta_one(1))*(theta_two(0) + theta_two(1)), -0.5);
        Eigen::VectorXd omega_one_one = aux*theta_one;
        Eigen::VectorXd omega_two_two = aux*theta_two;

        Eigen::VectorXd omega_det = (omega_one_one.cwiseProduct(omega_two_two) - covar_omega_one_two.cwiseAbs2()).cwiseInverse();
        Eigen::MatrixXd omega = (omega_two_two.cwiseProduct(omega_det)).asDiagonal();
        beta_one = (covariate_matrix.transpose()*omega*covariate_matrix).inverse()*covariate_matrix.transpose()*omega*Y_one;
        residual_one = Y_one - covariate_matrix*beta_one;
        theta_one = estimate_theta_with_omega(residual_one.cwiseAbs2(), aux, omega);//(aux.transpose()*omega.cwiseAbs2()*aux).inverse()*aux.transpose()*omega.cwiseAbs2()*residual_one.cwiseAbs2();
        
        omega = (omega_one_one.cwiseProduct(omega_det)).asDiagonal();
        beta_two =  (covariate_matrix.transpose()*omega*covariate_matrix).inverse()*covariate_matrix.transpose()*omega*Y_two;
        residual_two = Y_two - covariate_matrix*beta_two;
        theta_two = estimate_theta_with_omega(residual_two.cwiseAbs2(), aux, omega);

        covar_residual = residual_one.cwiseProduct(residual_two);
        omega = (-covar_omega_one_two).cwiseProduct(omega_det).cwiseAbs2().asDiagonal();
        covar_theta = (aux.transpose()*omega*aux).inverse()*aux.transpose()*omega*covar_residual;

        if(constrain_parameter == 1){
            Eigen::VectorXd omega_vector = omega.diagonal();
            covar_theta(0) = covar_residual.dot(omega_vector)/omega_vector.sum();
            covar_theta(1) = 0.0;
        }else if (constrain_parameter == 2){
            Eigen::VectorXd omega_vector = omega.diagonal();
            covar_theta(0) = 0.0;
            covar_theta(1) = covar_residual.dot(omega_vector.cwiseProduct(aux.col(1)))/aux.col(1).cwiseAbs2().dot(omega_vector);    
        }

        cout << "covar/var: " << covar_theta(0)/(covar_theta(0) + covar_theta(1)) << " " << covar_theta(1)/(covar_theta(0) + covar_theta(1)) << endl;
   // Eigen::MatrixXd omega = (aux*covar_theta).cwiseInverse().cwiseAbs2().asDiagonal();
    
     rhog = covar_theta(1)/sqrt(theta_one(1)*theta_two(1));
    
    if(rhog > 1.0){
        covar_theta(1) = sqrt(theta_one(1)*theta_two(1));
        rhog = 1.0;
    }else if (rhog < -1.0){
        covar_theta(1) = -sqrt(theta_one(1)*theta_two(1));
        rhog = -1.0;
    }

    
  
     rhoe = covar_theta(0)/sqrt(theta_one(0)*theta_two(0));
    
    if(rhoe > 1.0){
        covar_theta(0) = sqrt(theta_one(0)*theta_two(0));
        rhoe = 1.0;
    }else if (rhoe < -1.0){
        covar_theta(0) = -sqrt(theta_one(0)*theta_two(0));
        rhoe = -1.0;
    }
        cout << "rhog : " << rhog << endl;
        cout << "rhoe : "  << rhoe << endl;
        cout << "rhop : " << (covar_theta(0) + covar_theta(1))/sqrt((theta_one(0) + theta_one(1))*(theta_two(0)+theta_two(1))) << endl;
        cout << "covar/var: " << covar_theta(0)/(covar_theta(0) + covar_theta(1)) << " " << covar_theta(1)/(covar_theta(0) + covar_theta(1)) << endl << endl;*/
    parameters(4) =  rhog;
    parameters(5) =  rhoe;
   /* }else if (constrain_parameter == 1){
        Eigen::VectorXd covar_residual = residual_one.cwiseProduct(residual_two);
        double covar_theta = covar_residual.sum()/covar_residual.rows();

   // Eigen::MatrixXd omega = (aux*covar_theta).cwiseInverse().cwiseAbs2().asDiagonal();
    
        double rhog = 0.0;
    


    
  
        double rhoe = covar_theta/sqrt(theta_one(0)*theta_two(0));
    
        if(rhoe > 1.0){
            //covar_theta(0) = sqrt(theta_one(0)*theta_two(0));
            rhoe = 1.0;
        }else if (rhoe < -1.0){
            //covar_theta(0) = -sqrt(theta_one(0)*theta_two(0));
            rhoe = -1.0;
        }

        parameters(4) = rhog;
        parameters(5) = rhoe;        
    }else if (constrain_parameter == 2){
        Eigen::VectorXd covar_residual = residual_one.cwiseProduct(residual_two);
        double covar_theta = covar_residual.dot(aux.col(1))/aux.col(1).squaredNorm();

   // Eigen::MatrixXd omega = (aux*covar_theta).cwiseInverse().cwiseAbs2().asDiagonal();
    
        double rhog = covar_theta/sqrt(theta_one(1)*theta_two(1));
    
        if(rhog > 1.0){
         //   covar_theta(1) = sqrt(theta_one(1)*theta_two(1));
            rhog = 1.0;
        }else if (rhog < -1.0){
         //   covar_theta(1) = -sqrt(theta_one(1)*theta_two(1));
            rhog = -1.0;
        }


    
  
        double rhoe = 0.0;//covar_theta/sqrt(theta_one(0)*theta_two(0));
        
        if(rhoe > 1.0){
            //covar_theta(0) = sqrt(theta_one(0)*theta_two(0));
            rhoe = 1.0;
        }else if (rhoe < -1.0){
            //covar_theta(0) = -sqrt(theta_one(0)*theta_two(0));
            rhoe = -1.0;
        }

        parameters(4) = rhog;
        parameters(5) = rhoe;        
    }*/
   
    for(int i = 0; i < covariate_matrix.cols(); i++){
        beta(i) = beta_one(i);
        beta(i + covariate_matrix.cols()) = beta_two(i);
    }   


}


static void calculate_initial_parameters(Eigen::VectorXd Y_one, Eigen::VectorXd Y_two, Eigen::MatrixXd covariate_matrix, Eigen::VectorXd lambda, Eigen::VectorXd & parameters, Eigen::VectorXd & beta, int constrain_parameter = 0){


    Eigen::MatrixXd aux = Eigen::MatrixXd::Ones(lambda.rows(), 2);
    aux.col(1) = lambda;

    
    Eigen::VectorXd beta_one = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_one;
    Eigen::VectorXd residual_one = Y_one - covariate_matrix*beta_one;
    Eigen::VectorXd theta_one = estimate_theta(residual_one.cwiseAbs2(), aux);
    
    parameters(2) = theta_one(0);

    parameters(0) = theta_one(1);//reverse_constraint(theta_one(1)/theta_one.sum());


    Eigen::VectorXd beta_two = (covariate_matrix.transpose()*covariate_matrix).inverse()*covariate_matrix.transpose()*Y_two;
    Eigen::VectorXd residual_two = Y_two - covariate_matrix*beta_two;
    Eigen::VectorXd theta_two = estimate_theta(residual_two.cwiseAbs2(), aux);

    parameters(3) = (theta_two(0));//sqrt(theta_two.sum());

    parameters(1) = theta_two(1);//reverse_constraint(theta_two(1)/theta_two.sum());
    /*
    double genetic_covar  = 0.0;
    for(int i = 0 ; i < lambda.rows();i++){
        double l = lambda(i);
        if(l != 0.0){
            genetic_covar += residual_one(i)*residual_two(i)/l;
        }
    }*/
    Eigen::VectorXd covar_residual = residual_one.cwiseProduct(residual_two);
    Eigen::VectorXd covar_theta(2);//
    double rhog, rhoe;
    if(constrain_parameter == 0){
       // Eigen::MatrixXd covar_omega = Eigen::MatrixXd::Identity(aux.rows(), aux.rows());//(aux*(theta_one.cwiseProduct(theta_two).cwiseSqrt())).cwiseInverse().cwiseAbs2().asDiagonal();
        covar_theta = (aux.transpose()*aux).inverse()*aux.transpose()*covar_residual;
        rhoe = covar_theta(0)/sqrt((parameters(2))*(parameters(3)));
        rhog = covar_theta(1)/sqrt((parameters(0)*parameters(1)));
        
    if(rhog > 1.0){
        covar_theta(1) = sqrt(parameters(0)*parameters(1));
        rhog = 0.98;
    }else if (rhog < -1.0){
        covar_theta(1) = -sqrt(parameters(0)*parameters(1));
        rhog = -0.98;
    } 
    if(rhoe > 1.0){
        covar_theta(0) = sqrt((parameters(2))*(parameters(3)));
        rhoe = 0.98;
    }else if (rhoe < -1.0){
        covar_theta(0) = -sqrt((parameters(2))*(parameters(3)));
        rhoe = -0.98;
    }
    rhog = (constrain_parameter == 1) ? 0.0 : rhog;
    rhoe = (constrain_parameter == 2) ? 0.0 : rhoe;
    parameters(4) = reverse_rho(rhog);
    parameters(5) = reverse_rho(rhoe);                       
    }else if (constrain_parameter == 1){
        //Eigen::VectorXd covar_omega = (aux*(theta_one.cwiseProduct(theta_two).cwiseSqrt())).cwiseInverse().cwiseAbs2();
        covar_theta(0) = covar_residual.sum()/covar_residual.rows();//covar_omega.dot(covar_residual)/covar_omega.sum();
        covar_theta(1) = 0.0;
        rhog = 0.0;
        rhoe = covar_theta(0)/sqrt(parameters(2)*parameters(3));
        if(rhoe > 1.0){
            rhoe = 0.98;
            covar_theta(0) = sqrt(parameters(2)*parameters(3));
        }else if(rhoe < -1.0){
            rhoe = -0.98;
            covar_theta(0) = -sqrt(parameters(2)*parameters(3));
        }
        parameters(4) = reverse_rho(rhog);
        parameters(5) = reverse_rho(rhoe);         
    }else if (constrain_parameter == 2){
        //igen::VectorXd covar_omega = (aux*(theta_one.cwiseProduct(theta_two).cwiseSqrt())).cwiseInverse().cwiseAbs2();
        covar_theta(1) = covar_residual.dot(lambda)/lambda.squaredNorm();//covar_residual.dot(covar_omega.cwiseProduct(lambda))/lambda.cwiseAbs2().dot(covar_omega);//covar_residual.dot(lambda/lambda.squaredNorm());
        covar_theta(0) = 0.0;
        rhoe = 0.0;
        rhog = covar_theta(1)/sqrt(parameters(0)*parameters(1));
        if(rhog > 1.0){
            rhog = 0.98;
            covar_theta(1) = sqrt(parameters(0)*parameters(1));
        }else if(rhog < -1.0){
            rhog = -0.98;
            covar_theta(1) = -sqrt(parameters(0)*parameters(1));
        }
        parameters(4) = reverse_rho(rhog);
        parameters(5) = reverse_rho(rhoe);           
    }

   // Eigen::MatrixXd omega = (aux*covar_theta).cwiseInverse().cwiseAbs2().asDiagonal();
    
    


    if(parameters(0) == 0.0 && parameters(2) > 0.0){
        parameters(0) = 0.0001*parameters(2);
        parameters(2) -= parameters(0);
    }
    if(parameters(1) == 0.0 && parameters(3) > 0.0){
        parameters(1) = 0.0001*parameters(3);
        parameters(3) -= parameters(1);
    }   
    if(parameters(2) == 0.0 && parameters(0) > 0.0){
        parameters(2) = 0.0001*parameters(0);
        parameters(0) -= parameters(2);
    }
    if(parameters(3) == 0.0 && parameters(1) > 0.0){
        parameters(3) = 0.0001*parameters(1);
        parameters(1) -= parameters(3);
    }      
    for(int i = 0; i < covariate_matrix.cols(); i++){
        beta(i) = beta_one(i);
        beta(i + covariate_matrix.cols()) = beta_two(i);
    }   
    if(theta_one(0) == 0.0){
        theta_one(0) = 0.0001;
        parameters(2) = theta_one(0);
    }
    if(theta_one(1) == 0.0){
        theta_one(1) = 0.0001;
        parameters(0) = theta_one(1);
    }    
    if(theta_two(0) == 0.0){
        theta_two(0) = 0.0001;
        parameters(3) = (theta_two(0));
    }
    if(theta_two(1) == 0.0){
        theta_two(1) = 0.0001;
        parameters(1) = (theta_two(1));
    } 
     
}

static double calculate_bivariate_model_fast(Eigen::VectorXd trait_one, Eigen::VectorXd trait_two, Eigen::MatrixXd covariate_matrix,\
                                        Eigen::VectorXd eigenvalues, Eigen::VectorXd & final_parameters, Eigen::VectorXd & T_wald){
    Eigen::VectorXd beta(covariate_matrix.cols()*2);
    Eigen::VectorXd parameters(6);
    calculate_parameters_fast(trait_one,trait_two, covariate_matrix, eigenvalues, parameters, beta, T_wald);
    final_parameters = combine_parameters_and_beta(parameters, beta);
    display_totals = 1;
    double loglik = calculate_loglikelihood_param(trait_one, trait_two, covariate_matrix, eigenvalues, final_parameters, 0);
    display_totals = 0;
    return loglik;
}

static double calculate_bivariate_model(Eigen::VectorXd trait_one, Eigen::VectorXd trait_two, Eigen::MatrixXd covariate_matrix, \
    Eigen::VectorXd eigenvalues, Eigen::VectorXd & final_parameters, Eigen::VectorXd & final_beta, Eigen::VectorXd & standard_errors, \
    int & iteration_count, int constrain_parameter = 0, bool debug = false){

    double best_loglikelihood;
    bool first_loglik = false;
    double h;
    Eigen::VectorXd beta(covariate_matrix.cols()*2);
    Eigen::VectorXd parameters(6);
    Eigen::VectorXd all_parameters(6 + beta.rows());
    /*Eigen::VectorXd coded_parameters(8);
    double var_one = 446.7052672872056*446.7052672872056;
    double var_two = 426.1420555072798*426.1420555072798;
    coded_parameters(0) = var_one*0.141505388791664;
    coded_parameters(1) = var_two*0.1436917572905575;
    coded_parameters(2) = var_one;
    coded_parameters(3) = var_two;
    coded_parameters(4) = sqrt(coded_parameters(0)*coded_parameters(1));
    coded_parameters(5) = 0.7165886195098408*sqrt((var_one - coded_parameters(0))*(var_two-coded_parameters(1)));
    coded_parameters(6) = 1829.29166983507;
    coded_parameters(7) = 1772.834934368391;
    const double coded_loglik = calculate_loglikelihood_param(trait_one, trait_two, covariate_matrix,eigenvalues, coded_parameters);
    Eigen::VectorXd coded_gradient(8);
    Eigen::MatrixXd coded_hessian(8,8);
    genetic_correlation_calculate_hessian_and_gradient(coded_hessian,  coded_gradient, trait_one, trait_two,\
                                                               covariate_matrix,  eigenvalues,  coded_parameters);    
    cout.precision(11);
    cout << "coded loglik: " << coded_loglik << endl;
    cout << "gradient: " << coded_gradient << endl;*/
//    parameters(0) = 0.0;
//    parameters(1) = 0.0;
  //  parameters(4) = 0.0;
//    parameters(5) = 0.0;

    calculate_initial_parameters(trait_one, trait_two, covariate_matrix, eigenvalues, parameters ,beta, constrain_parameter);

    all_parameters = combine_parameters_and_beta(parameters, beta);
    double loglik = calculate_loglikelihood_param(trait_one, trait_two, covariate_matrix, eigenvalues, all_parameters);

    Eigen::VectorXd delta = Eigen::VectorXd::Zero(6 + covariate_matrix.cols()*2);
    Eigen::MatrixXd hessian(6 + covariate_matrix.cols()*2, 6 + covariate_matrix.cols()*2);
    Eigen::VectorXd gradient(6 + covariate_matrix.cols()*2);
    int iteration = 0;
    double last_loglik = 0.0;
    double loglik_error;
    double max_delta;
    genetic_correlation_calculate_hessian_and_gradient(hessian,  gradient, trait_one, trait_two,\
                                                               covariate_matrix,  eigenvalues,  all_parameters);


    if(constrain_parameter == 1){
        gradient(4) = 0.0;
        for(int i = 0 ;i < hessian.rows(); i++){
            hessian(i, 4) = hessian(4, i) = 0.0;
        }
        hessian(4, 4) = 1.0;
    }
    if(constrain_parameter == 2){
        gradient(5) = 0.0;
        for(int i = 0 ;i < hessian.rows(); i++){
            hessian(i, 5) = hessian(5, i) = 0.0;
        }
        hessian(5, 5) = 1.0;
    }
    /*for(int i = 0; i < hessian.rows();i++){
        if(hessian(i, i) == 0.0) hessian(i, i) = 1.0;
    }*/
    double error;
    int converge_count = 0;
    int step_max = 80;
    double step_size = 1.0/step_max;
    double lambda = 0.125;  
    double D;
    /*
    if(debug ){
        cout << "Iteration: " << iteration << endl;
        cout << "h2r-one: " << calculate_constraint(all_parameters(0)) << " h2r-two: " << calculate_constraint(all_parameters(1)) << endl;
        cout << "sd-one: " << abs(all_parameters(2)) << " sd-two: " << abs(all_parameters(3)) << endl;
        cout << "rhog: " << calculate_rho(all_parameters(4)) << " rhoe: " << calculate_rho(all_parameters(5)) << endl;
        cout << "loglik: " << loglik << endl;
        cout << "step: " << 0.0 << endl;
        cout << "lambda: " << 0.0 << endl;  
    }*/
    if(debug ){
        cout << "Iteration: " << iteration << endl;
        cout << "h2r-one: " << all_parameters(0)/(all_parameters(0)+all_parameters(2)) << " h2r-two: " << all_parameters(1)/(all_parameters(1)+all_parameters(3))<< endl;
        cout << "sd-one: " << sqrt(all_parameters(0)+(all_parameters(2))) << " sd-two: " << sqrt(all_parameters(1) + (all_parameters(3))) << endl;
        cout << "rhog: " << calculate_rho(all_parameters(4)) << " rhoe: " << calculate_rho(all_parameters(5)) << endl;
        cout << "loglik: " << loglik << endl;
        cout << "step: " << 0.0 << endl;
        cout << "lambda: " << 0.0 << endl;  
        cout << "grad*delta: " << 0.0 << endl;
    }
    do{ 

        
        Eigen::VectorXd test_parameters;
        Eigen::VectorXd best_delta;
        double best_loglik = NAN;
        Eigen::VectorXd best_parameters;
      //  Eigen::MatrixXd hessian_diagonal = hessian.diagonal().asDiagonal();
        /*if(constrain_parameter == 1){
            hessian_diagonal(4, 4) = 0.0;
        }
        if(constrain_parameter == 2){
            hessian_diagonal(5, 5) = 0.0;
        }*/
        int loop_count = 0;
        lambda = 1.0;//0.125;
        double best_l;
        //do{
       // for(int step = 1; step <= step_max; step++){

            Eigen::MatrixXd new_hessian = hessian;// +lambda*hessian_diagonal;
            new_hessian = new_hessian.inverse();
           // cout << "hessian: " << hessian << endl;
            //cout << "hessian inverse: " << new_hessian << endl;
           // cout << "gradient: " << gradient << endl;
            delta = -new_hessian*gradient;
            D = delta.dot(gradient);
            double t_one = 10E-6*D;
 
            double t_two = abs(0.6*D);
           // cout << "t: " << t << endl;
          //  cout << "D: " << D << endl;
            double alpha = 2.0;
            double new_loglik = NAN;
            Eigen::VectorXd new_gradient;
            do{
                
                if(loop_count != 0) alpha = alpha*0.5;
                //t = -D*alpha + 0.5*D*alpha*alpha; 
            //for(double l = step; l <= 1.0; l += step){
                Eigen::VectorXd test_delta = alpha*delta;
                test_parameters = all_parameters + test_delta;
                if(test_parameters(0) < 0.0) test_parameters(0) = abs(test_parameters(0));
                if(test_parameters(1) < 0.0) test_parameters(1) = abs(test_parameters(1));

                if(test_parameters(2) < 0.0) test_parameters(2) = abs(test_parameters(2));
                if(test_parameters(3) < 0.0) test_parameters(3) = abs(test_parameters(3));
                /*
                double rhog = test_parameters(4)/sqrt(test_parameters(0)*test_parameters(1));
                if (constrain_parameter != 1){
                    if(rhog > 1.0){
                        test_parameters(4) = sqrt(test_parameters(0)*test_parameters(1));
                    }else if (rhog < -1.0){
                        test_parameters(4) = -sqrt(test_parameters(0)*test_parameters(1));
                    }

                }
                double rhoe = test_parameters(5)/sqrt((test_parameters(2)-test_parameters(0))*(test_parameters(3)-test_parameters(1)));
                if (constrain_parameter != 2){
                    if(rhoe > 1.0){
                        test_parameters(5) = sqrt((test_parameters(2)-test_parameters(0))*(test_parameters(3)-test_parameters(1)));
                        rhoe = 1.0;
                    }else if (rhoe < -1.0){
                        test_parameters(5) = -sqrt((test_parameters(2)-test_parameters(0))*(test_parameters(3)-test_parameters(1)));
                        rhoe = -1.0;
                    }

                }  */              
  
                /*
                double rhoe = test_parameters(5)/sqrt(test_parameters(2)*test_parameters(3));
        
                if (constrain_parameter != 2){
                    if(rhoe > 1.0){
                        test_parameters(5) = sqrt(test_parameters(2)*test_parameters(3));
                    }else if (rhoe < -1.0){
                        test_parameters(5) = -sqrt(test_parameters(2)*test_parameters(3));
                    }

                } */                   
                new_loglik = calculate_loglikelihood_param(trait_one, trait_two, covariate_matrix, eigenvalues, test_parameters);
                new_gradient = calculate_gradient_vector(trait_one, trait_two,covariate_matrix, eigenvalues, test_parameters);
                if(constrain_parameter == 1) new_gradient(4) =  0.0;
                if(constrain_parameter == 2) new_gradient(5) =  0.0;    
                /*if(test_loglik >= best_loglik  || (best_loglik != best_loglik && test_loglik == test_loglik)){
                    best_loglik = test_loglik;
                    best_parameters = test_parameters;
                    best_l = step*l;
                    best_delta = test_delta;
                }  */ 


            }while(new_loglik == new_loglik && (((new_loglik - loglik) < alpha*t_one) ) && ++loop_count < 50);

       // }
         /*  if(best_loglik == best_loglik && best_loglik >= loglik){
                
     
                break;
            }else{
                lambda *= 0.5;

                best_loglik = NAN;

            }
            loop_count++;
        }while(loop_count != 4);*/
        if((new_loglik != new_loglik || loop_count == 50) && iteration > 0 && loglik == loglik){
            converge_count++;
            break;
        }
        if(new_loglik != new_loglik || loop_count == 50){
            final_parameters = Eigen::VectorXd::Zero(6 + covariate_matrix.cols()*2);
            beta = Eigen::VectorXd::Zero(covariate_matrix.cols()*2);
            standard_errors = Eigen::VectorXd::Zero(6+covariate_matrix.cols()*2);
            return nan("");            
        }
        /*
        cout << (new_loglik - loglik)/(alpha*gradient.dot(delta) + 0.5*alpha*delta.dot(hessian*(alpha*delta))) << endl;
        cout << (new_loglik - loglik)/(gradient.dot(delta) -  0.5*delta.dot(hessian*(delta))) << endl;
         cout << (new_loglik - loglik)/(alpha*gradient.dot(delta) -  alpha*0.5*delta.dot(hessian*(alpha*delta))) << endl;
        cout << (new_loglik - loglik)/(-gradient.dot(delta) +  0.5*delta.dot(hessian*(delta))) << endl;
         cout << (new_loglik - loglik)/(-alpha*gradient.dot(delta) +  alpha*delta.dot(hessian*(alpha*delta))) << endl;      */   
        last_loglik = loglik;
        loglik = new_loglik;
        loglik_error = fabs((last_loglik - loglik)/last_loglik);
        all_parameters = test_parameters;
        gradient = new_gradient;
        //delta = best_delta;
        D = alpha*gradient.dot(delta);

        if (gradient.norm()  < 10E-6 || loglik_error == 0.0){
            converge_count++;
        }else{
            converge_count = 0;
        } 
    
       
        /*
        if(debug ){
            cout << "Iteration: " << iteration << endl;
            cout << "h2r-one: " << calculate_constraint(all_parameters(0)) << " h2r-two: " << calculate_constraint(all_parameters(1)) << endl;
            cout << "sd-one: " << abs(all_parameters(2)) << " sd-two: " << abs(all_parameters(3)) << endl;
            cout << "rhog: " << calculate_rho(all_parameters(4)) << " rhoe: " << calculate_rho(all_parameters(5)) << endl;
            cout << "loglik: " << loglik << endl;
            cout << "step: " << best_l << endl;
            cout << "lambda: " << lambda << endl;   

        }*/
     
        genetic_correlation_calculate_hessian_and_gradient(hessian,  gradient, trait_one, trait_two,\
                                                               covariate_matrix,  eigenvalues,  all_parameters);

        //Eigen::VectorXd new_gradient = calculate_gradient_vector(trait_one, trait_two,covariate_matrix, eigenvalues, all_parameters);
/*        if(constrain_parameter == 1) new_gradient(4) = 0.0;
        if(constrain_parameter == 2) new_gradient(5) = 0.0;        
        Eigen::VectorXd y = new_gradient - gradient;

        delta = delta*alpha;
        hessian = hessian + y*y.transpose()/(delta.dot(y)) - hessian*delta*delta.transpose()*hessian/(delta.dot(hessian*delta));
        gradient = new_gradient; */
        iteration++;                                                            
        if(debug ){
            cout << "Iteration: " << iteration << endl;
            cout << "h2r-one: " << all_parameters(0)/(all_parameters(2)+all_parameters(0)) << " h2r-two: " << all_parameters(1)/(all_parameters(1) + all_parameters(3))<< endl;
            cout << "sd-one: " << sqrt(abs(all_parameters(0)+all_parameters(2))) << " sd-two: " << sqrt(abs(all_parameters(1)+all_parameters(3))) << endl;
            cout << "rhog: " << calculate_rho(all_parameters(4))<< " rhoe: " << calculate_rho(all_parameters(5)) << endl;
            cout << "loglik: " << loglik << endl;
            cout << "alpha: " << alpha << endl;
            //cout << "lambda: " << lambda << endl;
            cout << "grad*delta: " << D << endl;           
           // cout << "lambda: " << 0.0 << endl;  
        }           
        if(constrain_parameter == 1){
            gradient(4) = 0.0;
            for(int i = 0 ;i < hessian.rows(); i++){
                hessian(i, 4) = hessian(4, i) = 0.0;
            }
            hessian(4, 4) = 1.0;
        }
        if(constrain_parameter == 2){
            gradient(5) = 0.0;
            for(int i = 0 ;i < hessian.rows(); i++){
                hessian(i, 5) = hessian(5, i) = 0.0;
            }
            hessian(5, 5) = 1.0;
        }
        /*
        for(int i = 0; i < hessian.rows();i++){
            if(hessian(i, i) == 0.0) hessian(i, i) = 1.0;
        }  */      

    }while(loglik == loglik  && iteration < MAX_ITERATIONS  && (converge_count  < 1));  

    if(loglik == loglik && converge_count >= 1){

    //genetic_correlation_calculate_hessian_and_gradient(hessian, gradient,  trait_one, trait_two,\
                                        covariate_matrix,  eigenvalues,  all_parameters);  
        best_loglikelihood = loglik;
        //all_parameters(2) = abs(all_parameters(2));
        //all_parameters(3) = abs(all_parameters(3));        
        for(int i = 0 ; i < 6 ; i++){
            parameters(i) = all_parameters(i);
        }
        for(int i = 0 ; i < beta.rows(); i++){
            beta(i) = all_parameters(6 + i);
        }
        final_parameters = parameters;
        final_beta = beta;
        first_loglik = true;
        iteration_count = iteration;
        /*
        Eigen::MatrixXd var_one_hessian(2, 2);
        Eigen::MatrixXd var_two_hessian(2, 2);

        var_one_hessian(0, 0) = hessian(0, 0 );
        var_one_hessian(0, 1) = var_one_hessian(1, 0) = hessian(0, 2);
        var_one_hessian(1, 1) = hessian(2, 2);


        var_two_hessian(0, 0) = hessian(1, 1);
        var_two_hessian(0, 1) = var_two_hessian(1, 0) = hessian(1, 3);
        var_two_hessian(1, 1) = hessian(3, 3);

        Eigen::VectorXd var_one_standard_errors = (-var_one_hessian).inverse().diagonal().cwiseAbs().cwiseSqrt();

        Eigen::VectorXd var_two_standard_errors = (-var_two_hessian).inverse().diagonal().cwiseAbs().cwiseSqrt();*/
/*
        Eigen::MatrixXd var_hessian(4, 4);


        var_hessian(0, 0) = hessian(0, 0 );
        var_hessian(0, 1) = var_hessian(1, 0) = hessian(0, 1);
        var_hessian(0, 2) = var_hessian(2, 0) = hessian(2, 0);
        var_hessian(0, 3) = var_hessian(3, 0) = hessian(3, 0);

        var_hessian(1, 1) = hessian(1, 1);
        var_hessian(1, 2) = var_hessian(2, 1) = hessian(1, 2);
        var_hessian(1, 3) = var_hessian(3, 1) = hessian(1, 3);

        var_hessian(2, 2) = hessian(2, 2);
        var_hessian(2, 3) = var_hessian(3, 2) = hessian(2, 3);

        var_hessian(3, 3) = hessian(3, 3);*/

        standard_errors =calculate_standard_error(trait_one, trait_two,covariate_matrix, \
                                                eigenvalues, all_parameters);// (-hessian).inverse().diagonal().cwiseAbs().cwiseSqrt();
        /*const double h2r_one_se = h2r_error(final_parameters(0), final_parameters(2), standard_errors(0), standard_errors(2));
        const double h2r_two_se = h2r_error(final_parameters(1), final_parameters(3), standard_errors(1), standard_errors(3));
        const double sd_one_se = sd_error(final_parameters(0), final_parameters(2), standard_errors(0), standard_errors(2));
        const double sd_two_se = sd_error(final_parameters(1), final_parameters(3), standard_errors(1), standard_errors(3));
        const double rhog_se = rho_error(final_parameters(4), standard_errors(4));
        const double rhoe_se = rho_error(final_parameters(5), standard_errors(5));

        standard_errors(0) = h2r_one_se;
        standard_errors(1) = h2r_two_se;
        standard_errors(2) = sd_one_se;
        standard_errors(3) = sd_two_se;
        standard_errors(4) = rhog_se;
        standard_errors(5) = rhoe_se;*/

        display_totals = 1;
        calculate_loglikelihood_param(trait_one, trait_two, covariate_matrix, eigenvalues, all_parameters);
        display_totals = 0;
        const double h2r_one = final_parameters(0)/(final_parameters(0) + final_parameters(2));
        const double h2r_two = final_parameters(1)/(final_parameters(1) + final_parameters(3));
        const double sd_one = sqrt(final_parameters(0) + final_parameters(2));
        const double sd_two = sqrt(final_parameters(1) + final_parameters(3));
        const double rhog = calculate_rho(final_parameters(4));
        const double rhoe = calculate_rho(final_parameters(5));///sqrt((final_parameters(2)-final_parameters(0))*(final_parameters(3)-final_parameters(1)));

        final_parameters(0) = h2r_one;
        final_parameters(1) = h2r_two;
        final_parameters(2) = sd_one;
        final_parameters(3) = sd_two;
        final_parameters(4) = rhog;
        final_parameters(5) = rhoe;

        /*
        standard_errors(0) =fabs(standard_errors(0)*calculate_dconstraint(all_parameters(0)));
        standard_errors(1) =fabs(standard_errors(1)*calculate_dconstraint(all_parameters(1)));
        standard_errors(4) =(constrain_parameter == 1) ? 0.0 : fabs(standard_errors(4)*calculate_rho_dconstraint(all_parameters(4)));
        standard_errors(5) =(constrain_parameter == 2) ? 0.0 : fabs(standard_errors(5)*calculate_rho_dconstraint(all_parameters(5))); 
        */         
      
    }
    
  
    if(first_loglik == false){
        final_parameters = Eigen::VectorXd::Zero(6 + covariate_matrix.cols()*2);
        beta = Eigen::VectorXd::Zero(covariate_matrix.cols()*2);
        standard_errors = Eigen::VectorXd::Zero(6+covariate_matrix.cols()*2);
        return nan("");
    }

  
    return best_loglikelihood;
}                                                             
 
static const char * calculate_genetic_correlation(Tcl_Interp * interp, vector<string> trait_list, const char* phenotype_filename, bool display_pvalues, bool debug, bool use_fast_version, const char * evd_data_filename = 0){
    vector<string> cov_list;
    vector<string> unique_cov_terms;
    int success;
    Covariate * c;
    
    int n_covariates = 0;
    for (int i = 0;( c = Covariate::index(i)); i++)
    {
        char buff[512];
        c->fullname(&buff[0]);
        cov_list.push_back(string(&buff[0]));
        CovariateTerm * cov_term;
        
        for(cov_term = c->terms(); cov_term; cov_term = cov_term->next){
            bool found = false;
            
            for(vector<string>::iterator cov_iter = unique_cov_terms.begin(); cov_iter != unique_cov_terms.end(); cov_iter++){
                if(!StringCmp(cov_term->name, cov_iter->c_str(), case_ins)){
                    found = true;
                    break;
                }
            }
            if(!found){
                unique_cov_terms.push_back(string(cov_term->name));
            }
        }
        n_covariates++;
    }
    vector<string> field_list;
    field_list.push_back(trait_list[0]);
    field_list.push_back(trait_list[1]);
    for(int i = 0; i < unique_cov_terms.size(); i++){
        field_list.push_back(unique_cov_terms[i]);
    }    
    solar_mle_setup * file_data;
    try{
        file_data = new solar_mle_setup(field_list, phenotype_filename,interp,  true); 
    }catch(Parse_Expression_Error & error){
    	
      
        return error.what().c_str();

    }catch(Solar_File_Error & error){
        return error.what().c_str();
 
    }catch(Expression_Eval_Error & error){
        return error.what().c_str();
      
    }catch(Misc_Error & error){
        
        
        return error.what().c_str();
    }catch(Syntax_Error &e){
    
    	return "Syntax Error occurred in expression";

    }catch(Undefined_Function &e){
    	return "Undefined Function Error occurred in expression";

    }catch(Unresolved_Name &e){
    	return "Unresolved Name Error occurred in expression";
 
    }catch(Undefined_Name &e){
        cout << e.name << endl;
    	return "Undefined Name Error occurred in expression";
    }catch(...){
    	
    	return "Unkown error occurred reading phenotype data";
    }
    Eigen::MatrixXd field_matrix = file_data->return_output_matrix();
    Eigen::VectorXd trait_one =  field_matrix.col(0);
    Eigen::VectorXd trait_two =  field_matrix.col(1);
    Eigen::MatrixXd covariate_matrix = Eigen::ArrayXXd::Ones(trait_one.rows(), n_covariates + 1);
    if (n_covariates > 0){
        Eigen::MatrixXd covariate_term_matrix(trait_one.rows(), field_list.size() - 2);
   
        for(int col = 0; col < unique_cov_terms.size(); col++){
        
            if(!StringCmp(unique_cov_terms[col].c_str(), "SEX", case_ins)){
                if((field_matrix.col(col + 2).array() == 2.0).count() != 0){
                for(int row = 0 ; row < covariate_term_matrix.rows(); row++){
                    if(field_matrix(row, col + 2) == 2.0){
                        covariate_term_matrix(row, col) = 1.0;
                    }else{
                        covariate_term_matrix(row, col) = 0.0;
                    }
                }
            }else{
                covariate_term_matrix.col(col) = field_matrix.col(col + 2);
            }
            
            }else if(strstr(unique_cov_terms[col].c_str(), "snp_") != NULL || strstr(unique_cov_terms[col].c_str(), "SNP_") != NULL){
                covariate_term_matrix.col(col) = field_matrix.col(col + 2).array() - field_matrix.col(col + 2).mean();
                continue;
            } else {
            covariate_term_matrix.col(col) =  field_matrix.col(col + 2).array() - field_matrix.col(col + 2).mean();
            }
        }
    
        Covariate * cov;
    
        for(int col = 0; (cov = Covariate::index(col)); col++){
            CovariateTerm * cov_term;
            for(cov_term = cov->terms(); cov_term; cov_term = cov_term->next){
                int index = 0;
            
                for(vector<string>::iterator cov_iter = unique_cov_terms.begin(); cov_iter != unique_cov_terms.end(); cov_iter++){
                    if(!StringCmp(cov_term->name, cov_iter->c_str(), case_ins)){
                        break;
                    }
                    index++;
                }
            
                if(cov_term->exponent == 1){
                    covariate_matrix.col(col) = covariate_matrix.col(col).array()*covariate_term_matrix.col(index).array();
                }else{
                    covariate_matrix.col(col) = covariate_matrix.col(col).array()*pow(covariate_term_matrix.col(index).array(), cov_term->exponent);
                }
            }
        
        }
    }    
    Eigen::MatrixXd eigenvectors; 
    Eigen::VectorXd eigenvalues;
    if(evd_data_filename){
        vector<string> evd_ids;
        string evd_id_filename = string(evd_data_filename) + string(".ids");
        ifstream ids_in(evd_id_filename.c_str());
       if(ids_in.is_open() == false){
            string error_string = "File " + evd_id_filename + ".ids not found";
            delete file_data;
            return error_string.c_str();
        }         
        string current_id;
        while (ids_in >> current_id) {
            evd_ids.push_back(current_id);
        }
        ids_in.close();
        vector<string> mle_setup_ids = file_data->get_ids();
        if(mle_setup_ids.size() != evd_ids.size()){
            string error_string = "ID count between EVD data IDs and FPHI trait reader IDs doesn't match\n EVD data ID count: " + to_string(evd_ids.size()) + " FPHI trait reader ID count: " + to_string(mle_setup_ids.size());
            delete file_data;
            return error_string.c_str();
        }
        for(int i  = 0 ; i < evd_ids.size(); i++){
            if(evd_ids[i] != mle_setup_ids[i]){
                string error_string = "At ID index " + to_string(i) + " EVD ID " + evd_ids[i] + "does not match FPHI ID " + mle_setup_ids[i];
            }
       }
       eigenvectors = Eigen::MatrixXd::Zero(evd_ids.size(), evd_ids.size());
       eigenvalues = Eigen::VectorXd::Zero(evd_ids.size());
       string evd_eigenvectors_filename = string(evd_data_filename) + ".eigenvectors"; 
       string evd_eigenvalues_filename = string(evd_data_filename) + ".eigenvalues";
       ifstream eigenvectors_stream(evd_eigenvectors_filename.c_str());
       if(eigenvectors_stream.is_open() == false){
            string error_string = "File " + evd_eigenvectors_filename  + ".eigenvectors not found";
            delete file_data;
            return error_string.c_str();
        }
       ifstream eigenvalues_stream(evd_eigenvalues_filename.c_str());
       if(eigenvalues_stream.is_open() == false){
            string error_string = "File " + evd_eigenvalues_filename + ".eigenvalues not found";
            delete file_data;
            return error_string.c_str();
        }       
       for(int row = 0 ; row < evd_ids.size(); row++){
            eigenvalues_stream >> eigenvalues(row) ;
            for(int col = 0; col < evd_ids.size() ; col++){
                eigenvectors_stream >>  eigenvectors(row, col);
            }
      }
    
      eigenvectors_stream.close();
      eigenvalues_stream.close();
    }else{
                 
        eigenvectors = file_data->get_eigenvectors().transpose();
        eigenvalues = file_data->get_eigenvalues();
    }
    Eigen::VectorXd ones = Eigen::ArrayXd::Ones(trait_one.rows());
    Eigen::VectorXd S_ones = eigenvectors*ones;
    cout.precision(11);
    trait_one = eigenvectors*trait_one;
    trait_two = eigenvectors*trait_two;
    covariate_matrix = eigenvectors*covariate_matrix;


    Eigen::VectorXd final_parameters(6);
    Eigen::VectorXd final_beta(covariate_matrix.cols()*2);
    if(use_fast_version){
        const char * error = 0;
        Eigen::VectorXd T_wald(2);
        double loglik = calculate_bivariate_model_fast(trait_one, trait_two, covariate_matrix, eigenvalues, final_parameters, T_wald);




        if(loglik != loglik ){
            Solar_Eval(interp, "loglike set 0");
            const char * error = "Failed to calculate genetic correlation fast estimates";
            delete file_data;
            return error;
        }
        double rhog_pvalue, rhoe_pvalue;
        cout << T_wald << endl;
        rhoe_pvalue = 2.0*chicdf(T_wald(0), 1);
        rhog_pvalue = 2.0*chicdf(T_wald(1), 1);          
        string loglik_command = "loglike set " + to_string(loglik);
        Solar_Eval(interp, loglik_command.c_str());          
        double h2r_one = final_parameters(0);
        double h2r_two = final_parameters(1);
        double sd_one = final_parameters(2); 
        double sd_two = final_parameters(3);

        final_parameters(0) = h2r_one;
        final_parameters(1) = h2r_two;
        final_parameters(2) = sd_one;
        final_parameters(3) = sd_two;
        string solar_command = "parameter rhog = " + to_string(final_parameters(4));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter rhoe = " + to_string(final_parameters(5));
        Solar_Eval(interp, solar_command.c_str()); 
        solar_command = "parameter h2r_one = " + to_string(final_parameters(0));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter h2r_two = " + to_string(final_parameters(1));
        Solar_Eval(interp, solar_command.c_str());   
        solar_command = "parameter sd_one = " + to_string(final_parameters(2));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter sd_two = " + to_string(final_parameters(3));
        Solar_Eval(interp, solar_command.c_str());         
        const double rhop = final_parameters(4)*sqrt(final_parameters(0))*\
             sqrt(final_parameters(1)) + final_parameters(5)*\
             sqrt(1.0-final_parameters(0))*sqrt(1.0-final_parameters(1));          
        cout << endl;     
        //string solar_command = "set RHOP " + to_string(rhop);
        cout <<  "************************************" << endl;
        cout <<  "*     Genetic Correlation Fast     *" << endl;
        cout <<  "************************************" << endl << endl;        
        cout << "Pedigree: " << currentPed->filename() << endl;
        cout << "Phenotype: " << phenotype_filename << endl;
        cout << "Number of Subjects: " << eigenvalues.rows() << endl;    
        string parameter_names[6 + covariate_matrix.cols()*2];

        parameter_names[0] = trait_list[0] + "-h2r";
        int largest_name = parameter_names[0].length();

        parameter_names[1] = trait_list[1] + "-h2r";
        largest_name = ( parameter_names[1].length() > largest_name )  ? parameter_names[1].length() : largest_name;

        parameter_names[2] = trait_list[0] + "-sd";
        largest_name = ( parameter_names[2].length() > largest_name )  ? parameter_names[2].length() : largest_name;

        parameter_names[3] = trait_list[1] + "-sd";
        largest_name = ( parameter_names[3].length() > largest_name )  ? parameter_names[3].length() : largest_name;

        parameter_names[4] = "rhog";
        largest_name = ( parameter_names[4].length() > largest_name )  ? parameter_names[4].length() : largest_name;  

        parameter_names[5] = "rhoe";
        largest_name = ( parameter_names[5].length() > largest_name )  ? parameter_names[5].length() : largest_name;

        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            parameter_names[i + 6] = trait_list[0] + "-b" + cov_list[i];
            largest_name = ( parameter_names[6 + i].length() > largest_name )  ? parameter_names[6 + i].length() : largest_name;

            parameter_names[i + 6 + covariate_matrix.cols()] = trait_list[1] + "-b" + cov_list[i];
            largest_name = ( parameter_names[i + 6 + covariate_matrix.cols()].length() > largest_name )  ? parameter_names[i + 6 + covariate_matrix.cols()].length() : largest_name;            
        }

        parameter_names[6 + covariate_matrix.cols() - 1] = trait_list[0] + "-mean";
        largest_name = ( parameter_names[6 + covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + covariate_matrix.cols() - 1].length() : largest_name;


        parameter_names[6 + covariate_matrix.cols()*2 - 1] = trait_list[1] + "-mean";
        largest_name = ( parameter_names[6 + 2*covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + 2*covariate_matrix.cols() - 1].length() : largest_name;        
        largest_name = (14 > largest_name ) ? 14 : largest_name;
        cout << setw(largest_name + 4) << "Parameter"  << setw(20) << "Value" << endl << endl;
        
        cout << setw(largest_name + 4) << parameter_names[6 +covariate_matrix.cols() - 1] << setw(20) << final_parameters(6 + covariate_matrix.cols() - 1) << endl;
        cout << setw(largest_name + 4) << parameter_names[6 + covariate_matrix.cols()*2 - 1] << setw(20) << final_parameters(6 + covariate_matrix.cols()*2  - 1)  << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            cout << setw(largest_name + 4) << parameter_names[6 + i] << setw(20) << final_parameters(6 + i) << endl;
            cout << setw(largest_name + 4) << parameter_names[6 + covariate_matrix.cols() + i] << setw(20) << final_parameters(6 + i + covariate_matrix.cols())  << endl;
        }

        for(int i = 0 ; i < 6; i++){
            cout << setw(largest_name + 4) << parameter_names[i] << setw(20) << final_parameters(i)<<  endl;
             
        }
        cout << setw(largest_name + 4) << "rhop" << setw(20) << rhop << endl;
        cout << setw(largest_name + 4) << "loglik" << setw(20) << loglik << endl;
        
        cout << setw(largest_name + 4) << "rhog pvalue" << setw(20) << rhog_pvalue << endl; 
        cout << setw(largest_name + 4) << "rhoe pvalue" << setw(20) << rhoe_pvalue << endl;
        string output_filename = "gen_corr-fast-" + trait_list[0] + "-" + trait_list[1] + ".out";
        ofstream output_stream(output_filename);
        output_stream << "Parameter,Value\n";
        output_stream << parameter_names[6 +covariate_matrix.cols() - 1] << "," << final_parameters(6 + covariate_matrix.cols() - 1)  << endl;
        output_stream << parameter_names[6 + covariate_matrix.cols()*2  - 1] << "," << final_parameters(6 + covariate_matrix.cols()*2  - 1) << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            output_stream << parameter_names[6 + i] << "," << final_parameters(6 + i) << endl;
            output_stream << parameter_names[6 + covariate_matrix.cols() + i] << "," << final_parameters(6 + i + covariate_matrix.cols())<< endl;
        }

        for(int i = 0 ; i < 6; i++){
            output_stream << parameter_names[i] << "," << final_parameters(i)<< endl;
             
        } 
        output_stream << "rhop," << rhop << ",\n";
        output_stream << "loglik," << loglik << ",\n";
        output_stream << "rhog p-value," << rhog_pvalue << ",\n";
        output_stream << "rhoe p-value," << rhoe_pvalue << ",\n"; 

        output_stream.close();                                  
        
        return error;
        
    }
   // double main_best_h;
    int main_iteration_count;
    Eigen::VectorXd standard_errors;
    double main_loglik = calculate_bivariate_model(trait_one,  trait_two,  covariate_matrix, eigenvalues,  final_parameters, final_beta, standard_errors,  main_iteration_count, 0,  debug); 
    if(main_loglik != main_loglik ){
        Solar_Eval(interp, "loglike set 0");
        const char * error = "Convergence of this model could not be achieved";
        delete file_data;
        return error;
    }

    string loglik_command = "loglike set " + to_string(main_loglik);
    Solar_Eval(interp, loglik_command.c_str());   
    if(display_pvalues){
        Eigen::VectorXd rhog_parameters(6);
        Eigen::VectorXd rhog_beta(covariate_matrix.cols()*2);
        int iteration_count;
      
        Eigen::VectorXd rhog_standard_errors;
        double rhog_loglik  = calculate_bivariate_model(trait_one,  trait_two,  covariate_matrix, eigenvalues,  rhog_parameters, rhog_beta, rhog_standard_errors, iteration_count, 1,  debug); 
        if(rhog_loglik != rhog_loglik){
            const char * error = "Convergence for rhog null model could not be achieved";
            delete file_data;
            return error;
        }
        
        
        
        Eigen::VectorXd rhoe_parameters(6);
        Eigen::VectorXd rhoe_beta(covariate_matrix.cols()*2);
        
        Eigen::VectorXd rhoe_standard_errors;
        double rhoe_loglik  = calculate_bivariate_model(trait_one,  trait_two,  covariate_matrix, eigenvalues,  rhoe_parameters, rhoe_beta, rhoe_standard_errors, iteration_count, 2,  debug); 
        if(rhog_loglik != rhog_loglik){
            const char * error = "Convergence for rhoe null model could not be achieved";
            delete file_data;
            return error;
        }
       
        double rhog_pvalue = 1.0;
        if(rhog_loglik == rhog_loglik){
            rhog_pvalue = 2.0*chicdf(2.0*(main_loglik - rhog_loglik), 1); 
        }
        double rhoe_pvalue = 1.0;
        if(rhoe_loglik == rhoe_loglik){
            rhoe_pvalue = 2.0*chicdf(2.0*(main_loglik - rhoe_loglik), 1);
        }
        const double rhop = final_parameters(4)*sqrt(final_parameters(0))*\
             sqrt(final_parameters(1)) + final_parameters(5)*\
             sqrt(1.0-final_parameters(0))*sqrt(1.0-final_parameters(1));        
        /*
        const double rhop = calculate_rho(final_parameters(4))*sqrt(calculate_constraint(final_parameters(0)))*\
             sqrt(calculate_constraint(final_parameters(1))) + calculate_rho(final_parameters(5))*\
             sqrt(1.0-calculate_constraint(final_parameters(0)))*sqrt(1.0-calculate_constraint(final_parameters(1)));

        final_parameters(0) = calculate_constraint(final_parameters(0));
        final_parameters(1) = calculate_constraint(final_parameters(1)); 
        final_parameters(4) = calculate_rho(final_parameters(4));
        final_parameters(5) = calculate_rho(final_parameters(5));  */             
        cout << endl;     
        //string solar_command = "set RHOP " + to_string(rhop);
        cout <<  "*******************************" << endl;
        cout <<  "*     Genetic Correlation     *" << endl;
        cout <<  "*******************************" << endl << endl;
        
        cout << "Pedigree: " << currentPed->filename() << endl;
        cout << "Phenotype: " << phenotype_filename << endl;
        cout << "Number of Subjects: " << eigenvalues.rows() << endl; 
        cout << "Total iterations: " << main_iteration_count << endl;
       // cout << "Numerical Differentiation Delta: " << main_best_h << endl << endl;
      //  cout << "Final parameters\n";
        string parameter_names[6 + covariate_matrix.cols()*2];

        parameter_names[0] = trait_list[0] + "-h2r";
        int largest_name = parameter_names[0].length();

        parameter_names[1] = trait_list[1] + "-h2r";
        largest_name = ( parameter_names[1].length() > largest_name )  ? parameter_names[1].length() : largest_name;

        parameter_names[2] = trait_list[0] + "-sd";
        largest_name = ( parameter_names[2].length() > largest_name )  ? parameter_names[2].length() : largest_name;

        parameter_names[3] = trait_list[1] + "-sd";
        largest_name = ( parameter_names[3].length() > largest_name )  ? parameter_names[3].length() : largest_name;

        parameter_names[4] = "rhog";
        largest_name = ( parameter_names[4].length() > largest_name )  ? parameter_names[4].length() : largest_name;  

        parameter_names[5] = "rhoe";
        largest_name = ( parameter_names[5].length() > largest_name )  ? parameter_names[5].length() : largest_name;

        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            parameter_names[i + 6] = trait_list[0] + "-b" + cov_list[i];
            largest_name = ( parameter_names[6 + i].length() > largest_name )  ? parameter_names[6 + i].length() : largest_name;

            parameter_names[i + 6 + covariate_matrix.cols()] = trait_list[1] + "-b" + cov_list[i];
            largest_name = ( parameter_names[i + 6 + covariate_matrix.cols()].length() > largest_name )  ? parameter_names[i + 6 + covariate_matrix.cols()].length() : largest_name;            
        }

        parameter_names[6 + covariate_matrix.cols() - 1] = trait_list[0] + "-mean";
        largest_name = ( parameter_names[6 + covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + covariate_matrix.cols() - 1].length() : largest_name;


        parameter_names[6 + covariate_matrix.cols()*2 - 1] = trait_list[1] + "-mean";
        largest_name = ( parameter_names[6 + 2*covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + 2*covariate_matrix.cols() - 1].length() : largest_name;        
        largest_name = (14 > largest_name ) ? 14 : largest_name;
        cout << setw(largest_name + 4) << "Parameter"  << setw(20) << "Value" << setw(20) << "Standard Error" << endl << endl;
        
        cout << setw(largest_name + 4) << parameter_names[6 +covariate_matrix.cols() - 1] << setw(20) << final_beta(covariate_matrix.cols() - 1) << setw(20) << standard_errors(6 + covariate_matrix.cols() - 1) << endl;
        cout << setw(largest_name + 4) << parameter_names[6 + covariate_matrix.cols()*2 - 1] << setw(20) << final_beta(covariate_matrix.cols()*2  - 1)  << setw(20) << standard_errors(6 + 2*covariate_matrix.cols() - 1) << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            cout << setw(largest_name + 4) << parameter_names[6 + i] << setw(20) << final_beta(i) << setw(20) << standard_errors(6 + i) << endl;
            cout << setw(largest_name + 4) << parameter_names[6 + covariate_matrix.cols() + i] << setw(20) << final_beta(i + covariate_matrix.cols()) << setw(20) << standard_errors(6 + covariate_matrix.cols() + i) << endl;
        }

        for(int i = 0 ; i < 6; i++){
            cout << setw(largest_name + 4) << parameter_names[i] << setw(20) << final_parameters(i)<< setw(20) << standard_errors(i) << endl;
             
        }
        cout << setw(largest_name + 4) << "rhop" << setw(20) << rhop << endl;
        cout << setw(largest_name + 4) << "loglik" << setw(20) << main_loglik << endl;
        cout << setw(largest_name + 4) << "rhog loglik" << setw(20) << rhog_loglik << endl;
        cout << setw(largest_name + 4) << "rhog p-value" << setw(20) << rhog_pvalue << endl;
        cout << setw(largest_name + 4) << "rhoe loglik" << setw(20) << rhoe_loglik << endl;
        cout << setw(largest_name + 4) << "rhoe p-value" << setw(20) << rhoe_pvalue << endl;
        
        


        string output_filename = "gen_corr-" + trait_list[0] + "-" + trait_list[1] + ".out";
        ofstream output_stream(output_filename);
        output_stream << "Parameter,Value,Standard Error\n";
        output_stream << parameter_names[6 +covariate_matrix.cols() - 1] << "," << final_beta(covariate_matrix.cols() - 1) << "," << standard_errors(6 + covariate_matrix.cols() - 1) << endl;
        output_stream << parameter_names[6 + covariate_matrix.cols()*2  - 1] << "," << final_beta(covariate_matrix.cols()*2  - 1) << "," << standard_errors(6 + 2*covariate_matrix.cols() - 1) << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            output_stream << parameter_names[6 + i] << "," << final_beta(i) << ","<< standard_errors(6 + i) << endl;
            output_stream << parameter_names[6 + covariate_matrix.cols() + i] << "," << final_beta(i + covariate_matrix.cols()) << ","<< standard_errors(6 + covariate_matrix.cols() + i) << endl;
        }

        for(int i = 0 ; i < 6; i++){
            output_stream << parameter_names[i] << "," << final_parameters(i)<< ","<< standard_errors(i) << endl;
             
        } 
        output_stream << "rhop," << rhop << ",\n";
        output_stream << "loglik," << main_loglik << ",\n";
        output_stream << "rhog p-value," << rhog_pvalue << ",\n";
        output_stream << "rhog loglik," << rhog_loglik << ",\n";
        output_stream << "rhoe p-value," << rhoe_pvalue << ",\n"; 
        output_stream << "rhoe loglik," << rhoe_loglik << ",\n"; 
        output_stream.close();    
        
    }else{
         
        const double rhop = final_parameters(4)*sqrt(final_parameters(0))*\
             sqrt(final_parameters(1)) + final_parameters(5)*\
             sqrt(1.0-final_parameters(0))*sqrt(1.0-final_parameters(1));

        /*
        const double rhop = calculate_rho(final_parameters(4))*sqrt(calculate_constraint(final_parameters(0)))*\
             sqrt(calculate_constraint(final_parameters(1))) + calculate_rho(final_parameters(5))*\
             sqrt(1.0-calculate_constraint(final_parameters(0)))*sqrt(1.0-calculate_constraint(final_parameters(1)));        
        final_parameters(0) = calculate_constraint(final_parameters(0));
        final_parameters(1) = calculate_constraint(final_parameters(1)); 
        final_parameters(4) = calculate_rho(final_parameters(4));
        final_parameters(5) = calculate_rho(final_parameters(5));  */
       // string rhop_command = "global gen_corr_RHOP ; set gen_corr_RHOP " + to_string(rhop);
      //  Solar_Eval(interp, rhop_command.c_str());
        string solar_command = "parameter rhog = " + to_string(final_parameters(4));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter rhoe = " + to_string(final_parameters(5));
        Solar_Eval(interp, solar_command.c_str()); 
        solar_command = "parameter h2r_one = " + to_string(final_parameters(0));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter h2r_two = " + to_string(final_parameters(1));
        Solar_Eval(interp, solar_command.c_str());   
        solar_command = "parameter sd_one = " + to_string(final_parameters(2));
        Solar_Eval(interp, solar_command.c_str());
        solar_command = "parameter sd_two = " + to_string(final_parameters(3));
        Solar_Eval(interp, solar_command.c_str());                                
        cout << endl;
        cout << "*******************************" << endl;
        cout <<  "*     Genetic Correlation     *" << endl;
        cout <<  "*******************************" << endl << endl;
        
        cout << "Pedigree: " << currentPed->filename() << endl;
        cout << "Phenotype: " << phenotype_filename << endl;
        cout << "Number of Subjects: " << eigenvalues.rows() << endl; 
        cout << "Total iterations: " << main_iteration_count << endl;
       // cout << "Numerical Differentiation Delta: " << main_best_h << endl << endl;
        
        string parameter_names[6 + covariate_matrix.cols()*2];

        parameter_names[0] = trait_list[0] + "-h2r";
        int largest_name = parameter_names[0].length();

        parameter_names[1] = trait_list[1] + "-h2r";
        largest_name = ( parameter_names[1].length() > largest_name )  ? parameter_names[1].length() : largest_name;

        parameter_names[2] = trait_list[0] + "-sd";
        largest_name = ( parameter_names[2].length() > largest_name )  ? parameter_names[2].length() : largest_name;

        parameter_names[3] = trait_list[1] + "-sd";
        largest_name = ( parameter_names[3].length() > largest_name )  ? parameter_names[3].length() : largest_name;

        parameter_names[4] = "rhog";
        largest_name = ( parameter_names[4].length() > largest_name )  ? parameter_names[4].length() : largest_name;  

        parameter_names[5] = "rhoe";
        largest_name = ( parameter_names[5].length() > largest_name )  ? parameter_names[5].length() : largest_name;

        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            parameter_names[i + 6] = trait_list[0] + "-b" + cov_list[i];
            largest_name = ( parameter_names[6 + i].length() > largest_name )  ? parameter_names[6 + i].length() : largest_name;

            parameter_names[i + 6 + covariate_matrix.cols()] = trait_list[1] + "-b" + cov_list[i];
            largest_name = ( parameter_names[i + 6 + covariate_matrix.cols()].length() > largest_name )  ? parameter_names[i + 6 + covariate_matrix.cols()].length() : largest_name;            
        }

        parameter_names[6 + covariate_matrix.cols() - 1] = trait_list[0] + "-mean";
        largest_name = ( parameter_names[6 + covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + covariate_matrix.cols() - 1].length() : largest_name;


        parameter_names[6 + covariate_matrix.cols()*2 - 1] = trait_list[1] + "-mean";
        largest_name = ( parameter_names[6 + 2*covariate_matrix.cols() - 1].length() > largest_name )  ? parameter_names[6 + 2*covariate_matrix.cols() - 1].length() : largest_name;        
        largest_name = (14 > largest_name ) ? 14 : largest_name;
       // cout << setw((largest_name + 4)*2) << "Final Parameters" << endl << endl;
        cout << setw(largest_name + 4) << "Parameter"  << setw(20) << "Value" << setw(20) << "Standard Error" << endl << endl;
        
        cout << setw(largest_name + 4) << parameter_names[6 +covariate_matrix.cols() - 1] << setw(20) << final_beta(covariate_matrix.cols() - 1) << setw(20) << standard_errors(6 + covariate_matrix.cols() - 1) << endl;
        cout << setw(largest_name + 4) << parameter_names[6 + 2*covariate_matrix.cols()  - 1] << setw(20) << final_beta(covariate_matrix.cols()*2  - 1) << setw(20) << standard_errors(6 + 2*covariate_matrix.cols() - 1) << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            cout << setw(largest_name + 4) << parameter_names[6 + i] << setw(20) << final_beta(i) << setw(20) << standard_errors(6 + i) << endl;
            cout << setw(largest_name + 4) << parameter_names[6 + covariate_matrix.cols() + i] << setw(20) << final_beta(i + covariate_matrix.cols()) << setw(20) << standard_errors(6 + covariate_matrix.cols() + i) << endl;
        }

        for(int i = 0 ; i < 6; i++){
            cout << setw(largest_name + 4) << parameter_names[i] << setw(20) << final_parameters(i)<< setw(20) << standard_errors(i) << endl;
             
        }
        cout << setw(largest_name + 4) << "rhop" << setw(20) << rhop << endl;
        cout << setw(largest_name + 4) << "loglik" << setw(20) << main_loglik << endl;
 


        string output_filename = "gen_corr-" + trait_list[0] + "-" + trait_list[1] + ".out";
        ofstream output_stream(output_filename);
        output_stream << "Parameter,Value,Standard Error\n";
        output_stream << parameter_names[6 +covariate_matrix.cols() - 1] << "," << final_beta(covariate_matrix.cols() - 1)  << "," << standard_errors(6 + covariate_matrix.cols() - 1) << endl;
        output_stream << parameter_names[6 + covariate_matrix.cols()*2 - 1] << "," << final_beta(covariate_matrix.cols()*2  - 1)  << "," << standard_errors(6 + 2*covariate_matrix.cols() - 1) << endl;
        for(int i = 0 ; i < covariate_matrix.cols() - 1; i++){
            output_stream << parameter_names[6 + i] << "," << final_beta(i) << "," << standard_errors(6 + i) << endl;
            output_stream << parameter_names[6 + covariate_matrix.cols() + i] << "," << final_beta(i + covariate_matrix.cols()) << ","<< standard_errors(6 + covariate_matrix.cols() + i) << endl;
        }

        for(int i = 0 ; i < 6; i++){
            output_stream << parameter_names[i] << "," << final_parameters(i)<< ","<< standard_errors(i) << endl;
             
        } 
        output_stream << "rhop," << rhop << ",\n";
        output_stream << "loglik," << main_loglik << ",\n";
        output_stream.close();             

        }                        
                                
       
  
    delete file_data; 
    const char *error = 0;
    return error;               
} 
static void print_genetic_correlation_help(Tcl_Interp * interp){
    Solar_Eval(interp, "help gen_corr");
} 

extern "C" int run_genetic_correlation(ClientData clientData, Tcl_Interp * interp,
                          int argc, const char * argv[]){
  
    bool debug_mode = false;
 


    const char * evd_data_filename = 0;
    
    bool use_fast_version = false;

    double h = 0.01;
    bool get_pvalues = false;
    for(int arg = 1 ;arg < argc ; arg++){
        if(!StringCmp(argv[arg], "help", case_ins) || !StringCmp(argv[arg], "-help", case_ins) || !StringCmp(argv[arg], "--help", case_ins)
           || !StringCmp(argv[arg], "h", case_ins) || !StringCmp(argv[arg], "-h", case_ins) || !StringCmp(argv[arg], "--help", case_ins)){
            print_genetic_correlation_help(interp);
            return TCL_OK;
        }else if (!StringCmp(argv[arg], "-d", case_ins) || !StringCmp(argv[arg], "--d", case_ins) || !StringCmp(argv[arg], "-debug", case_ins) ||
                  !StringCmp(argv[arg], "--debug", case_ins)){
            
            debug_mode  = true;
        }else if (!StringCmp(argv[arg], "-pvalues", case_ins) ||
                  !StringCmp(argv[arg], "--pvalues", case_ins)){
            
            get_pvalues  = true;
        }else if ((!StringCmp(argv[arg], "-evd_data", case_ins) || !StringCmp(argv[arg], "--evd_data", case_ins)) && arg + 1 < argc){
            evd_data_filename = argv[++arg];
        /*}else if ((!StringCmp(argv[arg], "-delta", case_ins) || !StringCmp(argv[arg], "--delta", case_ins)) && arg + 1 < argc){
            //evd_data_filename = argv[++arg];
            h = atof(argv[++arg]);*/
        }else if ((!StringCmp(argv[arg], "-fast", case_ins) || !StringCmp(argv[arg], "--fast", case_ins))) {
            use_fast_version = true;
        /*}else if ((!StringCmp(argv[arg], "-delta", case_ins) || !StringCmp(argv[arg], "--delta", case_ins)) && arg + 1 < argc){
            //evd_data_filename = argv[++arg];
            h = atof(argv[++arg]);*/
        }else{
            RESULT_LIT("Invalid argument enter see help");
            return TCL_ERROR;
        }
    }
    
    if(!loadedPed()){
        RESULT_LIT("No pedigree has been loaded");
        return TCL_ERROR;
    }
    const char * phenotype_filename = 0;
    phenotype_filename = Phenotypes::filenames();
    if(!phenotype_filename){
	RESULT_LIT("No phenotype file has bee loaded");
	return TCL_ERROR;
    }
    const char * pedigree_filename = currentPed->filename();
    
    if (Trait::Number_Of() != 2){
        RESULT_LIT( "Genetic correlation command requires two traits");
        return TCL_ERROR;
    }
    vector<string> trait_list;
    trait_list.push_back(string(Trait::Name(0)));
    trait_list.push_back(string(Trait::Name(1)));

    const char * error_msg = calculate_genetic_correlation(interp, trait_list,  phenotype_filename, get_pvalues, debug_mode, use_fast_version, evd_data_filename);
    if(error_msg){
        RESULT_BUF(error_msg);
        return TCL_ERROR;
    }
    return TCL_OK;
}
extern "C" int transfer_pedigree_filename(ClientData clientData, Tcl_Interp * interp,
                          int argc, const char * argv[]){
      if(!loadedPed()){
        RESULT_LIT(0);
        return TCL_ERROR;
    }
    const char * pedigree_filename = currentPed->filename();
    RESULT_BUF(pedigree_filename);
    return TCL_OK;
    
} 
