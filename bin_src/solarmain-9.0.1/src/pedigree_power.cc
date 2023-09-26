#include "solar.h"
#include "Eigen/Dense"
#include <iostream>
#include <iomanip>
using namespace std;
extern bool loadedPed ();
extern Pedigree *currentPed;

extern "C" void symeig_ (int*, double*, double*, double*, double*, int*);
static void calculate_eigenvectors_and_eigenvalues (double * phi2, double * eigenvectors,double * eigenvalues, int n)
{

    double* e =  new double[n];
    memset(e, 0, sizeof(double)*n);
    int * info = new int;
    *info  = 0;
    symeig_(&n, phi2, eigenvalues, e, eigenvectors, info);
    delete [] e;
    delete [] info;
}

static inline double function_g(const double h2, const double geo_mean){
    return 1.0 + h2*(geo_mean - 1.0);
}

static const char  * pedigree_power(Tcl_Interp * interp, const double null_h2r){
    Matrix * static_phi2 = 0;
    static_phi2 = Matrix::find("phi2");
    if (!static_phi2) {
        Solar_Eval(interp, "matrix load phi2.gz phi2");
        static_phi2 = Matrix::find("phi2");
        if(!static_phi2){
            return "Phi2 matrix could not be loaded";
        }
    }
    const int highest_ibdid = Matrix::Pedindex_Highest_Ibdid;
    double * phi2_matrix = new double[highest_ibdid*highest_ibdid];
    
    for(int col = 0; col < highest_ibdid; col++){
        for (int row = col; row < highest_ibdid; row++){
            double phi2_value;
            try{
                phi2_value = static_phi2->get(col + 1, row+1);
            }catch(...){
                string error = "Failed to load phi2 value with ibdids " + to_string(col+1) + " col " + to_string(row+1)  + " row";
                return error.c_str();
            }
            phi2_matrix[col*highest_ibdid + row] = phi2_matrix[row*highest_ibdid + col] = phi2_value;
        }
   }
   double * temp_eigenvectors = new double[highest_ibdid*highest_ibdid];
   double * temp_eigenvalues = new double[highest_ibdid];
   calculate_eigenvectors_and_eigenvalues (phi2_matrix, temp_eigenvectors, temp_eigenvalues, highest_ibdid);
   delete [] phi2_matrix;
   Eigen::VectorXd eigenvalues = Eigen::Map<Eigen::VectorXd>(temp_eigenvalues, highest_ibdid);
   Eigen::VectorXd demeaned_eigenvalues = (eigenvalues.array() - 1).matrix();
   const double variance = demeaned_eigenvalues.squaredNorm()/(eigenvalues.rows() - 1);

   
   cout << "* * * * * * Pedigree Power * * * * * *\n \n";
   cout << "        Pedigree: " <<  currentPed->filename() << endl;
   cout << "     h2r     ELRT\n";
   for (double h2r = 0.1; h2r <= 1.0; h2r += 0.1){
        const double elrt = 1.0 + ((eigenvalues.rows() - 1)*variance*pow(h2r - null_h2r, 2)/2.0);
        cout << setw(8) << h2r <<"    " << setw(8) << elrt << endl;
   }

   
/*
   cout << "Pedigree Power Method Four\n";
   cout << setw(6) << "h2 " << setw(8) << "ELRT\n";
   for (double h2r = 0.0 ; h2r <= 1.0; h2r += 0.1){
        const double elrt = -log(1.0 + h2r*(eigenvalues.array() - 1.0)).sum();
        cout << setw(6) << h2r << setw(8) << elrt << endl;
   }   */
 /*  cout << "Pedigree Power Method B\n";
   cout << setw(6) << "h2 " << setw(8) << "ELRT\n";
   double geometric_mean = 1.0;
   for(int i = 0; i < eigenvalues.rows(); i++){
    geometric_mean *= eigenvalues(i);
   }
   geometric_mean = pow(geometric_mean, 1.0/eigenvalues.rows());
   Eigen::VectorXd q = (log(eigenvalues.array())).matrix();
   double q_variance = 0.0;
   for(int i = 0; i < eigenvalues.rows(); i++){
    q_variance += pow(q(i) - log(geometric_mean), 2);
   }
   q_variance /= (q.rows() - 1);
   cout << geometric_mean << " " << eigenvalues.rows() << endl;
   for (double h2r = 0.1 ; h2r <= 1.0; h2r += 0.1){
        const double elrt = q.rows()*(-log(function_g(h2r, geometric_mean)/function_g(null_h2r, geometric_mean)) + (function_g(h2r, geometric_mean)/function_g(null_h2r, geometric_mean)) - 1.0) + 1.0 + (((q.rows() - 1.0)*q_variance*geometric_mean)/2.0)*((null_h2r*(null_h2r-1.0)/pow(function_g(null_h2r, geometric_mean), 2)) + (h2r*(h2r-1.0)/pow(function_g(h2r, geometric_mean),2)) - ((geometric_mean*(null_h2r*geometric_mean + null_h2r -1.0)*(h2r - null_h2r))/pow(function_g(null_h2r, geometric_mean), 3)));
        

       
        cout << setw(6) << h2r << setw(8) << elrt << endl;
   }*/
      
   delete [] temp_eigenvectors;
   delete [] temp_eigenvalues;    
     
   return 0;
}    
    

extern "C" int pedigree_power_command(ClientData clientData, Tcl_Interp * interp,
                          int argc, const char * argv[]){
     double null_h2r = 0.0;
     
     for(int arg = 1; arg < argc; arg++){
        if((!StringCmp(argv[arg], "--null", case_ins) || !StringCmp(argv[arg], "-null", case_ins) || !StringCmp(argv[arg], "--n", case_ins)
           || !StringCmp(argv[arg], "-n", case_ins)) && arg + 1 < argc){
           null_h2r = atof(argv[++arg]);
           if (null_h2r < 0.0 || null_h2r > 1.0){
                RESULT_LIT("Null h2r must be greater than or equal to 0.0 or less than or equal to 1.0");
                return TCL_ERROR;
           }
        }else{
            RESULT_LIT("Invalid argument enter see help");
            return TCL_ERROR;
        }
     }
                           
                            
    if(!loadedPed()){
        RESULT_LIT("No pedigree has been loaded");
        return TCL_ERROR;
    }
    const char * error = pedigree_power(interp, null_h2r);
    if (error){
        RESULT_LIT(error);
        return TCL_ERROR;
    }
    
    return TCL_OK;
    
}
    
        
