/* solvermath.cpp: Mathematical algorithms for the Localization Solver
 * 
 * @author Brian Shaw
 * 
 */

#include "solver.h"

#define PATHLOSS_AT_1M 7.47579

//BEGIN items moved from solver.cpp to here

// Key global required to find RSSI. Determined through calibration.
float rssi_at_1m;

struct global_params gparams;

//END items moved from solver.cpp to here

//NOTE: Be sure to implement inverses for each path loss model we make!


//Compute distance using RSSI!
//This MUST NOT be run until after calibration!! Otherwise rssi_at_1m is invalid
//Note: comparable ToA calculations implemented in the packet class
//FIXME??? Add calibration_complete???
float rssi_to_dist(float rssi, bool calibration_complete){
  
  float internal_rssi_1m;
  if (calibration_complete)
    internal_rssi_1m = rssi_at_1m;
  else
    internal_rssi_1m = DEFAULT_RSSI_1M;
  
  /*
  //Calcuate Distance from AP RSSI
  float lambda= 2.45 / C_NSEC;
  //double alpha=2; //set to what we choose best fit of our choice
  
  //This is FREE SPACE path loss! We need a better model. FIXME
  double temp;
  
  temp = pow(10, ((rssi - PATHLOSS_AT_1M - rssi_at_1m) / -20));
  temp = sqrt(temp);
  float dist = temp * lambda / (4 * M_PI);
  //printf("dist = %lf\n", dist);
  return dist;
  */
  double temp = (rssi - internal_rssi_1m) / -20;
  return pow(10, temp) + 1;
}


//Convert 1 distance to an RSSI value
//With the real BBN code this should only be used after calibration
float dist_to_rssi(double dist, bool calibration_complete){
  
  float internal_rssi_1m;
  if (calibration_complete)
    internal_rssi_1m = rssi_at_1m;
  else
    internal_rssi_1m = DEFAULT_RSSI_1M;
  
  /*
  //Inverse of FREE SPACE path loss!
  double lambda = 2.45 / C_NSEC;
  
  double temp = 4 * M_PI * dist / lambda;
  float rssi = -20 * (float) log10(temp * temp);
  
  //printf("RSSI calculated is %f\n", rssi);
  //Note: "rssi" of 1m is equal to -PATHLOSS_AT_1M when distance is 1m. 
  return rssi + internal_rssi_1m; // + PATHLOSS_AT_1M;
  */
  //FIXME: -10 * alpha
  return (-20 * log10(dist - 1) + internal_rssi_1m);
  
}
  


//Average (mean) of the elements in an array of doubles
double findavg(double* array, int maxindex){
  int a = 0;
  double accum = 0;
  //printf("Max index is %d\n", maxindex);
  for (a = 0; a < maxindex; a++){
    //printf("Array element %d is %lf\n", a, array[a]);
    accum += array[a];
  }
  return accum / (double) maxindex;
}

//Standard deviation of the elements in an array of doubles
//Overloading so we don't have to repeat averaging (when average is available)
double findstddev(double mean, double* array, int maxindex){
  int a;
  double accum = 0;
  for (a = 0; a < maxindex; a++){
    double dev2 = (mean - array[a]) * (mean - array[a]);
    accum += dev2;
  }
  return sqrt(accum / ((double) maxindex - 1));
}


//Standard deviation of the elements in an array of doubles
//In case average hasn't been calculated yet
double findstddev(double* array, int maxindex){
  double mean = findavg(array, maxindex);
  return findstddev(mean, array, maxindex);
}


/* Basic Kalman filter. This is a derivative work of code by Adrian Boeing.
 * 
 * The original source can be found at: http://snippets.dzone.com/posts/show/11215
 * 
 * It has been converted to a C++ class to serve as a basic, generic Kalman filter for our application
 * 
 * @author Adrian Boeing
 * @author Brian Shaw
 * 
 */

//Return in range (-1, 1)
double frand(void) {
    return 2*((rand()/(double)RAND_MAX) - 0.5);
}


//Constructor. Take in initial guess for the variable, Q, and R.
//If you're not sure what the initial guess is, start with 0 and tune it.
//Keep in mind the calibration period of 1m LOS from one BS
SimpleKalman::SimpleKalman(double var_estimate, float QQ, float RR){
  var = var_estimate;
  P_last = 0;
  Q = QQ;
  R = RR;
}


double SimpleKalman::kfilter(double measurement){
    //initial values for the kalman filter
    //the noise in the system
    //float Q = 0.022;
    //float R = 0.617;
    float K;
    float P;
    float P_temp;
    
    //do a prediction
    P_temp = P_last + Q;
    //calculate the Kalman gain
    K = P_temp * (1.0/(P_temp + R));
    //measure
    //measurement = z_real + frand()*0.09; //the real measurement plus noise
    //correct
    var = var + ((double) K) * (measurement - var); 
    P = (1 - K) * P_temp;
    //we have our new system, update P_last
    P_last = P;
    
    return var;
}

