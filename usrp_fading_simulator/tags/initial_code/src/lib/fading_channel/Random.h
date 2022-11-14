/*definitions used by the random functions*/
#ifndef RANDOM
#define RANDOM


#include <math.h>

#define Ia   16807
#define Im   2147483647
#define Am   (1.0/Im)
#define Iq   127773
#define Ir   2836
#define Ntab 32
#define Ndiv (1+(Im-1)/Ntab)
#define Eps  1.2e-7
#define Rnmx (1.0-Eps)
 
// extern float drand48();
 
float Uniform(long *idum);
/*Returns a uniform RV in (0,1) - Any seed<-1 can be used*/
 
float Gaussian(long *idum); 
/*Returns a Gaussian RV ~ N(0,1) - Uses  Uniform from above - 
                                 - Any seed<-1 can be used */

#endif

