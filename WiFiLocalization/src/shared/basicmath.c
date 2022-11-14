/** basicmath.c: Basic mathematical tools
 * 
 * @author Brian Shaw
 * 
 */
/* 
 * This file is part of WiFi Localization
 * 
 * This program is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation; either version 2 of the License, or 
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License along 
 * with this program; if not, write to the Free Software Foundation, Inc., 
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
#include "basicmath.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h> //For RAND_MAX
#include <stdint.h>

#ifndef M_PI
#define M_PI	3.14159265
#endif

///Randomly generate the time variance to +/- "desired"
double generate_uniform_variance(double desired){
  //Generate random fraction
  double randnum = ((double) rand()) / ((double) RAND_MAX);
  //printf("Random decimal: %lf\n", randnum);
  //Scale by the desired range
  return desired * randnum;
}


///Generate normally distributed noise using Box-Mueller Transform
double generate_normal_variance(double mu, double sigma){
  double u1 = generate_uniform_variance(1);
  u1 = u1 + !u1; //If u1 == 0, u1 = 0 + 1; else u1 = u1 + 0;
  double u2 = generate_uniform_variance(1);
  u2 = u2 + !u2;
  double Z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
  return mu + Z * sigma;
}


#ifdef __cplusplus
} /* end of extern "C" */
#endif
