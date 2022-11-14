/** read_params.cpp: Reads in sensor's parameters
 * 
 * (FUTURE) use something similar for solver and simulator's
 * default parameters?
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

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

//#define NRL_PARAMS_FILE_NAME	"nrl_solver_config.txt"
#include "read_params.h" //comment out for testing
#include "sensor_globals.h"

using namespace std;

/* dummy Location for testing only
class Location{
  int a;
};
*/


Location* nrl_read_params(char* solvername, int solvernamelen, int* port){
  FILE* paramf = fopen(NRL_PARAMS_FILE_NAME, "r");
  if (paramf == NULL){
    perror("Error: unable to open parameters file");
    exit(1);
  }
  
  char buf[256];
  fgets(buf, 256, paramf);
  if (strncmp(buf, "# NRL Solver Params", 19)){
    printf("Error: invalid params file!\n");
    exit(1);
  }
  
  int x = -1;
  int y = -1;
  int z = -1;
  
  //Get host name and store it.
  fgets(buf, 256, paramf);
  char* ptr = strchr(buf, '\n');
  *ptr = '\0'; //Replace the newline at the end with string delimeter.
  if (!strncasecmp(buf, "host", 4))
    strncpy(solvername, &buf[5], solvernamelen);
  else {
    printf("Error: Unable to read hostname\n");
    exit(1);
  }
  printf("Using solver host %s\n", buf);
  
  fgets(buf, 256, paramf);
  if (!strncasecmp(buf, "port", 4)){
    char* ptr = strchr(buf, '\n');
    *ptr = '\0'; //Replace the newline at the end with string delimeter.
    *port = atoi(&buf[5]);
  } else {
    printf("Error: Unable to read hostname\n");
    exit(1);
  }
  printf("Using port %d\n", buf);
  
  
  //Get X, Y, Z
  fgets(buf, 256, paramf);
  if (buf[0] == 'X')
    x = atof(&buf[2]);
  else
    printf("Warning: Could not read X value\n");
  fgets(buf, 256, paramf);
  if (buf[0] == 'Y')
    y = atof(&buf[2]);
  else
    printf("Warning: Could not read X value\n");
  fgets(buf, 256, paramf);
  if (buf[0] == 'Z')
    z = atof(&buf[2]);
  else
    printf("Warning: Could not read X value\n");
  printf("X: %d Y: %d Z: %d\n", x, y, z);
  return new Location();
}

/* test main() function
int main(int argc, char* argv[]){
  char buf2[256];
  nrl_read_params(buf2, 256);
  printf("Buf2 is %s\n", buf2);
  return 0;
}
*/