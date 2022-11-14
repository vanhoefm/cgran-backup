/** location.cpp: Implementation of location class functions
 * used for manipulating 2D positions of objects
 * FUTURE: Expand this class into a 3D position object
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

#include <iostream>
#include <math.h>

#include "bitutils.h"
#include "location.h"

#ifdef M_PI
  #define PI M_PI
#else
  #define PI 3.14159265
#endif

using namespace std;

/*----------Location class implementations-------------*/

int getLocationSize(void) {
  return (int) (3 * sizeof(float));
}

//Avoid using this one!
Location::Location(void){
  x = 0;
  y = 0;
  z = 0;
}

Location::Location(float xx, float yy, float zz){
  x = xx;
  y = yy;
  z = zz;
}

Location::Location(Location* other){
  if (other == NULL){
    cerr << "Warning: Made location using NULL pointer. Defaulting to 0, 0, 0\n";
    x = 0;
    y = 0;
    z = 0;
  } else {
    //Customary behavior
    x = other->x;
    y = other->y;
    z = other->z;
  }
}


int Location::serialize(char* buf, int bufsize){
  if (bufsize < 3 * sizeof(float))
    throw "ERROR: Buffer is not big enough to store location.";
  int a = 0;
  memset(buf, '\0', bufsize);
  nrl_htonf(x, &buf[a]);
  a += sizeof(float);
  nrl_htonf(y, &buf[a]);
  a += sizeof(float);
  nrl_htonf(z, &buf[a]);
  a += sizeof(float);
  return a;
}


Location::Location(char* buf, int bufsize){
  if (bufsize < 3 * sizeof(float))
    throw "ERROR: Buffer is not big enough to read location from.";
  int a = 0;
  x = nrl_ntohf(&buf[a]);
  a += sizeof(float);
  y = nrl_ntohf(&buf[a]);
  a += sizeof(float);
  z = nrl_ntohf(&buf[a]);
  return;
}


void Location::print(ostream* fs){
  *fs << "{ " << x << " " << y << " " << z << " }\n";
}

void Location::printcsv(ostream* fs){
  //Note: This prints Locations using comma separations. 
  //Packets use semicolon separations. 
  *fs << x << "," << y << "," << z;
}

Location* Location::copy(void){
  return new Location(*this);
}


bool Location::isSame(Location* other){
  bool issame = true;
  issame = issame && (x == other->x);
  issame = issame && (y == other->y);
  issame = issame && (z == other->z);
  return issame;
}


Location Location::operator+(Location other){
  return Location(x + other.x, y + other.y, z + other.z);
}

//Used for inverting the velocity
Location Location::operator-(void){
  x = -x;
  y = -y;
  z = -z;
  return *this;
}

Location Location::operator-(Location other){
  int xx, yy, zz;
  xx = x - other.x;
  yy = y - other.y;
  zz = z - other.z;
  return Location(xx, yy, zz);
}


double Location::distanceToDest(Location* dest){
  //Note: The returned value is always positive and doesn't indicate direction.
  Location diff = *dest - *this;
  return sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}


//BUG: Does not account for Z dimension
IntruderBitmapType Location::indicateDirection(void){
  //First check to see if the velocity is small enough to be considered 0
  if ((x * x + y * y) < SMALL_VELOCITY * SMALL_VELOCITY)
    return VEL_STOP;
  
  //Calculate the angle and use it to determine our direction
  //First, convert XY into polar coordinates
  float theta = 180 / PI * atanf(x/y); //The only error condition is NaN and it defaults to East
  //Map [-90, 90] to [0, 360] and correct for quadrant
  if (y < 0)
    theta += 270;
  else
    theta += 90;
  
  //Oops, we accidentally have it 180 degress off. This is likely due to a coordinate system conflict.
  theta = fmodf(theta + 180, 360);
    
  //Now that we have polar form, return appropriate orientation  
  if (22.5 <= theta && theta < 67.5)
    return VEL_NE;
  else if (67.5 <= theta && theta < 112.5)
    return VEL_N;
  else if (112.5 <= theta && theta < 157.5)
    return VEL_NW;
  else if (157.5 <= theta && theta < 202.5)
    return VEL_W;
  else if (202.5 <= theta && theta < 247.5)
    return VEL_SW;
  else if (247.5 <= theta && theta < 292.5)
    return VEL_S;
  else if (292.5 <= theta && theta < 337.5)
    return VEL_SE;
  else
    return VEL_E;
}
