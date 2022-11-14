/** location.h: Class for manipulating 2D positions of objects
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
#ifndef NRL_LOCATION_H
#define NRL_LOCATION_H

#include "templates.h"
#include <stdint.h>

using namespace std;

///Tolerance for what velocity values indicate "stopped"
#define SMALL_VELOCITY		5

enum IntruderBitmapType {
  VEL_N = 10,
  VEL_NE,
  VEL_E,
  VEL_SE,
  VEL_S,
  VEL_SW,
  VEL_W,
  VEL_NW,
  VEL_STOP
};


class Location : public Printable, public Serializable {
public:
  float x;
  float y;
  float z;
  Location();
  Location(float, float, float);
  Location(Location*);
  //Printing and serialization functions
  Location(char* buf, int buflen);
  void print(ostream* fs);
  void printcsv(ostream* fs);
  Location* copy(void);
  int serialize(char* buf, int buflen);
  bool isSame(Location* other);
  
  Location operator+(Location other);
  Location operator-(void); //Unary -, aka negation
  Location operator-(Location other);
  
  double distanceToDest(Location* dest); //Call this as "source->distanceToDest(dest)"
  IntruderBitmapType indicateDirection(void);
};

//Function for finding out how big a Location is
int getLocationSize(void) ;


#endif
