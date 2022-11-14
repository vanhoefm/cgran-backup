/** simplegeometriclocalizer.h: A simplistic position-finding algorithm
 * 
 * This design is based on Ryan Dobbins' calculations,
 * programmed by Brian Shaw
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
#ifndef SIMPLE_GEO_LOCALIZER_H
#define SIMPLE_GEO_LOCALIZER_H

#include "modeltemplates.h"

using namespace std;


struct SimpleGeometricLocalizerParams : public DistanceLocalizerParams{
  float zoffset;
  
  virtual SimpleGeometricLocalizerParams* copy(void);
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  ParamsType getType(void) { return PT_SGEOLOC; };
};


class SimpleGeometricLocalizer : public DistanceLocalizer{
protected:
  float zoffset;
  vector<Location*> returnerror(void); //Helper function for returning NULLs
public:
  SimpleGeometricLocalizer(SimpleGeometricLocalizerParams* params);
  ///findposition: calculate node's position
  ///Pre: given distances and locations for each sensor, in order
  ///Post: returns calculated location, uncertainty in a vector of size 2
  virtual vector<Location*> findposition(vector<double> distances, vector<Location*> lcns);
  virtual bool setparams(DistanceLocalizerParams* params);
};


#endif
