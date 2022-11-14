/** lightspeedtoa.h: Simple time-based distance calculator
 * uses customizable speed of light (free space, in air, through solid material, etc)
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
#ifndef LIGHTSPEED_TOA_H
#define LIGHTSPEED_TOA_H

#include "modeltemplates.h"
#include "measurement.h"
#include "timestamp.h"
#include <vector>
#include <ostream>

using namespace std;

struct LightspeedToAParams : public ToADistanceCalculatorParams{ //ToADistanceCalculatorParams{
  double c_nsec; ///Speed of light being used
  //vector<Timestamp*> corrtime; ///Constant correction factors for each time
  //vector<Location*> corrloc; ///  Locations of each sensor being corrected
  Timestamp* timecorr; ///processing delay corresponding to two RF frontends
  
  //Params required functions
  virtual ParamsType getType(void) { return PT_LIGHTSPEED; };
  //Printable required functions
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  virtual LightspeedToAParams* copy(void);
};

/// Simple ToA model based on lightspeed. Defaults to speed of light in a vacuum
class LightspeedToA : public ToADistanceCalculator{
protected:
  double c_nsec;
  //vector<Timestamp*> corrtime;
  //vector<Location*> corrloc;
  Timestamp* timecorr;
  //helper function for time correction
  //Previous implementation was O(s)
  Timestamp* correctTime(Packet* pkt);
  //void addTimeCorrection(Packet* pkt);
public:
  LightspeedToA(LightspeedToAParams* params);
  ~LightspeedToA(void);
  
  virtual vector<double> toa_to_dist(Measurement* meas);
  virtual Timestamp* dist_to_toa(double dist, Timestamp* time); 
  virtual bool setparams(ToADistanceCalculatorParams* params);
};

#endif
