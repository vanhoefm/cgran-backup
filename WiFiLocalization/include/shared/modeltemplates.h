/** modeltemplates.h: Generic solver model and framework types
 * and their respective parameter structs
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
#ifndef MODEL_TEMPLATES_H
#define MODEL_TEMPLATES_H

#include "measurement.h"
#include "timestamp.h"
#include "params.h"
#include <vector>

using namespace std;


//Made up value for the default RSSI at 1 meter. 
//It should never be seen, but having it makes the code more robust.
#define DEFAULT_RSSI_1M		-27


/// Template for all localization frameworks
struct LocalizationFrameworkParams : public Params{
  virtual LocalizationFrameworkParams* copy(void) = 0;
  //Other virtual functions inherited from Params
};

class LocalizationFramework{
public:
  virtual void localize(Measurement* meas) = 0;
  virtual bool setparams(LocalizationFrameworkParams* params) = 0;
  virtual bool isUnifiedFramework(void) = 0;
  virtual bool isFourStepFramework(void) = 0;
  virtual bool isFiveStepFramework(void) = 0;
};

#if 0

struct UnifiedRSSILocalizerParams : public Params{
};

/// Template for all unified RSSI localization algorithms
/// NOTE: Not used as of 0.1.0, but useful later for mapping-based algorithms
class UnifiedRSSILocalizer{
protected:
  ParamsWrapper* params;
public:
  virtual void localize(Measurement* meas) = 0;
  //FIXME: Add function so these models can be used to generate packets
  virtual bool setparams(ParamsWrapper* params) = 0;
};

#endif

struct ToADistanceCalculatorParams : public Params{
  virtual ToADistanceCalculatorParams* copy(void) = 0;
  //Other virtual functions inherited from Params
};

/// Template for all ToA Distance Calculators
class ToADistanceCalculator{
public:
  virtual vector<double> toa_to_dist(Measurement* meas) = 0;
  virtual Timestamp* dist_to_toa(double dist, Timestamp* time) = 0; ///Returns difference between the two times
  virtual bool setparams(ToADistanceCalculatorParams* params) = 0;
};


struct PathLossModelParams : public Params{
  virtual PathLossModelParams* copy(void) = 0;
  //Other virtual functions inherited from Params
};

/// Template for all RSSI Distance Calculators
class PathLossModel{
protected:
  bool calibration_complete;
  float rssi_1m;
public:
  virtual double rssi_to_dist(float rssi) = 0;
  virtual vector<double> rssi_to_dist(Measurement* meas) = 0;
  virtual float dist_to_rssi(double dist) = 0;
  void setknownrssi(float rssi, double dist);
  virtual bool setparams(PathLossModelParams* params) = 0;
};


struct DistanceLocalizerParams : public Params{
  virtual DistanceLocalizerParams* copy(void) = 0;
  //Other virtual functions inherited from Params
};

/// Template for all Localization Solvers using distances as inputs
class DistanceLocalizer{
public:
  ///findposition: calculate node's position
  ///Pre: given distances and locations for each sensor, in order
  ///Post: returns calculated location, uncertainty in a vector of size 2
  virtual vector<Location*> findposition(vector<double> distances, vector<Location*> lcns) = 0;
  virtual bool setparams(DistanceLocalizerParams* params) = 0;
};


struct DataSelectorParams : public Params{
  virtual DataSelectorParams* copy(void) = 0;
  //Other virtual functions inherited from Params
};

/// Template for algorithms that combine ToA and RSSI based positions
class DataSelector{
public:
  ///Pre:  both measurements are valid and identical except for the position and uncertainty
  ///Post: returns position and uncertainty from the appropriate measurement
  ///      arguments are unchanged
  ///Complexity: O(s) where s is the number of sensors
  virtual vector<Location*> combine(Measurement* toameas, Measurement* rssimeas) = 0;
  ///Configure new set of parameters, discarding the old set.
  virtual bool setparams(DataSelectorParams* params) = 0;
};


#endif
