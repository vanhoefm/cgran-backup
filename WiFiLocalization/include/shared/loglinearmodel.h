/** loglinearmodel.h: Log-Linear path loss model for RSSI distance calculation
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
#ifndef LOG_LINEAR_MODEL_H
#define LOG_LINEAR_MODEL_H

#include "modeltemplates.h"
#include "measurement.h"
#include <vector>
#include <ostream>

using namespace std;

struct LogLinearModelParams : public PathLossModelParams{
  float lossfactor;
  
  //Params required functions
  virtual ParamsType getType(void) { return PT_LOGLINEAR; };
  //Printable required functions
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  virtual LogLinearModelParams* copy(void);
};

/// Simple path loss model. Assumes ideal exponential path loss.
class LogLinearModel : public PathLossModel{
  float alpha;
public:
  LogLinearModel(LogLinearModelParams* params);
  
  virtual vector<double> rssi_to_dist(Measurement* meas);
  virtual double rssi_to_dist(float rssi);
  virtual float dist_to_rssi(double dist);
  virtual bool setparams(PathLossModelParams* params);
};

#endif
