/** frameworks.h: specific localization frameworks and their parameter structs
 * Frameworks are broad techniques for localization. See the examples within.
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
#ifndef NRL_FRAMEWORKS_H
#define NRL_FRAMEWORKS_H

#include "measurement.h"
#include "timestamp.h"
#include "location.h"
#include "packet.h"
#include "params.h"
#include "basicmath.h"
#include <vector>
#include "modeltemplates.h"

using namespace std;


/* --------- Frameworks --------------- */
/// Frameworks indicate the overall structure of the solving algorithm.

#if 0
/// Template for all unified position calculation algorithms
/// Not used as of 0.1.0, but useful later for other kinds of algorithms
class UnifiedFramework: public LocalizationFramework{
public:
  virtual void localize(Measurement* meas) = 0;
  virtual bool setparams(ParamsWrapper* params) = 0;
  virtual bool isUnifiedFramework(void) {return true;};
  virtual bool isFourStepFramework(void) {return false;};
  virtual bool isFiveStepFramework(void) {return false;};
};
#endif


#if 0
/// "Four-Step Model" Framework
/// Not used in 0.1.0 but useful for algorithms that use RSSI mapping
/** 
 * 1) ToA algorithm finds distance using time
 * 2) General localization algorithm finds position using ToA distances
 * 3) RSSI algorithm finds position
 * 4) Selector combines both calculations into a single final answer
 */
class FourStepModelFramework: public LocalizationFramework{
protected:
  ToADistanceCalculator* tdc;
  UnifiedRSSILocalizer* rl;
  DistanceLocalizer* dl;
  DataSelector* se;
public:
  virtual void localize(Measurement* meas);
  virtual bool setparams(ParamsWrapper* params);
  virtual bool isUnifiedFramework(void) {return false;};
  virtual bool isFourStepFramework(void) {return true;};
  virtual bool isFiveStepFramework(void) {return false;};
};

struct FourStepModelFrameworkParams{
  //Data items
  ToADistanceCalculatorParams* tdc;
  UnifiedRSSILocalizerParams* rl;
  DistanceLocalizerParams* dl;
  DataSelectorParams* se;
  //Functions
  virtual LocalizationFramework* generateFramework(void) = 0;
};

#endif

/// "Five-Step" Framework
/// This is the one we actually use (as of 0.1.0)
/** 
 * 1) ToA algorithm finds distance using time
 * 2) RSSI algorithm finds distance using time
 * 3) General localization algorithm finds position using ToA distances
 * 4) General localization algorithm finds position using RSSI distances
 * 5) Selector combines both calculations into a single final answer
 */

struct FiveStepFrameworkParams : public LocalizationFrameworkParams{
  ToADistanceCalculatorParams* tdc;
  PathLossModelParams* rdc;
  DistanceLocalizerParams* dl;
  DataSelectorParams* se;
  
  //Params required functions
  virtual ParamsType getType(void) { return PT_5STEP; };
  //Printable required functions
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  virtual FiveStepFrameworkParams* copy(void);
};

class FiveStepFramework: public LocalizationFramework{
protected:
  ToADistanceCalculator* tdc;
  PathLossModel* rdc;
  //DistanceCorrector tcorr;
  //DistanceCorrector rcorr;
  DistanceLocalizer* dl;
  DataSelector* se;
public:
  FiveStepFramework(FiveStepFrameworkParams* params);
  
  virtual void localize(Measurement* meas);
  virtual bool setparams(LocalizationFrameworkParams* params);
  virtual bool isUnifiedFramework(void) {return false;};
  virtual bool isFourStepFramework(void) {return false;};
  virtual bool isFiveStepFramework(void) {return true;};
};

#endif
