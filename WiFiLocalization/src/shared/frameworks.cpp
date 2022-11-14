/** frameworks.h: specific localization frameworks and their parameter structs
 * Frameworks are broad techniques for localization.
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
#include "frameworks.h"
//The following are needed for the constructor
#include "switch.h"
#include "lightspeedtoa.h"
#include "loglinearmodel.h"
#if NRL_SWITCH_SOLVER
  //These only exist in the solver
  #include "shortestdistancechooser.h"
  #include "simplegeometriclocalizer.h"
#endif

using namespace std;

#if 0

/* -------- FourStepModelFramework implementation --------- */

void FourStepModelFramework::localize(Measurement* meas){
  //Make working copies of the measurement
  Measurement* rssimeas = meas->copy();
  //Localization process
  rl->localize(rssimeas);
  vector<double> distvec = tdc->toa_to_dist(meas);
  vector<Location*> tempresult = dl->findposition(distvec, meas->getLocations());
  meas->position = tempresult[0];
  meas->uncertainty = tempresult[1];
  tempresult = se->combine(meas, rssimeas);
  meas->position = tempresult[0];
  meas->uncertainty = tempresult[1];
  //Cleanup
  delete rssimeas;
}


/// Set the parameters for each algorithm contained in this framework
bool FourStepModelFramework::setparams(FourStepModelFrameworkParams* params){
  //BUG: It does not check to ensure you are passing it the correct type of params
  FourStepModelFrameworkParams* p = (FourStepModelFrameworkParams*) params;
  rl->setparams(p->rl);
  tdc->setparams(p->tdc);
  dl->setparams(p->dl);
  se->setparams(p->se);
  delete p;
}  

#endif

/* -------- FiveStepFrameworkParams implementation --------- */

FiveStepFrameworkParams* FiveStepFrameworkParams::copy(void){
  FiveStepFrameworkParams* p = new FiveStepFrameworkParams();
  p->tdc = tdc->copy();
  p->dl = dl->copy();
  p->rdc = rdc->copy();
  p->se = se->copy();
  return p;
}


void FiveStepFrameworkParams::printcsv(ostream* fs){
  tdc->printcsv(fs);
  *fs << ";";
  rdc->printcsv(fs);
  *fs << ";";
  dl->printcsv(fs);
  *fs << ";";
  se->printcsv(fs);
}


void FiveStepFrameworkParams::print(ostream* fs){
  *fs << "Framework: Five Step Model" << endl;
  tdc->print(fs);
  rdc->print(fs);
  dl->print(fs);
  se->print(fs);
}


/* -------- FiveStepFramework implementation --------- */

FiveStepFramework::FiveStepFramework(FiveStepFrameworkParams* params){
  cout << 0 << endl;
  if (params->getType() != PT_5STEP)
    throw "FiveStepFramework cannot be constructed";
  
  //FUTURE: These case statements might be shared by multiple frameworks,
    //so give the Framework parent class some helper functions
  //cout << "Creating RDC by type" << endl;
  switch (params->rdc->getType()){
    case PT_LOGLINEAR:
      rdc = new LogLinearModel( (LogLinearModelParams*) params->rdc );
      break;
    default:
      throw "Invalid RSSI Dist Calc in 5Step creation";
  }
  //cout << "Creating TDC by type" << endl;
  switch (params->tdc->getType()){
    case PT_LIGHTSPEED:
      tdc = new LightspeedToA( (LightspeedToAParams*) params->tdc );
      break;
    default:
      throw "Invalid ToA Dist Calc in 5Step creation";
  }
#if NRL_SWITCH_SOLVER
  //cout << "Creating DL by type" << endl;
  switch (params->dl->getType()){
    case PT_SGEOLOC:
      dl = new SimpleGeometricLocalizer( (SimpleGeometricLocalizerParams*) params->dl );
      break;
    default:
      throw "Invalid Localizer in 5Step creation";
  }
  //cout << "Creating SE by type" << endl;
  switch (params->se->getType()){
    case PT_SDCHOOSE:
      se = new ShortestDistanceChooser( (ShortestDistanceChooserParams*) params->se );
      break;
    default:
      throw "Invalid Chooser in 5Step creation";
  }
  //cout << "About to delete params" << endl;
  delete params;
#else
  //Shortest distance chooser and geometric localizer don't exist in the simulator
  throw "Cannot instantiate 5StepFrameworks outside the solver!";
#endif
}


void FiveStepFramework::localize(Measurement* meas){
  //cout << "FiveStepFramework localize called" << endl;
  //Make working copies of the measurement
  Measurement* toameas = meas->copy();
  //cout << "Toa meas copied" << endl;
  //Localization process for ToA
  vector<double> distvec = tdc->toa_to_dist(meas);
  //cout << "ToA Distances calculated" << endl;
  vector<Location*> measlocs = meas->getLocations();
  vector<Location*> tempresult = dl->findposition(distvec, measlocs);
  //cout << "ToA Positions calculated" << endl;
  toameas->position = tempresult[0];
  toameas->uncertainty = tempresult[1];
  //Localization process for RSSI
  distvec = rdc->rssi_to_dist(meas);
  //cout << "RSSI distances calculated" << endl;
  tempresult = dl->findposition(distvec, measlocs);
  //cout << "RSSI Positions calculated" << endl;
  meas->position = tempresult[0];
  meas->uncertainty = tempresult[1];
  
  //Combine results
  //First, check to see if one or the other position calculations failed.
  //If so, use the one that succeeded.
  bool toaworked = false;
  bool rssiworked = false;
  if (toameas->position != NULL)
    toaworked = true;
  if (meas->position != NULL)
    rssiworked = true;
  if (toaworked && rssiworked) {
    tempresult = se->combine(toameas, meas);
    cout << "ToA and RSSI results being combined" << endl;
    meas->position = tempresult[0];
    meas->uncertainty = tempresult[1];
  } else if (toaworked) {
    cout << "Warning: Forcing ToA-based position (RSSI position not available)" << endl;
    meas->position = toameas->position;
    meas->uncertainty = toameas->uncertainty;
  } else if (rssiworked) {
    cout << "Warning: Forcing RSSI-based position (ToA position not available)" << endl;
    //meas already has rssi position
  } else {
    throw "FiveStepModel::localize(): Both ToA and RSSI localization failed";
  }
  //cout << "Cleanup localization" << endl;
  //Cleanup
  delete toameas;
  int a;//Delete measlocs, a duplicate copy of the Measurement's locations
  for (a = 0; a < measlocs.size(); a++)
    delete measlocs[a];
}


/// Set the parameters for each algorithm contained in this framework
bool FiveStepFramework::setparams(LocalizationFrameworkParams* params){
  bool worked = true;
  cout << 0 << endl;
  if (params->getType() != PT_5STEP){
    cout << "Warning! FiveStepFramework parameters are incorrect type" << endl;
    return false;
  }
  FiveStepFrameworkParams* p = (FiveStepFrameworkParams*) params; 
  //cout << 1 << endl;
  worked = worked && rdc->setparams(p->rdc);
  //cout << 2 << endl;
  worked = worked && tdc->setparams(p->tdc);
  //cout << 3 << endl;
  worked = worked && dl->setparams(p->dl);
  //cout << 4 << endl;
  worked = worked && se->setparams(p->se);
  //cout << 5 << endl;
  delete p;
  //cout << 6 << endl;
  return worked;
}
