/** loglinearmodel.cpp: Log-Linear path loss model for RSSI distance calculation
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
#include "loglinearmodel.h"
#include <iostream>
#include <cmath>

using namespace std;

/* -------- LogLinearModelParams implementation -------- */

LogLinearModelParams* LogLinearModelParams::copy(void){
  LogLinearModelParams* p = new LogLinearModelParams();
  p->lossfactor = lossfactor; //floating point number is easily copied
  return p;
}


void LogLinearModelParams::printcsv(ostream* fs){
  *fs << lossfactor;
}


void LogLinearModelParams::print(ostream* fs){
  *fs << "/---Log-Linear Model Parameters---\\" << endl;
  *fs << "Path Loss Gradient: " << lossfactor << endl;
  *fs << "\\---------------------------------/" << endl;
}


/* -------- LogLinearModel implementation -------- */

double LogLinearModel::rssi_to_dist(float rssi){
  
  if (!calibration_complete)
    cout << "Warning! Converting RSSI to Distance without a known RSSI at 1 meter!" << endl;
  
  double temp = (rssi - rssi_1m) / (-10 * alpha);
  return (double) (pow(10, temp) + 1);
}


vector<double> LogLinearModel::rssi_to_dist(Measurement* meas){
  vector<double> distlist;
  distlist.reserve(meas->pktarray->size());
  int a;
  for (a = 0; a < meas->pktarray->size(); a++){
    Packet* pkt = (*(meas->pktarray))[a];
    double dist;
    if (pkt->isCalibrationPkt()){
      //Use packet for calibration
      cout << "RSSI Using packet for calibration" << endl;
      setknownrssi(pkt->getrssi(), pkt->getcalibdist());
      dist = pkt->getcalibdist();
    } else {
      //Typical case, not a calibration packet
      dist = rssi_to_dist(pkt->getrssi());
    }
    distlist.push_back(dist);
  }
  return distlist;
}


float LogLinearModel::dist_to_rssi(double dist){
  
  if (!calibration_complete)
    cout << "Warning! Converting Distance to RSSI without a known RSSI at 1 meter!" << endl;
  
  return -10 * alpha * log10(dist - 1) + rssi_1m;
}


LogLinearModel::LogLinearModel(LogLinearModelParams* params){
  calibration_complete = false;
  rssi_1m = DEFAULT_RSSI_1M;
  if (!setparams(params))
    throw "Unable to create LogLinearModel";
}


bool LogLinearModel::setparams(PathLossModelParams* params){
  if (params->getType() != PT_LOGLINEAR){
    cout << "Warning! LogLinearModel setparams given struct of incorrect type" << endl;
    return false;
  }
  alpha = ((LogLinearModelParams*) params)->lossfactor;
  if (alpha <= 0)
    throw "Invalid alpha for path loss model\n";
  delete params;
  return true;
}
