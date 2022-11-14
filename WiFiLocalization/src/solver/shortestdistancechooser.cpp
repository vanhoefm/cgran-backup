/** shortestdistancechooser.h: Chooses between ToA and RSSI results based on which position is closer to the sensors
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
#include "shortestdistancechooser.h"

using namespace std;


/* ----- ShortestDistanceChooserParams implementation ----- */

ShortestDistanceChooserParams* ShortestDistanceChooserParams::copy(void){
  ShortestDistanceChooserParams* p = new ShortestDistanceChooserParams();
  return p;
}


void ShortestDistanceChooserParams::print(ostream* fs){
  *fs << "Using Shortest Distance Chooser" << endl;
}

void ShortestDistanceChooserParams::printcsv(ostream* fs){
  //No parameters to print!
}


/* ----- ShortestDistanceChooser implementation ----- */

ShortestDistanceChooser::ShortestDistanceChooser(ShortestDistanceChooserParams* params){
  setparams(params);
}


bool ShortestDistanceChooser::setparams(DataSelectorParams* params){
  delete params;
}


vector<Location*> ShortestDistanceChooser::combine(
		  Measurement* toameas, Measurement* rssimeas){
  
  //NOTE: This method does not consider the uncertainty (not that our localizer calculates one...)
  
  if (toameas->position == NULL)
    throw "ToA meas hasn't beeen localized! Unable to select one or the other";
  if (rssimeas->position == NULL)
    throw "RSSI meas hasn't beeen localized! Unable to select one or the other";
  
  int a;
  vector<Location*> locvec;
  locvec.reserve(toameas->pktarray->size());
  //Obtain copies of the location from each packet in the Measurements
  for (a = 0; a < toameas->pktarray->size(); a++)
    locvec.push_back( (*(toameas->pktarray))[a]->getloc() );
  
  double rssimaxdist = 0;
  double toamaxdist = 0;
  //Calculate maximum distance from sensor for ToA
  for (a = 0; a < toameas->pktarray->size(); a++){
    double temp = locvec[a]->distanceToDest(toameas->position);
    if (temp > toamaxdist)
      toamaxdist = temp;
  }
  
  //Calculate maximum distance from sensor for RSSI
  for (a = 0; a < toameas->pktarray->size(); a++){
    double temp = locvec[a]->distanceToDest(rssimeas->position);
    if (temp > rssimaxdist)
      rssimaxdist = temp;
  }
  
  vector<Location*> ret;
  ret.reserve(2);
  if (rssimaxdist > toamaxdist){
    //Use ToA
    ret[0] = toameas->position;
    ret[1] = toameas->uncertainty;
  } else {
    //Use RSSI
    ret[0] = rssimeas->position;
    ret[1] = rssimeas->uncertainty;
  }
  return ret;
}
