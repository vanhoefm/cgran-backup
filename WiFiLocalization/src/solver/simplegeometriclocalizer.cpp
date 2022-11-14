/** simplegeometriclocalizer.h: A simplistic position-finding algorithm
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
#include "simplegeometriclocalizer.h"
#include <math.h> //Note: C99 is needed for isinf() and isnan()
#include <cstdlib>

using namespace std;

/* --- SimpleGeometricLocalizerParams implementation --- */

SimpleGeometricLocalizerParams* SimpleGeometricLocalizerParams::copy(void){
  SimpleGeometricLocalizerParams* p = new SimpleGeometricLocalizerParams();
  p->zoffset = zoffset;
  return p;
}


void SimpleGeometricLocalizerParams::print(ostream* fs){
  *fs << "/-Simple DToA Geometric Solver Parameters-\\" << endl;
  *fs << "Known Z offset of intruder: " << zoffset << " meters" << endl;
  *fs << "\\-----------------------------------------/" << endl;
}


void SimpleGeometricLocalizerParams::printcsv(ostream* fs){
  *fs << zoffset;
}


/* --- SimpleGeometricLocalizer implementation --- */

SimpleGeometricLocalizer::SimpleGeometricLocalizer(SimpleGeometricLocalizerParams* params){
  if (!setparams(params))
    throw "Invalid SimpleGeometricLocalizer Params\n";
}

bool SimpleGeometricLocalizer::setparams(DistanceLocalizerParams* params){
  if (params->getType() != PT_SGEOLOC){
    cout << "Warning! Invalid SimpleGeometricLocalizer Params type" << endl;
    return false;
  }
  zoffset = ((SimpleGeometricLocalizerParams*) params)->zoffset;
  delete params;
  return true;
}


//Helper function for returning vector of NULL (failed localization)
vector<Location*> SimpleGeometricLocalizer::returnerror(void){
  vector<Location*> retvec;
  retvec.reserve(2);
  retvec.push_back(NULL); //Position
  retvec.push_back(NULL); //Uncertainty
  return retvec;
}


//O(s^2), except s is required to be 3.
vector<Location*> SimpleGeometricLocalizer::findposition(vector<double> distances,
						     vector<Location*> lcns){
  
  if (distances.size() != 3)
    throw "SimpleGeometricLocalizer: Vector of distances not of size 3!";
  if (lcns.size() != 3)
    throw "SimpleGeometricLocalizer: Vector of locations not of size 3!";
  
  double dist1, dist2, dist3;
  Location* first, *second, *third;
  
  // Figure out which of the Locations is (0, 0, 0)
  // This is required by the algorithm
  Location node0; //Defaults to (0, 0, 0)
  bool havefirst = false;
  bool havesecond = false;
  bool havethird = false;
  int a;
  for (a = 0; a < 3; a++){ //O(s)
    if (node0.isSame(lcns[a])){
      //Found node1 (0, 0, 0)
      if (havefirst){
	cout << "Warning! Duplicate sensors at (0, 0, 0)" << endl;
	return returnerror();
      }
      first = lcns[a];
      dist1 = distances[a];
      havefirst = true;
    } else if (lcns[a]->y == 0){
      //Found node2 (x, 0, 0)
      if (havesecond){
	cout << "Warning! Duplicate sensors at (x, 0, 0)" << endl;
	//This is what we did in AK, so we'll need a different 1/0 proof algorithm.
	return returnerror();
      }
      second = lcns[a];
      dist2 = distances[a];
      havesecond = true;
    } else {
      //Found node3 (x, y, 0)
      if (havethird){
	cout << "Warning! Duplicate sensors at (x, y, 0)" << endl;
	return returnerror();
      }
      third = lcns[a];
      dist3 = distances[a];
      havethird = true;
    }
  }
  
  //At this point, all three points have been filled
  //One of the three warnings would have been encountered otherwise
  //since there are 3 cases, 3 loop iterations, 
  //and each case will throw a warning if it occurs multiple times
  
  double dist_thresh = 1000; // * 1000; //1 km threshold
  
  //Sanity check distances. Throw warning and divide by 1000 if overrun.
  if ((fabs(dist1) > dist_thresh) || (fabs(dist2) > dist_thresh) || (fabs(dist3) > dist_thresh)){
    cout << "Warning: Distances are very large and fail sanity check." << endl;
    cout << dist1 << ", " << dist2 << ", " << dist3 << endl;
    return returnerror();
  }
  
  cout << "Distances: " << dist1 << ", " << dist2 << ", " << dist3 << endl;
  
  //Correct the distances using height offset
  //Note: Squared distances are retained to make computations simpler
  float d1sq = dist1*dist1 - zoffset*zoffset;
  float d2sq = dist2*dist2 - zoffset*zoffset;
  float d3sq = dist3*dist3 - zoffset*zoffset;
  
  //Now for some math!
  float temp1 = second->y;
  float x = (d1sq - d2sq + temp1*temp1) / (2 * temp1);
  temp1 = third->x;
  float temp2 = third->y;
  float temp3 = d1sq - d3sq + temp1*temp1 + temp2*temp2;
  float temp4 = 2 * temp1 * x;
  float y = (temp3 - temp4) / (2 * temp2);
  
  cout << "Position: " << "{ " << x << ", " << y << " }" << endl;
  
  if ( isnan(x) || isnan(y) ){
    cout << "Warning! Distances produced are not numbers (NaN)." << endl;
    cout << "Check for packets with duplicate ToA times" << endl;
    return returnerror();
  }
  if ( isinf(x) || isinf(y) ){
    cout << "Warning! Distances produced are infinite (-inf or inf)." << endl;
    cout << "Check for intruders directly at sensors" << endl;
    return returnerror();
  }
    
  Location* position = new Location(x, y, zoffset);
  vector<Location*> retvec;
  retvec.reserve(2);
  retvec.push_back(position);
  retvec.push_back(NULL); //This algorithm does not calculate uncertainty
  
  cout << "Info: Returning position data of:";
  position->print(&cout);
  
  return retvec;
}
