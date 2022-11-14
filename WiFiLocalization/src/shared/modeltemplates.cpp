/** modeltemplates.cpp: implementations of common functions the model templates provide
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

#include "modeltemplates.h"

using namespace std;

/* -------- PathLossModel implementation --------- */

///Given any known distance/RSSI combination, set the RSSI at 1 meter
void PathLossModel::setknownrssi(float rssi, double dist){
  calibration_complete = true;
  rssi_1m = 0; //Needs to be 0 for dist_to_rssi
  rssi_1m = rssi - dist_to_rssi(dist);
} 



