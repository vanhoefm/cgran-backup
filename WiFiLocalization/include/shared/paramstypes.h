/** paramstypes.h: Enumerated values used for typechecking parameter structs
 * Since every single Params-derived class must have one of these, the file might get long
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
#ifndef PARAMS_TYPES_H
#define PARAMS_TYPES_H

using namespace std;

//Enumerated types 
enum ParamsType{
  PT_AGG, //Aggregator thread
  PT_SOLV, //Solver thread
  PT_5STEP, //5-step localization framework
  PT_LOGLINEAR, //LogLinearPathLoss
  PT_LIGHTSPEED, //LightspeedToA
  PT_SDCHOOSE, //ShortestDistanceChooser
  PT_SGEOLOC, //Simple Geometric Localizer
  
  PT_OTHER //Default, used for errors (not currently used)
};

#endif
