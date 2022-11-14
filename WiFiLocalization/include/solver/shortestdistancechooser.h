/** shortestdistancechooser.h: Chooses between ToA and RSSI results based on which position is closer to the sensors
 * 
 * This algorithm assumes one of the two data streams is invalid.
 * Invalid data streams usually place the intruders very far from the sensors
 * When one stream is junk data and the other is good data, the good data will likely prevail
 * When both streams are equally valid (or junky) the resulting selection may vary.
 * Note: If one or both data streams is producing all 0's, it will be chosen
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
#ifndef SHORTEST_DISTANCE_CHOOSER_H
#define SHORTEST_DISTANCE_CHOOSER_H

#include "modeltemplates.h"

using namespace std;

struct ShortestDistanceChooserParams : public DataSelectorParams{
  virtual ShortestDistanceChooserParams* copy(void);
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  ParamsType getType(void) { return PT_SDCHOOSE; };
};


class ShortestDistanceChooser : public DataSelector{
public:
  ShortestDistanceChooser(ShortestDistanceChooserParams* params);
  virtual vector<Location*> combine(Measurement* toameas, Measurement* rssimeas);
  virtual bool setparams(DataSelectorParams* params);
};

#endif

