/** intruderpattern.h: "Patterns" for determining simulated intruder behavior
 * Specific example patterns are derived from the generic "pattern" class
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

#ifndef NRL_INTRUDER_PATTERN_H
#define NRL_INTRUDER_PATTERN_H

#include "timestamp.h"
#include "location.h"
#include <stdint.h>

//Maximum legal value for a Time To Live (in seconds)
#define MAX_TIME_TO_LIVE	100

using namespace std;

//Note: These IDs must line up with the order that they appear in dialog boxes
enum NRLPatternID {
  ID_SquarePattern1 = 0
};


///Abstract class for all Intruder patterns
class IntruderPattern{
public:
  virtual Location* getNewVelocity(Timestamp* timealive) = 0;
  virtual bool isIntruderDead(Timestamp* timealive) = 0;
};


///A simple pattern that makes the intruder dance in a diamond shape
class SquarePattern1 : public IntruderPattern{
private:
  Location vel;
  int steps;
  int64_t stepstaken;
  int64_t steps_before_death;
public:
  SquarePattern1(Location velocity, int numsteps, int64_t totalsteps);
  ~SquarePattern1(void);
  
  virtual Location* getNewVelocity(Timestamp* timealive);
  virtual bool isIntruderDead(Timestamp* timealive);
};

///Maximum value allowed for "steps"
#define SQ1_MAX_NUM_STEPS	15

///Other patterns go here

#endif
  