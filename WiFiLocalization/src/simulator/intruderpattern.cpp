/** intruderpattern.cpp: "Patterns" for determining simulated intruder behavior
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

#include "intruderpattern.h"

using namespace std;

//Note: SquarePattern uses a step size of 0 when velocity = NULL
//(errors won't result)
//NOTE: This forms a diamond pattern (square rotated 45 degrees)
SquarePattern1::SquarePattern1(Location velocity, int numsteps, int64_t totalsteps){
  vel = velocity;
  steps = numsteps;
  
  //Use the data in velocity to determine where within the 4-step cycle the Intruder is
  if (vel.x >= 0 && vel.y > 0){
    stepstaken = 0;
    steps_before_death = totalsteps;
  } else if (vel.x > 0 && vel.y <= 0){
    stepstaken = steps;
    steps_before_death = totalsteps + steps;
  } else if (vel.x <= 0 && vel.y < 0){
    stepstaken = 2 * steps;
    steps_before_death = totalsteps + 2 * steps;
  } else if (vel.x < 0 && vel.y >= 0){
    stepstaken = 3 * steps;
    steps_before_death = totalsteps + 3 * steps;
  } else { //both 0
    steps = 0;
    stepstaken = 0;
    steps_before_death = totalsteps;
  }
}


SquarePattern1::~SquarePattern1(void){
  //delete vel;
}


//FIXME: This assumes the update rate is constant and always the same
Location* SquarePattern1::getNewVelocity(Timestamp* timealive){
  cout << "gettingNewVelocity" << endl;
  
  stepstaken++;
  
  if (steps == 0)
    return NULL; //No change in velocity: it stays in one place
  
  int tempsteps = stepstaken % (steps * 4);
  if (tempsteps == 0)
    vel.x = - vel.x;
  if (tempsteps == steps)
    vel.y = - vel.y;
  if (tempsteps == 2 * steps)
    vel.x = - vel.x;
  if (tempsteps == 3 * steps)
    vel.y = - vel.y;
  
  cout << "About to return updated velocity" << endl;
  if (tempsteps % steps == 0) //velocity needs to be updated
    return new Location(vel); //uses C++ built in copy constructor
  else
    return NULL; //Don't waste time updating it if the velocity hasn't changed
}


bool SquarePattern1::isIntruderDead(Timestamp* timealive){
  return steps_before_death <= stepstaken;
}
