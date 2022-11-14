/** solverintruder.h: Classes for tracking and displaying the locations of wireless transmitters
 * (the simulator has a similar set of classes for simulating said transmitters)
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

#ifndef NRL_SOLVER_INTRUDERS_H
#define NRL_SOLVER_INTRUDERS_H

#include <deque>

#include "nrlintruder.h"
#include "measurement.h"
#include "timestamp.h"

using namespace std;


//FIXME: Move to parameter struct
#define MAX_INTRUDER_COASTS	3


class Intruder : public NRLIntruder{ // : public Printable{
  int wascoasted; //Number of times this intruder was coasted
  Location uncertainty;
public:
  Intruder(Measurement* meas, NRLApp* app);
  ~Intruder(void);
  bool update(std::deque<Measurement*>* update); //Boolean says if it should be deleted
};


///A class that the Frame uses to manage its Intruders
class TheIntruders : public NRLTheIntruders{
private:
  std::vector<Intruder*> intruders;
  
public:
  TheIntruders(NRLApp* app);
  void updateAll(std::deque<Measurement*>* update);
};

#endif
