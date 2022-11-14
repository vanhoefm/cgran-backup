/** nrlintruder.h: Common features and functions for both simulator and solver Intruder classes
 * (most of these relate to how they are displayed on the GUI)
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
#ifndef NRL_INTRUDER_H
#define NRL_INTRUDER_H

#include <vector>
#include <iostream> //For debugging print statements

#include <wx/statbmp.h>
#include <wx/bmpbuttn.h>
#include <wx/gdicmn.h> //for wxpoint

#include "templates.h"
#include "location.h"
#include "timestamp.h"
#include "app.h"
#include "macaddr.h"

using namespace std;

//C-style global function declaration!
//Both Intruders and Frames need this helper function
//It is here because both Intruders and Frames include this header (Frames do so indirectly)
wxPoint computeItemPosition(Location* loc, int size_r, int size_c);


class NRLIntruder{
  friend class NRLTheIntruders;
  friend class TheIntruders;
private:
  NRLApp* theapp;
  static wxBitmap* global_pic_n;
  static wxBitmap* global_pic_ne;
  static wxBitmap* global_pic_e;
  static wxBitmap* global_pic_se;
  static wxBitmap* global_pic_s;
  static wxBitmap* global_pic_sw;
  static wxBitmap* global_pic_w;
  static wxBitmap* global_pic_nw;
  static wxBitmap* global_pic_stop;
  wxBitmapButton* button;
protected:
  int buttonid;
  MacAddr* mac;
  Location lcn;
  Location velocity;
  Timestamp time_first_appeared;
  //Inherited from the oldest ToA of the packets
  //that constitute the first measurement used to make this Intruder 
public:
  NRLIntruder(MacAddr* macaddr, Location* loc, NRLApp* app);
  void makeVisible(int id);
  ~NRLIntruder(void);
  void updateIcon(void);
};


///A class that the Frame uses to manage its Intruders
class NRLTheIntruders{
protected:
  NRLApp* theapp;
  
public:
  NRLTheIntruders(NRLApp* app);
  ~NRLTheIntruders(void);
};

#endif
