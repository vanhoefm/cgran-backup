/** nrlintruder.cpp: Common features and functions for both simulator and solver Intruder classes
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
#include "nrlintruder.h"
#include "gui_sizes.h"

using namespace std;

//Initialize globals to NULL. This gets redefined later
wxBitmap* NRLIntruder::global_pic_n = NULL;
wxBitmap* NRLIntruder::global_pic_ne = NULL;
wxBitmap* NRLIntruder::global_pic_e = NULL;
wxBitmap* NRLIntruder::global_pic_se = NULL;
wxBitmap* NRLIntruder::global_pic_s = NULL;
wxBitmap* NRLIntruder::global_pic_sw = NULL;
wxBitmap* NRLIntruder::global_pic_w = NULL;
wxBitmap* NRLIntruder::global_pic_nw = NULL;
wxBitmap* NRLIntruder::global_pic_stop = NULL;


NRLIntruder::NRLIntruder(MacAddr* macaddr, Location* loc, NRLApp* app){
  mac = macaddr;
  lcn = *loc;
  delete loc;
  velocity = Location(0, 0, 0);
  theapp = app;
  
  //Show it as a bitmap? This should probably be done in another function
  button = NULL;
}


NRLIntruder::~NRLIntruder(void){
  delete button; //Button will become invisible next time frame calls Show(true)
}


void NRLIntruder::makeVisible(int id){
  buttonid = id;
  delete button; //recently added
  button = new wxBitmapButton(theapp->panel, buttonid, *global_pic_stop,
			      computeItemPosition(&lcn, 70, 70),
			      wxSize(70, 70), 0);
}


//Get intruder list set up with nothing in it
NRLTheIntruders::NRLTheIntruders(NRLApp* app){
  
  //Load global intruder bitmaps
  NRLIntruder::global_pic_n = new wxBitmap();
  NRLIntruder::global_pic_ne = new wxBitmap();
  NRLIntruder::global_pic_e = new wxBitmap();
  NRLIntruder::global_pic_se = new wxBitmap();
  NRLIntruder::global_pic_s = new wxBitmap();
  NRLIntruder::global_pic_sw = new wxBitmap();
  NRLIntruder::global_pic_w = new wxBitmap();
  NRLIntruder::global_pic_nw = new wxBitmap();
  NRLIntruder::global_pic_stop = new wxBitmap();
  NRLIntruder::global_pic_n->LoadFile(_("images/intruder_n.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_ne->LoadFile(_("images/intruder_ne.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_e->LoadFile(_("images/intruder_e.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_se->LoadFile(_("images/intruder_se.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_s->LoadFile(_("images/intruder_s.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_sw->LoadFile(_("images/intruder_sw.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_w->LoadFile(_("images/intruder_w.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_nw->LoadFile(_("images/intruder_nw.bmp"), wxBITMAP_TYPE_BMP);
  NRLIntruder::global_pic_stop->LoadFile(_("images/intruder_stopped.bmp"), wxBITMAP_TYPE_BMP);
  //Store location of "TheAPP"
  theapp = app;
}


NRLTheIntruders::~NRLTheIntruders(void){
}


///Helper function for converting Locations into a position on the screen
//Positional accuracy is valued more than ensuring everything will fit on the screen!
//Note: (0, 0) is mapped to the top left
wxPoint computeItemPosition(Location* loc, int size_r, int size_c){
  int r, c;
  if (loc == NULL)
    throw "Cannot compute item position using NULL location";
  /* This is left over from when position was arbitrary units (not cm)
  r = 100 + (int) (loc->x * 50) - size_r/2;
  c = 100 + (int) (loc->y * 50) - size_c/2;
  */
  r = meters_to_x_pixels(loc->x);
  c = meters_to_y_pixels(loc->y);
  return wxPoint(r, c);
} 


//Helper function for updating the button image
//FIXME: Use a method that performs better 
//(no destructing/constructing bitmap containers)
//If that doesn't work, keep an old copy of "picture" and update only if
//the new one is different.
void NRLIntruder::updateIcon(void){
  //Update the image we are using as well as where our button is
  wxBitmap* picture;
  switch (velocity.indicateDirection()){
    
    case VEL_N:
      picture = global_pic_n;
      break;
    case VEL_NE:
      picture = global_pic_ne;
      break;
    case VEL_E:
      picture = global_pic_e;
      break;
    case VEL_SE:
      picture = global_pic_se;
      break;
    case VEL_S:
      picture = global_pic_s;
      break;
    case VEL_SW:
      picture = global_pic_sw;
      break;
    case VEL_W:
      picture = global_pic_w;
      break;
    case VEL_NW:
      picture = global_pic_nw;
      break;
    case VEL_STOP:
      picture = global_pic_stop;
      break;
  }
  
  delete button;
  button = new wxBitmapButton(theapp->panel, buttonid, *picture,
			      computeItemPosition(&lcn, 70, 70),
			      wxSize(70, 70), 0);
  return;
}
