/** solverapp.cpp: Main wxWidgets application with GUI basics for NRLMQP localization solverapp 
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

#include "app.h"
#include "solverframe.h"
#include "bitutils.h"
#include "sensor.h"

using namespace std;

/* OnInit: Initialization function for the application
 * follows wxWidgets specifications
 * Return value: Whether or not the application should be started
 * @author Brian Shaw
 */
bool NRLApp::OnInit(){
  //Initialize all modules that don't use constructors
  initUtils();
  
  //Create application window
  NRLFrame *frame = new NRLFrame(wxT("NRLMQP Localization Solver"), this);
  
  //FIXME: Borrowed from WxBitmapButton example!
  panel = new wxPanel( frame, -1 ,wxPoint(0,100),wxSize(871, 934));
  
  //Load images
  wxBitmap backgroundbitmap;
  backgroundbitmap.LoadFile(_("images/background.bmp"), wxBITMAP_TYPE_BMP);
  Sensor::picture = new wxBitmap;
  Sensor::picture->LoadFile(_("images/sensor.bmp"), wxBITMAP_TYPE_BMP);
  
  //Load background
  wxStaticBitmap* backgroundbmp = new wxStaticBitmap(panel, wxID_STATIC, backgroundbitmap,
						     wxPoint(0, 0), wxSize(871, 934));
  
  //Show the frame
  frame->Show(true);
  SetTopWindow(frame);
  //Start event loop
  return true;
}

//Macros telling wxWidgets to use the NRLApp object as the application
IMPLEMENT_APP(NRLApp)
DECLARE_APP(NRLApp)

//Function that is called when application exits
int NRLApp::OnExit(){
  delete Sensor::picture;
  return 0;
}