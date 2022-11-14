/** solverframe.cpp: Application frame class for solver GUI
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
#include "solverframe.h"
#include "packet.h"
#include "measurement.h"
//Needed because we create these solver algorithms by default (in Timer, perhaps elsewhere)
#include "frameworks.h"
#include "lightspeedtoa.h"
#include "loglinearmodel.h"
#include "simplegeometriclocalizer.h"
#include "shortestdistancechooser.h"
//We also need speed of light in a vacuum, in m/ns
#define C_NSEC 2.99792458

#include <fstream>

using namespace std;

// IDs for the controls and the menu commands
enum {
    // menu items
    NRLID_TEST = 1000,
    // socket items
    SERVER_START,
    SERVER_ID,
    SOCKET_ID,
    NRLID_TIMER,
    NRLID_CALIB,
    
    //this MUST be the last enum!
    //this plus 0 to 10 is used for buttons!
    NRLID_MAX
};


//Associate GUI events with their handlers
BEGIN_EVENT_TABLE(NRLFrame, wxFrame)
  EVT_MENU(NRLID_TEST, NRLFrame::OnLogData)
  EVT_MENU(SERVER_START, NRLFrame::OnServerStart)
  EVT_MENU(wxID_ABOUT, NRLFrame::OnAbout)
  EVT_MENU(wxID_EXIT, NRLFrame::OnQuit)
  EVT_MENU(NRLID_CALIB, NRLFrame::OnCalib)
  EVT_TIMER(NRLID_TIMER, NRLFrame::OnTimerEvent)
  EVT_SOCKET(SERVER_ID,  NRLFrame::OnServerEvent)
  EVT_SOCKET(SOCKET_ID,  NRLFrame::OnSocketEvent)
  
  //Massive list of button events (not all will be used at once)
  EVT_BUTTON(NRLID_MAX, NRLFrame::onButtonClick0)
  EVT_BUTTON(NRLID_MAX + 1, NRLFrame::onButtonClick1)
  EVT_BUTTON(NRLID_MAX + 2, NRLFrame::onButtonClick2)
  EVT_BUTTON(NRLID_MAX + 3, NRLFrame::onButtonClick3)
  EVT_BUTTON(NRLID_MAX + 4, NRLFrame::onButtonClick4)
  EVT_BUTTON(NRLID_MAX + 5, NRLFrame::onButtonClick5)
  EVT_BUTTON(NRLID_MAX + 6, NRLFrame::onButtonClick6)
  EVT_BUTTON(NRLID_MAX + 7, NRLFrame::onButtonClick7)
  EVT_BUTTON(NRLID_MAX + 8, NRLFrame::onButtonClick8)
  EVT_BUTTON(NRLID_MAX + 9, NRLFrame::onButtonClick9)
  EVT_BUTTON(NRLID_MAX + 10, NRLFrame::onButtonClick10)
END_EVENT_TABLE()


void NRLFrame::OnAbout(wxCommandEvent& event){
  wxString msg;
  msg.Printf(wxT("This program is relying upon %s"), wxVERSION_STRING);
  wxMessageBox(msg, wxT("About NRLSolver"), wxOK | wxICON_INFORMATION, this);
}

void NRLFrame::OnQuit(wxCommandEvent& event){
  cout << "Quitting" << endl;
  try {
    //Close all open connections
    int a;
    while (connections.size() > 0){
      cout << "Destroying connection" << endl;
      connections.back()->Destroy();
      connections.pop_back();
    }
    m_server->Destroy();
    delete m_server;
    
    delete intruders;
    delete aggsolvmgr;
    delete logger; //Must be the last thread to be deleted
  } catch (const char* text) {
    cout << "Fatal error while closing: " << text << endl;
  }
  //Destroy the frame
  Close();
}

NRLFrame::NRLFrame(const wxString& title, NRLApp* app)
	: wxFrame(NULL, wxID_ANY, title){
	  
  // Set up internal data
  theapp = app;
  m_server_ok = false;
  //Set up button pool
  buttonsavailable.reserve(11);
  int a;
  for (a = 0; a < 10; a++)
    buttonsavailable[a] = true;

  //Get the threads set up
  connections.reserve(NUM_SENSORS_DESIRED);
  try {
    logger = new TheLoggerThread();
    aggsolvmgr = new AggSolvMgr(); //Note: this by itself doesn't start the thread, though the buffer is created.
    
  } catch (const char* text){
    cout << "Fatal error: " << text << endl;
    //Close frame
    Close();
  }
  
  // Set up GUI //
  
  //Set icon (implement later)
  
  //Create file menu
  wxMenu *fileMenu = new wxMenu();
  fileMenu->Append(SERVER_START, _("&Start\tAlt-S"), _("Start the solver"));
  fileMenu->Append(wxID_EXIT, wxT("&Exit\tAlt-X"), wxT("Quit this program"));
  
  //Create solver menu
  wxMenu *solverMenu = new wxMenu();
  solverMenu->Append(NRLID_CALIB, _("&Calibrate MAC\tAlt-M"),
		     _("Specify a new MAC address for calibration"));
  
  //Put "about" item in help menu
  wxMenu *helpMenu = new wxMenu();
  helpMenu->Append(wxID_ABOUT, wxT("&About...\tF1"),
		   wxT("Show about dialog"));
  
  //Debug item menu, for use when developing the solver program
  wxMenu* debugMenu = new wxMenu();
  debugMenu->Append(NRLID_TEST, wxT("&SendToLogFiles\tAlt-D"), 
		    wxT("Send Test Data To Logfiles"));
  
  //Now append the freshly created menus to the menu bar
  wxMenuBar *menuBar = new wxMenuBar();
  menuBar->Append(fileMenu, wxT("&File"));
  menuBar->Append(solverMenu, wxT("&Solver"));
  menuBar->Append(helpMenu, wxT("&Help"));
  menuBar->Append(debugMenu, wxT("&Debug"));
  
  //Attach menu bar to the frame
  SetMenuBar(menuBar);
  
}


void NRLFrame::OnCalib(wxCommandEvent& event){
  if (calibdialog != NULL){
    cout << "A dialog is already open!" << endl;
    calibdialog->Show(); //make it pop back to the front of the screen
    return;
  }
  calibloc = NULL;
  calibmac = NULL;
  //Launch the dialog box
  calibdialog = new CalibDialog();
  cout << "Creating calibration dialog" << endl;
  calibdialog->Create(this, theapp, &calibdialog, &calibmac, &calibloc);
  calibdialog->Show();
}


void NRLFrame::OnLogData(wxCommandEvent& event){
  try {
    //Timestamp* ts = new Timestamp((double) 0);
    //Location* lcn = new Location(3, 3, 3);
    //Timestamp* ts2 = new Timestamp((double) 0);
    
    /*
    char buf[256];
    strcpy(buf, "I'm a little teapot\n");
    
    float rssi = -35;
    
    Packet* pkt = new Packet(buf, ts, rssi, lcn, ts2);
    
    delete ts;
    delete ts2;
    delete lcn;
    */
    
    sensorlist.PrintALL();    
    
    //logger->log(lcn, LT_OTHER);
    //printf("OK1\n");
    //delete lcn; //We keep ownership of the Printable being logged
  } catch (const char* text){
    cout << "Fatal error: " << text << endl;
  }
}


//Timer every half second to make sure the event loop runs while MessageBox is displayed
//Updates intruders
void NRLFrame::OnTimerEvent(wxTimerEvent& event){
  try {
    int a;
    
    ///Retrieve calibration MAC addresses deposited by the dialog.
    //FIXME: Only one is allowed at the time, so there's no need to store it in a vector.
    
    //FIXME: insert utilities for adding additional sensors (after the first 3) and notifying aggsolvmgr thread here
    
    ///Toggle aggregator and solver on and off depending on whether or not we have enough sensors connected.
    ///If there are 3+ sensors and aggsolvmgr isn't running yet, run it.
    if ((aggsolvmgr->isRunning() == false) && (sensorlist.size() >= 3)){ //FIXME: hardcoded minimum number of sensors 
      //BUG: this will reset all parameters to defaults if a sensor is lost and re-established.
      //cout << "About to make aggsolvmgr parameters" << endl;
      AggParams* aparam = new AggParams();
      //cout << "Done making aggparams, need to fill in the data" << endl;
      aparam->time_toler = new Timestamp((int64_t) 0, (double) 100); //FIXME: Hardcoded 100ns time tolerance between matches
      aparam->time_to_wait = new Timestamp((int64_t) 1, (double) 0); //FIXME: Hardcoded 1 second delay from when packet was received at sensor and when it expires
      aparam->num_sensors = 3; //FIXME: Hardcoded 3 sensors.
      //cout << "About to make paramswrapper" << endl;
      aggparams = new ParamsWrapper(aparam);
      
      SolvParams* sparam = new SolvParams();
      cout << "about to make solvparams" << endl;
      FiveStepFrameworkParams* p1 = new FiveStepFrameworkParams();
      sparam->fparam = p1;
      LightspeedToAParams* p2 = new LightspeedToAParams(); //WARNING We need to provide sensor locations!
      p1->tdc = p2;
      p2->c_nsec = C_NSEC; //FIXME: Hardcoded default to free space lightspeed
      //vector<Timestamp*> corrtime;
      //corrtime.push_back(new Timestamp(0, (double) 11)); //FIXME: hardcoded 11 ns correction factor
      //corrtime.push_back(new Timestamp(0, (double) 11)); //FIXME: hardcoded 11 ns correction factor
      //corrtime.push_back(new Timestamp(0, (double) 11)); //FIXME: hardcoded 11 ns correction factor
      //p2->corrtime = corrtime;
      //p2->corrloc = sensorlist.getLocations(); //NOTE: We don't know what order they appear in!
      p2->timecorr = new Timestamp(0, 11.0);  //FIXME: hardcoded 11 ns correction factor
      LogLinearModelParams* p3 = new LogLinearModelParams();
      p1->rdc = p3;
      p3->lossfactor = 7; //FIXME: Hardcoded default to alpha = 7
      SimpleGeometricLocalizerParams* p4 = new SimpleGeometricLocalizerParams();
      p1->dl = p4;
      p4->zoffset = 0; //FIXME: Hardcoded default to Z offset of 0 meters
      p1->se = new ShortestDistanceChooserParams();
            
      solverparams = new ParamsWrapper(sparam);
      //cout << "About to make aggregator thread" << endl;
      
      sparam->fparam->getType();
      
      aggsolvmgr->startThreads(aggparams, solverparams);
      //cout << "Made aggregator thread" << endl;
      aggparams->print(&cout);
      
    } else if (aggsolvmgr->isRunning() && sensorlist.size() < 3){ //FIXME hardcoded minimum # of sensors
      cout << "Stopping aggregator and solver: not enough sensors." << endl;
      aggsolvmgr->stopThreads();
    }
    
    ///Obtain measurements (updates) from solver thread
    //cout << "Grabbing solver results" << endl;
    vector<Measurement*> measvec = aggsolvmgr->grabSolverResults();
    
    //cout << "Done grabbing results, moving on to print measurements" << endl;
    
    ///Update our intruders using the measurements received
    //FIXME: this just prints them instead
    for (a = 0; a < measvec.size(); a++)
      measvec[a]->print(&cout);
    
    //cout << "Done printing measurements" << endl;
    
  } catch (const char* text){
    cout << "Error updating: " << text << endl;
  } catch (...){
    cout << "Error updating\n";
  }
}


/*----------Socket Event handlers------------*/

//From example file
void NRLFrame::OnServerStart(wxCommandEvent& WXUNUSED(event))
{
  try {
    if (m_server_ok){
      wxMessageBox(_("The solver is already running!\n"), _("Warning!"));
      cout << "Warning: Socket was already open\n";
      return;
    }
    
    //FIXME:Temporarily moved
    intruders = new TheIntruders(theapp);
    
    
    // Create the address - defaults to localhost
    wxIPV4address addr;
    addr.Service(3000);

    // Create the socket, we maintain a class pointer so we can shut it down
    m_server = new wxSocketServer(addr);

    // We use Ok() here to see if the server is really listening
    if (! m_server->Ok())
    {
      wxMessageBox(_("Unable to start solver. Please wait several seconds and retry.\n"),
		  _("Warning!"));
      cout << "Socket was not created properly\n";
      return;
    }

    cout << "Solver listening for sensor connections" << endl; //Should be printed to GUI or MsgBox'd
    
    m_server_ok = true;
    
    // Setup the event handler and subscribe to connection events
    m_server->SetFlags(wxSOCKET_REUSEADDR);
    m_server->SetEventHandler(*this, SERVER_ID);
    m_server->SetNotify(wxSOCKET_CONNECTION_FLAG);
    m_server->Notify(true);
    
    //Add a timer to test features
    m_timer = new wxTimer(this, NRLID_TIMER);
    m_timer->Start(1000); //milliseconds
  } catch (const char* err){
    cout << "Error starting solver: " << err << endl;
  } catch (...) {
    cout << "Unknown error thrown while starting solver" << endl;
  }
}


void NRLFrame::OnServerEvent(wxSocketEvent& WXUNUSED(event))
{
  // Accept the new connection and get the socket pointer
  wxSocketBase* sock = m_server->Accept(false);
  int buttonid;
  try{
    buttonid = getNextButtonID();
  } catch (const char* err){
    cout << "Error getting button ID for sensor: " << err;
  }
  // Associate socket with a sensor. Note that it won't become visible until a test packet arrives.
  try {
    sensorlist.append(sock, buttonid);
  } catch (const char* err){
    cout << "Error appending new socket to sensorlist: " << err;
  } catch (...) {
    cout << "Unknown error appending new socket to sensorlist.";
  }
  // Tell the new socket how and where to process its events
  connections.push_back(sock);
  sock->SetEventHandler(*this, SOCKET_ID);
  sock->SetNotify(wxSOCKET_INPUT_FLAG | wxSOCKET_LOST_FLAG);
  sock->Notify(true);

  cout << "Accepted incoming connection.\n";
}


void NRLFrame::OnSocketEvent(wxSocketEvent& event)
{
  try{
    //cout << "Socket event" << endl;
    wxSocketBase* sock = event.GetSocket();
    // Process the event
    switch(event.GetSocketEvent())
    {
      case wxSOCKET_INPUT:
      { 
	// Read the data into a packet
	char buf[256];
	sock->Read(buf, sizeof(buf));
	Packet* pkt = new Packet(buf, 256);
	
	//Determine if it's a calibration packet and set flags accordingly
	vector<MacAddr*> macvec;
	vector<Location*> locvec;
	macvec.push_back(calibmac);
	locvec.push_back(calibloc);
	pkt->setCalibration(macvec, locvec);
	
	//pkt->print(&cout);
	//cout << "About to add item to aggregator" << endl;
	aggsolvmgr->add(pkt);
	/*
	//Check if this is a test packet used to say "This is where the sensor is"
	if (pkt.isATestPkt()){ //First time per sensor only
	  //Record sensor location if it doesn't already exist
	  Sensor* sens = sensorlist.getSensor(sock); //addLocation(sock, pkt.getSensorLoc());
	  bool locationadded = sens->setLocation(pkt.getloc());
	  //Make sensor visible
	  if (locationadded)
	    makeSensorVisible(sens);
	  else
	    cout << "Did not add location or button\n";
	  
	  cout << "Made it to isATestPkt\n";
	  
	} else { //Customary behavior
	  //Pass packet along to logger and aggsolvmgr
	  logger->log(&pkt, LT_RECV_PKT);
	  //aggsolvmgr->addToQueue(&pkt, new Timestamp());
	}
	*/
	break;
      }
      case wxSOCKET_LOST:
      {
	cout << "Sensor connection dropped\n";
	Sensor* sens = sensorlist.getSensor(sock);
	//makeSensorInvisible(sens);
	sensorlist.remove(sock);
	int a;
	for (a = 0; connections.size(); a++){
	  if (connections[a] == sock){
	    sock->Destroy();
	    connections.erase(connections.begin() + a);
	    Show(true);
	    return;
	  }
	}
	cout << "Warning! Connection being destroyed not found in list of open connections" << endl;
	sock->Destroy();
	break;
      }
    }
  } catch (const char* err){
    cout << "Error on socket event: " << err;
  } catch (...){
    cout << "Unknown error on socket event\n";
  }
}


void NRLFrame::makeSensorVisible(Sensor* sens){
  sens->button = new wxBitmapButton(theapp->panel, NRLID_MAX + sens->id, *(sens->picture),
				    computeItemPosition(sens->loc, 70, 75), wxSize(70, 75), 0);
  Show(true);
}


void NRLFrame::makeSensorInvisible(Sensor* sens){
  delete sens->button;
  sens->button = NULL;
  Show(true);
}


//Refresh frame using all intruders that are present
//O(n^2)
void NRLFrame::updateIntruders(std::deque<Measurement*>* update){
  try {
    intruders->updateAll(update);
    Show(true);
  delete update;
  } catch (const char* err){
    cout << "Error updating intruders: " << err;
  } catch (...){
    cout << "Unknown error updating intruders\n";
  }
}


/********* The many button events*********/

int NRLFrame::getNextButtonID(void){
  int a;
  for (a = 0; a < 11; a++){
    if (buttonsavailable[a]){
      buttonsavailable[a] = false;
      return a + NRLID_MAX;
    }
  }
  throw "Error: Ran out of buttons in the button pool!\n";
}


void NRLFrame::freeButtonID(int id){
  //Sanitize inputs against segfaults
  if (id < NRLID_MAX)
    throw "Button ID too small!\n";
  if (id > NRLID_MAX + 10)
    throw "Button ID too large!\n";
  
  if (buttonsavailable[id - NRLID_MAX])
    cout << "Warning: Button being freed was available" << id << endl;
  buttonsavailable[id - NRLID_MAX] = true;
}


//Helper function called by event handlers
void NRLFrame::onButtonClick(int buttonid){
  
  
  cout << "Button " << buttonid << " clicked\n";
   
}


void NRLFrame::onButtonClick0(wxCommandEvent& event){
  onButtonClick(0);
}

void NRLFrame::onButtonClick1(wxCommandEvent& event){
  onButtonClick(1);
}

void NRLFrame::onButtonClick2(wxCommandEvent& event){
  onButtonClick(2);
}

void NRLFrame::onButtonClick3(wxCommandEvent& event){
  onButtonClick(3);
}

void NRLFrame::onButtonClick4(wxCommandEvent& event){
  onButtonClick(4);
}

void NRLFrame::onButtonClick5(wxCommandEvent& event){
  onButtonClick(5);
}

void NRLFrame::onButtonClick6(wxCommandEvent& event){
  onButtonClick(6);
}

void NRLFrame::onButtonClick7(wxCommandEvent& event){
  onButtonClick(7);
}

void NRLFrame::onButtonClick8(wxCommandEvent& event){
  onButtonClick(8);
}

void NRLFrame::onButtonClick9(wxCommandEvent& event){
  onButtonClick(9);
}

void NRLFrame::onButtonClick10(wxCommandEvent& event){
  onButtonClick(10);
}


  
