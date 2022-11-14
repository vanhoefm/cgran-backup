/** nrlframe.cpp: Application frame class for GUI
 * Note: Two files by this name exist, one for the simulator,
 * one for the solver (this one is for the simulator)
 * 
 * @author Brian Shaw
 * @author Julian Smart
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

#include "simframe.h"
#include <fstream>

using namespace std;

// IDs for the controls and the menu commands
enum {
  CLIENT_OPEN = 1000,
  CLIENT_CLOSE,
  CLIENT_SEND_TEST,
  NRLID_QUIT,
  NRLID_SIMSTART,
  NRLID_SIMSTOP,
  NRLID_TIMER,
  NRLID_SOCKET,
  NRLID_NEWSENS,
  NRLID_NEWINTR,
  NRLID_WHATNEXT,
  NRLID_EASYSETUP,
  //this MUST be the last enum!
  //this plus 0 to 10 is used for buttons!
  NRLID_MAX
};


//Associate GUI events with their handlers
BEGIN_EVENT_TABLE(NRLFrame, wxFrame)
  EVT_MENU(NRLID_QUIT, NRLFrame::OnQuit)
  EVT_MENU(CLIENT_OPEN, NRLFrame::OnConnectToServer)
  
  EVT_MENU(CLIENT_SEND_TEST, NRLFrame::OnSendingTest)
  
  EVT_MENU(NRLID_SIMSTART, NRLFrame::OnSimStart)
  EVT_MENU(NRLID_SIMSTOP, NRLFrame::OnSimStop)
  EVT_MENU(wxID_ABOUT, NRLFrame::OnAbout)
  EVT_MENU(wxID_EXIT, NRLFrame::OnQuit)
  EVT_MENU(NRLID_NEWSENS, NRLFrame::OnNewSensor)
  EVT_MENU(NRLID_NEWINTR, NRLFrame::OnNewIntruder)
  EVT_MENU(NRLID_WHATNEXT, NRLFrame::OnWhatNext)
  EVT_MENU(NRLID_EASYSETUP, NRLFrame::OnEasySetup)
  EVT_TIMER(NRLID_TIMER, NRLFrame::OnTimerEvent)
  EVT_SOCKET(NRLID_SOCKET,  NRLFrame::OnSocketEvent)
  
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
  try {
    delete sender;
    //Close all open connections
    int a;
    while (connections.size() > 0){
      cout << "Destroying connection" << endl;
      connections.back()->Destroy();
      connections.pop_back();
    }
    delete intruders;
    delete logger; //Must be the last thread to be deleted
    
    delete generator; //Packet generator object (Not a thread)
    
  } catch (const char* text) {
    cout << "Fatal error while closing: " << text << endl;
  }
  //Destroy the frame
  Close();
}


NRLFrame::NRLFrame(const wxString& title, NRLApp* app)
	: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(871, 934), wxDEFAULT_FRAME_STYLE, _("NRLMQP Simulator")){
    
  // Set up internal data
  theapp = app;
  updates_enabled = false;
  newintruderdialog = NULL;
  newsensordialog = NULL;
  helpdialog = NULL;
  werecalibrating = false;
  
  //Set up button pool
  buttonsavailable.reserve(11);
  int a;
  for (a = 0; a < 10; a++)
    buttonsavailable[a] = true;
  
  //Set up sockets and threads
  cout << "About to build intruders" << endl;
  connections.reserve(NUM_SENSORS_DESIRED);
  intruders = new TheIntruders(theapp);
  cout << "About to build llmp" << endl;
  LogLinearModelParams* llmp = new LogLinearModelParams();
  llmp->lossfactor = 7; //FIXME: hardcoded default to alpha of 7
  PathLossModel* rmodel = new LogLinearModel(llmp);
  cout << "About to build ltp" << endl;
  LightspeedToAParams* ltp = new LightspeedToAParams();
  ltp->c_nsec = .299792458; //FIXME: hardcoded Free space light speed
  ltp->timecorr = new Timestamp(0, 11.0);//FIXME: Hardcoded 11ns processing delay
  tmodel = new LightspeedToA(ltp);
  cout << "About to build packetgen" << endl;
  generator = new PacketGen(rmodel, tmodel);
  //No need to delete param structs, they are now owned by the models.
  cout << "About to build threads" << endl;
  //Get the threads set up
  try {
    logger = new TheLoggerThread();
    // //Launch packet sending thread to handle send requests.
    // //Also creates an interface for passing it outbound packet data.
    sender = new PktSender();
  } catch (const char* text){
    cout << "Fatal error: " << text << endl;
    //Close frame
    Close();
  }
  
  // Set up GUI //
  
  //Set icon (implement later)

  // Make menus
  wxMenu *m_menuFile = new wxMenu();
  m_menuFile->Append(NRLID_SIMSTART, _("&Start Simulation\tAlt-S"), _("Start the simulation"));
  m_menuFile->Append(NRLID_SIMSTOP, _("&Pause Simulation\tAlt-E"), _("Pause the simulation"));
  m_menuFile->Append(wxID_EXIT, _("&Exit\tAlt-X"), _("Quit simulator"));

  wxMenu *m_menuSocket = new wxMenu();
  m_menuSocket->Append(NRLID_NEWSENS, _("&New Sensor\tAlt-N"), _("Add a new sensor to the simulation"));
  m_menuSocket->Append(NRLID_NEWINTR, _("&New Intruder\tAlt-I"), _("Add a new intruder to the simulation"));
  m_menuSocket->Append(CLIENT_OPEN, _("&Connect\tAlt-C"), _("Connect to solver"));
  m_menuSocket->Append(CLIENT_CLOSE, _("&Disconnect"), _("Disconnect from server")); //this may be unnecessary

  //Put "about" item in help menu
  wxMenu *helpMenu = new wxMenu();
  helpMenu->Append(wxID_ABOUT, _("&About...\tF1"), _("Show about dialog"));
  helpMenu->Append(NRLID_WHATNEXT, _("&What Next?\tAlt-H"), _("Figure out what to do next"));
  
  //Debug menu--for making simulator development easier (will not be part of a final deployment)
  wxMenu *debugMenu = new wxMenu();
  debugMenu->Append(NRLID_EASYSETUP, _("&Easy Setup\tAlt-D"), _("Create example test setup"));
  debugMenu->Append(CLIENT_SEND_TEST, _("&Send Info\tAlt-B"), _("Send some information to server")); 
  
  // Append menus to the menubar
  wxMenuBar *m_menuBar = new wxMenuBar();
  m_menuBar->Append(m_menuFile, _("&File"));
  m_menuBar->Append(m_menuSocket, _("&Simulation"));
  m_menuBar->Append(helpMenu, _("&Help"));
  m_menuBar->Append(debugMenu, _("&Debug"));
  //Attach menu bar to the frame
  SetMenuBar(m_menuBar);
  
  //Add a timer for updating things
  timer = new wxTimer(this, NRLID_TIMER);
  timer->Start(1000); //milliseconds

}


void NRLFrame::OnNewSensor(wxCommandEvent& event){
  if ((newsensordialog != NULL) || (newintruderdialog != NULL)){
    cout << "A dialog is already open!" << endl;
    if (newsensordialog != NULL)
      newsensordialog->Show(); //Make it pop back to the front of the screen
    else 
      newintruderdialog->Show();
    return;
  }
  
  if (connections.size() < 3){
    cout << "Please connect to the solver before creating a new sensor." << endl;
    //because the sensors have to be associated with open connections.
    return;
  }
  
  newsensorloc = NULL;
  
  //Launch the dialog box
  newsensordialog = new NewSensorDialog();
  cout << "Creating new sensor dialog\n";
  newsensordialog->Create(this, theapp, &newsensordialog, &newsensorloc);
  
  newsensordialog->Show();
}
  

void NRLFrame::OnNewIntruder(wxCommandEvent& event){
  if ((newintruderdialog != NULL) || (newsensordialog != NULL)){
    cout << "A dialog is already open!" << endl;
    if (newintruderdialog != NULL)
      newintruderdialog->Show(); //Make it pop back to the front of the screen
    else
      newsensordialog->Show();
    return;
  }
  
  newintruder = NULL;
  //Launch the dialog box
  newintruderdialog = new NewIntruderDialog();
  newintruderdialog->Create(this, theapp, &newintruderdialog, &newintruder);
  
  newintruderdialog->Show();
  //The new intruder is added to our list next time the Timer updates, regardless of whether or not the simulation started
}


///Create 3 sensors and an intruder for easy debugging
void NRLFrame::OnEasySetup(wxCommandEvent& event){
  cout << "Easy setup not implemented" << endl;
}


/// A help feature telling the user what they need to do next.
/// Creates a "helpdialog" (which goes away on its own when the dialog is closed)
void NRLFrame::OnWhatNext(wxCommandEvent& event){
  
  if (helpdialog != NULL){
    cout << "Help dialog is already open!" << endl;
    helpdialog->Show();
    return;
  }
  
  helpdialog = new NRLHelpDialog();
  
  ///Check the state of the program and launch an appropriate help dialog.
  if (connections.size() == 0)
    helpdialog->Create(&helpdialog, _("Connect to the localization solver via Simulation->Connect."), this);
  else if ( sensorlist.size() < 3 ) //FIXME hardcoded 3 sensor minimum
    helpdialog->Create(&helpdialog, _("Add some sensors via Simulation->Add Sensor."), this);
  else if ( !updates_enabled )
    helpdialog->Create(&helpdialog, _("Start the simulation via File->Start Simulation."), this);
  else
    helpdialog->Create(&helpdialog, _("Simulation is running. Try adding intruders via Simulation->Add Intruder\nYou can pause the simulation at any time via File->Pause."), this);
  
  helpdialog->Show();
}


void NRLFrame::OnSendingTest(wxCommandEvent& event){
  if (sender == NULL){
    cout << "Cannot test sender. Sender hasn't been created! Run the simulation please" << endl;
    return;
  }
  
  try {    
    MacAddr mac;
    bool err = mac.macBuilder(0xFEDCBA012300ull);
    if (err)
      throw "Arbitrary test MAC addresses shouldn't be invalid.";
    
    if (connections.size() < 3)
      throw "Need at least 3 connections to test sending";
    
    float c_nsec = .299792458; //Free space light speed in m/ns
    Timestamp sendtime;
    
    Location lcn(3, 4, 0);
    Location lcn2(0, 0, 0); //Intruder is at (3, 0, 0)
    Location lcn3(5, 0, 0);
    double dist1 = 4.0;
    double dist2 = 3.0;
    double dist3 = 2.0;
    
    Timestamp* toa1 = tmodel->dist_to_toa(dist1, &sendtime);
    Timestamp* toa2 = tmodel->dist_to_toa(dist2, &sendtime);
    Timestamp* toa3 = tmodel->dist_to_toa(dist3, &sendtime);
    
#ifdef USING_AUXTIME
    Packet* pkt = new Packet(&mac, toa1, (float) -95, &lcn, &sendtime);
    Packet* pkt2 = new Packet(&mac, toa2, (float) -85, &lcn2, &sendtime);
    Packet* pkt3 = new Packet(&mac, toa3, (float) -75, &lcn3, &sendtime);
#else
    Packet* pkt = new Packet(&mac, toa1, (float) -95, &lcn); 
    Packet* pkt2 = new Packet(&mac, toa2, (float) -85, &lcn2); 
    Packet* pkt3 = new Packet(&mac, toa3, (float) -75, &lcn3); 
#endif
    
    delete toa1;
    delete toa2;
    delete toa3;
    /*
    //Log the packet that we sent
    logger->log(pkt, LT_RECV_PKT);
    logger->log(pkt2, LT_RECV_PKT);
    logger->log(pkt3, LT_RECV_PKT);
    */
    
    //Send a copy of the test packet out on each link.
    cout << "Sending a test packet on " << connections.size() << " links" << endl;
    
    //int a;
    //for (a = 0; a < connections.size(); a++)
    //  sender->send(pkt, connections[a]); //Eats the packet
    //sender->send no longer consumes the packets!
    sender->send(pkt, connections[0]);
    sender->send(pkt2, connections[1]);
    sender->send(pkt3, connections[2]);
    
    delete pkt;
    delete pkt2;
    delete pkt3;
        
    //cout << "Done sending over link" << endl;
    
  } catch (const char* text){
    cout << "Error on sending test: " << text << endl;
  }
}


//Timer every half second to make sure the event loop runs while MessageBox is displayed
//TODO: USE THIS TO UPDATE THE INTRUDERS!
void NRLFrame::OnTimerEvent(wxTimerEvent& event){
 
  //Add new intruders, if any, when dialog isn't open
  if ((newsensorloc != NULL) && (newsensordialog == NULL)){
    int id = getNextButtonID();
    Sensor* sens = new Sensor(newsensorloc, id);
    associateConnection(sens);
    sensorlist.append(sens);  //Add all new intruders
    makeSensorVisible(sens);
    newsensorloc = NULL; //Clear list of sensors to be added
  }
  
  //Add new intruders, if any, when dialog isn't open
  if ((newintruder != NULL) && (newintruderdialog == NULL)){
    intruders->add(newintruder);  //Add all new intruders
    newintruder->makeVisible(getNextButtonID());
    newintruder = NULL; // Remove all new intruders from the list of new intruders to be added
  }
  
  if (updates_enabled){
    try {
      //cout << "updating intruders" << endl;
      
      //Update intruder locations, velocities, and images.
      std::vector<int> deleted = intruders->updateAll();
      
      //Send out packets based on new intruder locations.
      sendAllPackets();
      
      //Clean up the buttons by freeing space in the ID pool
      int a;
      for (a = 0; a < deleted.size(); a++)
	freeButtonID(deleted[a]);
      //cout << "done updating intruders" << endl;
    } catch (...){
      cout << "Error updating intruders\n";
    }
  }
  Show(true);
}


void NRLFrame::sendAllPackets(void){
  //cout << "Sending all packets" << endl;
  int a, b;
  Timestamp nowtime; //initialize to current wall clock time
  std::vector<Packet*> mypkts;
  mypkts.reserve(intruders->size() * sensorlist.size()); //Preallocate memory for better performance
  
  //NOTE: Missed packets are naively accounted for in genPkt()
  
  for (a = 0; a < sensorlist.size(); a++){
    for (b = 0; b < intruders->size(); b++){
      Packet* pkt = generator->genPacket(intruders->intruders[b],
					 sensorlist.slist[a],
					 &nowtime);
      mypkts.push_back(pkt); //NOTE: It's more efficient to send these in arrays
      //Log the packet that we are sending
      //logger->log(pkt, LT_RECV_PKT);
    }
    //Send all packets generated at this sensor
    //cout << "About to send group of packets" << endl;
    sender->sendGroup(mypkts, sensorlist.slist[a]->sock);
    //cout << "Sending done" << endl;
    for (b = 0; b < mypkts.size(); b++)
      delete mypkts[b];
    mypkts.clear();
  }
  
  //cout << "All sending done" << endl;
}



/*----------Socket Event handlers------------*/

//From example file
void NRLFrame::OnSimStart(wxCommandEvent& WXUNUSED(event))
{
  if (updates_enabled){
    cout << "Simulation cannot start--it's already running!" << endl;
    return;
  }
  
  if (connections.size() != 3){
    cout << "Warning: cannot start simulation. " << connections.size() << " open connections (need 3)" << endl;
    return;
  }
  
  //Enable updating of intruder positions, sending of packets.
  updates_enabled = true;
}


void NRLFrame::OnSimStop(wxCommandEvent& WXUNUSED(event)){
  updates_enabled = false;
}


void NRLFrame::OnConnectToServer(wxCommandEvent& WXUNUSED(event)){
  // Ask user for server address
  wxString hostname = wxGetTextFromUser(
  _("Enter the Internet address of the Localization Solver:"),
  _("Connect ..."), _("localhost"));

  wxIPV4address addr; //BUG: Program cannot handle IPv6
  addr.Hostname(hostname);
  addr.Service(3000);
  
  int a;
  for (a = 0; a < NUM_SENSORS_DESIRED; a++){
    // Create the socket
    wxSocketClient* sock = new wxSocketClient();
    // Set up the event handler and subscribe to relevant events
    sock->SetEventHandler(*this, NRLID_SOCKET);
    sock->SetNotify(wxSOCKET_CONNECTION_FLAG | wxSOCKET_LOST_FLAG);
    sock->Notify(true);
    sock->Connect(addr, false);
    //Connections haven't been made yet. We will get events for each one individually.
  }
}


void NRLFrame::OnSocketEvent(wxSocketEvent& event){
  wxSocketBase* sock = event.GetSocket();
  // Process the event
  switch(event.GetSocketEvent()){
    
    case wxSOCKET_CONNECTION:
    {
      connections.push_back(sock);
      //FIXME: Add sensor to sensorlist
      cout << "Connection successful\n";
      break;
    }

    // The server hangs up after sending the data
    case wxSOCKET_LOST:
    {  
      cout << "Sensor connection lost\n";
      
      //Remove sensor from sensor list. User will need to recreate it.
      Sensor* sens = sensorlist.getSensor(sock);
      sender->cancel(sock); //Blacklist it so the sender won't try to access the socket being destroyed
      sensorlist.remove(sock);     
      //Remove socket from list of open connections.
      int a;
      for (a = 0; a < connections.size(); a++){
	if (connections[a] == sock){
	  sock->Destroy();
	  connections.erase(connections.begin() + a);
	  Show(true);
	  return;
	}
      }
      cout << "Warning! Connection being destroyed not found in list of open connections" << endl;
      sock->Destroy();
      Show(true);
      break;
    }
  }
}


void NRLFrame::makeSensorVisible(Sensor* sens){
  //cout << "makeSensorVisible id is " << sens->id << endl;
  sens->button = new wxBitmapButton(theapp->panel, sens->id, *(sens->picture),
				    computeItemPosition(sens->loc, 70, 75), wxSize(70, 75), 0);
  //cout << "OK4" << endl;
  Show(true);
}


void NRLFrame::makeSensorInvisible(Sensor* sens){
  delete sens->button;
  sens->button = NULL;
  Show(true);
}

//Complexity: O(s) * sensorlist.getSensor()
///@return: Whether or not a connection was associated with the sensor.
bool NRLFrame::associateConnection(Sensor* sens){
  //Note: this function may be called whether or not any connections exist.
  //Note: this function assumes that the sensor passed in as an argument is part of sensorlist.
  int a;
  //cout << "Attempting to associate sensor with socket" << endl;
  Sensor* temp = NULL;
  for (a = 0; a < connections.size(); a++){
    //cout << "Examining connections[" << a << "]: " << connections[a] << endl;
    temp = sensorlist.getSensor(connections[a]); //Check to see if connection is in use
    if (temp == NULL){
      //That connection has no sensor associated with it.
      sens->setSocket(connections[a]);
      //cout << "Associated sensor with socket" << endl;
      return true;
    }
  }
  return false;
}

/********** The many button events*********/

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
  
  // FIXME: Expand this in the future to provide information about the Intruder or Sensor.
  
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
