/** NRLHelpDialog.h: A wxMessageDialog that clears its own pointer, used for "WHat Next?" help dialogs
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
#ifndef NRL_HELP_DIALOG_H
#define NRL_HELP_DIALOG_H

#include <wx/dialog.h>

using namespace std;

class NRLHelpDialog: public wxDialog{    
  DECLARE_EVENT_TABLE()
private:
  NRLHelpDialog** mydialog; //Pointer to main frame's pointer to us. Used for determining if the dialog box is open.
  /// Returns the data from the dialog
  void OnOKClick(wxCommandEvent& WXUNUSED); //Clear main frame's pointer on exit
public:
  /// Default constructor and destructor. Use Create() to initialize everything.
  bool Create(NRLHelpDialog** dialog, wxString text, wxWindow* form);
};


#endif
