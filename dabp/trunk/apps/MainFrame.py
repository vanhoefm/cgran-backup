#Boa:Frame:MainFrame

import wx
import os.path
import dabp_fic
import dabp_msc

def create(parent):
    return MainFrame(parent)

[wxID_MAINFRAME, wxID_MAINFRAMEBUTTONCONFIG, wxID_MAINFRAMEBUTTONPLAY, 
 wxID_MAINFRAMEBUTTONRECORD, wxID_MAINFRAMEBUTTONSAMPLE, 
 wxID_MAINFRAMEBUTTONSCAN, wxID_MAINFRAMEBUTTONSTOP, wxID_MAINFRAMECONFIG, 
 wxID_MAINFRAMEFREQ, wxID_MAINFRAMEGAIN, wxID_MAINFRAMEMAINPANEL, 
 wxID_MAINFRAMEMAINSTATUSBAR, wxID_MAINFRAMEMODE, wxID_MAINFRAMEPROGLISTVIEW, 
 wxID_MAINFRAMERBSAMPLE, wxID_MAINFRAMERBUSRP, wxID_MAINFRAMESAMPLE, 
 wxID_MAINFRAMESTCONFIG, wxID_MAINFRAMESTFREQ, wxID_MAINFRAMESTFREQUNIT, 
 wxID_MAINFRAMESTGAIN, wxID_MAINFRAMESTGAINUNIT, wxID_MAINFRAMESTMODE, 
 wxID_MAINFRAMESTMODETAIL, wxID_MAINFRAMESTSAMPLE, 
] = [wx.NewId() for _init_ctrls in range(25)]

[wxID_MAINFRAMEMENUFILECONFIGURATION, wxID_MAINFRAMEMENUFILEEXIT, 
] = [wx.NewId() for _init_coll_menuFile_Items in range(2)]

[wxID_MAINFRAMEMENUHELPABOUT] = [wx.NewId() for _init_coll_menuHelp_Items in range(1)]

class MainFrame(wx.Frame):
    def _init_coll_buttonFlexGridSizer_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.buttonScan, 0, border=0, flag=0)
        parent.AddWindow(self.buttonRecord, 0, border=0, flag=0)
        parent.AddWindow(self.buttonPlay, 0, border=0, flag=0)
        parent.AddWindow(self.buttonStop, 0, border=0, flag=0)

    def _init_coll_leftBoxSizer_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.paramFlexGridSizer, 0, border=2, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.buttonFlexGridSizer, 0, border=0, flag=0)

    def _init_coll_paramFlexGridSizer_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.stMode, 0, border=2,
              flag=wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL)
        parent.AddWindow(self.mode, 0, border=2, flag=wx.EXPAND | wx.ALL)
        parent.AddWindow(self.stModeTail, 0, border=0, flag=wx.GROW)
        parent.AddWindow(self.stFreq, 0, border=2,
              flag=wx.GROW | wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.freq, 0, border=2, flag=wx.EXPAND | wx.ALL)
        parent.AddWindow(self.stFreqUnit, 0, border=0, flag=wx.GROW)
        parent.AddWindow(self.stGain, 0, border=2,
              flag=wx.GROW | wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.gain, 0, border=2, flag=wx.EXPAND | wx.ALL)
        parent.AddWindow(self.stGainUnit, 0, border=0, flag=wx.GROW)
        parent.AddWindow(self.stConfig, 0, border=2,
              flag=wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL)
        parent.AddWindow(self.config, 0, border=2, flag=wx.EXPAND | wx.ALL)
        parent.AddWindow(self.buttonConfig, 0, border=0, flag=0)
        parent.AddWindow(self.stSample, 0, border=2,
              flag=wx.ALL | wx.GROW | wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.sample, 0, border=2, flag=wx.EXPAND | wx.ALL)
        parent.AddWindow(self.buttonSample, 0, border=0, flag=0)
        parent.AddWindow(self.rbUsrp, 0, border=2,
              flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.rbSample, 0, border=2,
              flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

    def _init_coll_paramFlexGridSizer_Growables(self, parent):
        # generated method, don't edit

        parent.AddGrowableCol(1)

    def _init_coll_mainBoxSizer_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.leftBoxSizer, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.progListView, 0, border=2, flag=wx.EXPAND)

    def _init_coll_mainMenuBar_Menus(self, parent):
        # generated method, don't edit

        parent.Append(menu=self.menuFile, title=u'File')
        parent.Append(menu=self.menuHelp, title=u'Help')

    def _init_coll_menuHelp_Items(self, parent):
        # generated method, don't edit

        parent.Append(help=u'About DAB+ Receiver',
              id=wxID_MAINFRAMEMENUHELPABOUT, kind=wx.ITEM_NORMAL,
              text=u'About')
        self.Bind(wx.EVT_MENU, self.OnMenuHelpAboutMenu,
              id=wxID_MAINFRAMEMENUHELPABOUT)

    def _init_coll_menuFile_Items(self, parent):
        # generated method, don't edit

        parent.Append(help=u'Set configuration file',
              id=wxID_MAINFRAMEMENUFILECONFIGURATION, kind=wx.ITEM_NORMAL,
              text=u'Configuration')
        parent.Append(help=u'Exit the receiver program',
              id=wxID_MAINFRAMEMENUFILEEXIT, kind=wx.ITEM_NORMAL, text=u'Exit')
        self.Bind(wx.EVT_MENU, self.OnMenuFileConfigurationMenu,
              id=wxID_MAINFRAMEMENUFILECONFIGURATION)
        self.Bind(wx.EVT_MENU, self.OnMenuFileExitMenu,
              id=wxID_MAINFRAMEMENUFILEEXIT)

    def _init_coll_progListView_Columns(self, parent):
        # generated method, don't edit

        parent.InsertColumn(col=0, format=wx.LIST_FORMAT_CENTRE, heading=u'ID',
              width=40)
        parent.InsertColumn(col=1, format=wx.LIST_FORMAT_LEFT,
              heading=u'Program', width=230)

    def _init_coll_mainStatusBar_Fields(self, parent):
        # generated method, don't edit
        parent.SetFieldsCount(1)

        parent.SetStatusText(number=0, text=u'status')

        parent.SetStatusWidths([-1])

    def _init_sizers(self):
        # generated method, don't edit
        self.mainBoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.leftBoxSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.paramFlexGridSizer = wx.FlexGridSizer(cols=3, hgap=0, rows=0,
              vgap=0)

        self.buttonFlexGridSizer = wx.FlexGridSizer(cols=2, hgap=0, rows=0,
              vgap=0)

        self._init_coll_mainBoxSizer_Items(self.mainBoxSizer)
        self._init_coll_leftBoxSizer_Items(self.leftBoxSizer)
        self._init_coll_paramFlexGridSizer_Items(self.paramFlexGridSizer)
        self._init_coll_paramFlexGridSizer_Growables(self.paramFlexGridSizer)
        self._init_coll_buttonFlexGridSizer_Items(self.buttonFlexGridSizer)

        self.mainPanel.SetSizer(self.mainBoxSizer)

    def _init_utils(self):
        # generated method, don't edit
        self.menuFile = wx.Menu(title=u'')

        self.menuHelp = wx.Menu(title=u'')

        self.mainMenuBar = wx.MenuBar()

        self._init_coll_menuFile_Items(self.menuFile)
        self._init_coll_menuHelp_Items(self.menuHelp)
        self._init_coll_mainMenuBar_Menus(self.mainMenuBar)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_MAINFRAME, name=u'MainFrame',
              parent=prnt, pos=wx.Point(551, 244), size=wx.Size(644, 446),
              style=wx.DEFAULT_FRAME_STYLE, title=u'DAB+ Receiver')
        self._init_utils()
        self.SetClientSize(wx.Size(636, 412))
        self.SetMenuBar(self.mainMenuBar)
        self.SetIcon(wx.Icon(u'Boa.ico',wx.BITMAP_TYPE_ICO))

        self.mainStatusBar = wx.StatusBar(id=wxID_MAINFRAMEMAINSTATUSBAR,
              name=u'mainStatusBar', parent=self, style=0)
        self._init_coll_mainStatusBar_Fields(self.mainStatusBar)
        self.SetStatusBar(self.mainStatusBar)

        self.mainPanel = wx.Panel(id=wxID_MAINFRAMEMAINPANEL, name=u'mainPanel',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(636, 370),
              style=wx.TAB_TRAVERSAL)

        self.progListView = wx.ListView(id=wxID_MAINFRAMEPROGLISTVIEW,
              name=u'progListView', parent=self.mainPanel, pos=wx.Point(358, 0),
              size=wx.Size(322, 370), style=wx.LC_REPORT)
        self.progListView.SetLabel(u'Program List')
        self._init_coll_progListView_Columns(self.progListView)

        self.stMode = wx.StaticText(id=wxID_MAINFRAMESTMODE, label=u'DAB Mode',
              name=u'stMode', parent=self.mainPanel, pos=wx.Point(2, 2),
              size=wx.Size(70, 21), style=0)

        self.stFreq = wx.StaticText(id=wxID_MAINFRAMESTFREQ, label=u'Frequency',
              name=u'stFreq', parent=self.mainPanel, pos=wx.Point(2, 27),
              size=wx.Size(70, 21), style=0)

        self.stGain = wx.StaticText(id=wxID_MAINFRAMESTGAIN, label=u'USRP Gain',
              name=u'stGain', parent=self.mainPanel, pos=wx.Point(2, 52),
              size=wx.Size(70, 21), style=0)

        self.stConfig = wx.StaticText(id=wxID_MAINFRAMESTCONFIG,
              label=u'Configuration', name=u'stConfig', parent=self.mainPanel,
              pos=wx.Point(2, 77), size=wx.Size(70, 21), style=0)

        self.mode = wx.TextCtrl(id=wxID_MAINFRAMEMODE, name=u'mode',
              parent=self.mainPanel, pos=wx.Point(76, 2), size=wx.Size(247, 21),
              style=wx.TE_PROCESS_ENTER, value=u'1')
        self.mode.Bind(wx.EVT_TEXT_ENTER, self.OnModeTextEnter)

        self.freq = wx.TextCtrl(id=wxID_MAINFRAMEFREQ, name=u'freq',
              parent=self.mainPanel, pos=wx.Point(76, 27), size=wx.Size(247,
              21), style=wx.TE_PROCESS_ENTER, value=u'204.64')
        self.freq.Bind(wx.EVT_TEXT_ENTER, self.OnFreqTextEnter)

        self.gain = wx.TextCtrl(id=wxID_MAINFRAMEGAIN, name=u'gain',
              parent=self.mainPanel, pos=wx.Point(76, 52), size=wx.Size(247,
              21), style=wx.TE_PROCESS_ENTER, value=u'8')
        self.gain.Bind(wx.EVT_TEXT_ENTER, self.OnGainTextEnter)

        self.config = wx.TextCtrl(id=wxID_MAINFRAMECONFIG, name=u'config',
              parent=self.mainPanel, pos=wx.Point(76, 77), size=wx.Size(247,
              21), style=0, value=u'channel.conf')
        self.config.SetEditable(False)

        self.buttonScan = wx.Button(id=wxID_MAINFRAMEBUTTONSCAN, label=u'Scan',
              name=u'buttonScan', parent=self.mainPanel, pos=wx.Point(0, 158),
              size=wx.Size(75, 23), style=0)
        self.buttonScan.Bind(wx.EVT_BUTTON, self.OnButtonScanButton)

        self.buttonRecord = wx.Button(id=wxID_MAINFRAMEBUTTONRECORD,
              label=u'Record', name=u'buttonRecord', parent=self.mainPanel,
              pos=wx.Point(75, 158), size=wx.Size(75, 23), style=0)
        self.buttonRecord.Bind(wx.EVT_BUTTON, self.OnButtonRecordButton)

        self.buttonPlay = wx.Button(id=wxID_MAINFRAMEBUTTONPLAY, label=u'Play',
              name=u'buttonPlay', parent=self.mainPanel, pos=wx.Point(0, 181),
              size=wx.Size(75, 23), style=0)
        self.buttonPlay.Bind(wx.EVT_BUTTON, self.OnButtonPlayButton)

        self.buttonStop = wx.Button(id=wx.ID_STOP, label=u'Stop',
              name=u'buttonStop', parent=self.mainPanel, pos=wx.Point(75, 181),
              size=wx.Size(75, 23), style=0)
        self.buttonStop.Bind(wx.EVT_BUTTON, self.OnButtonStopButton)

        self.stModeTail = wx.StaticText(id=wxID_MAINFRAMESTMODETAIL, label=u'',
              name=u'stModeTail', parent=self.mainPanel, pos=wx.Point(325, 0),
              size=wx.Size(25, 25), style=0)

        self.stFreqUnit = wx.StaticText(id=wxID_MAINFRAMESTFREQUNIT,
              label=u'MHz', name=u'stFreqUnit', parent=self.mainPanel,
              pos=wx.Point(325, 25), size=wx.Size(25, 25), style=0)

        self.stGainUnit = wx.StaticText(id=wxID_MAINFRAMESTGAINUNIT,
              label=u'dB', name=u'stGainUnit', parent=self.mainPanel,
              pos=wx.Point(325, 50), size=wx.Size(25, 25), style=0)

        self.buttonConfig = wx.Button(id=wxID_MAINFRAMEBUTTONCONFIG,
              label=u'...', name=u'buttonConfig', parent=self.mainPanel,
              pos=wx.Point(325, 75), size=wx.Size(20, 23), style=0)
        self.buttonConfig.Bind(wx.EVT_BUTTON, self.OnMenuFileConfigurationMenu)

        self.rbUsrp = wx.RadioButton(id=wxID_MAINFRAMERBUSRP, label=u'USRP',
              name=u'rbUsrp', parent=self.mainPanel, pos=wx.Point(2, 127),
              size=wx.Size(65, 21), style=wx.RB_GROUP)
        self.rbUsrp.SetValue(True)
        self.rbUsrp.Bind(wx.EVT_RADIOBUTTON, self.OnRbUsrpRadiobutton)

        self.rbSample = wx.RadioButton(id=wxID_MAINFRAMERBSAMPLE,
              label=u'Sample', name=u'rbSample', parent=self.mainPanel,
              pos=wx.Point(76, 127), size=wx.Size(106, 21), style=0)
        self.rbSample.SetValue(False)
        self.rbSample.Bind(wx.EVT_RADIOBUTTON, self.OnRbSampleRadiobutton)

        self.stSample = wx.StaticText(id=wxID_MAINFRAMESTSAMPLE,
              label=u'Sample', name=u'stSample', parent=self.mainPanel,
              pos=wx.Point(2, 102), size=wx.Size(70, 21), style=0)

        self.sample = wx.TextCtrl(id=wxID_MAINFRAMESAMPLE, name=u'sample',
              parent=self.mainPanel, pos=wx.Point(76, 102), size=wx.Size(247,
              21), style=0, value=u'sample.dat')
        self.sample.SetEditable(False)
        self.sample.Enable(False)

        self.buttonSample = wx.Button(id=wxID_MAINFRAMEBUTTONSAMPLE,
              label=u'...', name=u'buttonSample', parent=self.mainPanel,
              pos=wx.Point(325, 100), size=wx.Size(20, 23), style=0)
        self.buttonSample.Enable(False)
        self.buttonSample.Bind(wx.EVT_BUTTON, self.OnButtonSampleButton)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)
        self.confFile='channel.conf'
        self.config.ChangeValue(self.confFile)
        self.sampFile='sample.dat'
        self.sample.ChangeValue(self.sampFile)
        self.dabMode=1
        self.mode.ChangeValue(str(self.dabMode))
        self.dabFreq=204.64
        self.freq.ChangeValue(str(self.dabFreq))
        self.dabGain=8
        self.gain.ChangeValue(str(self.dabGain))
        self.grblk = None
        # fill program list
        self.FillProgramList()

    def FillProgramList(self):
        if os.path.isfile(self.confFile):
            with open(self.confFile, 'r') as f:
                self.progListView.DeleteAllItems()
                lcnt = 0
                for line in f:
                    # one line at a time
                    if line[0]!='#': # skip comments
                        ar=line.split(',')
                        self.progListView.InsertStringItem(lcnt,ar[0])
                        self.progListView.SetStringItem(lcnt,1,ar[2])
                        lcnt=lcnt+1
                self.progListView.Select(0) # select the first program
        
    def OnMenuFileConfigurationMenu(self, event):
        dlg = wx.FileDialog(self, 'Select a configuration file', '.', '', '*.conf', wx.OPEN)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.confFile = dlg.GetPath()
                self.mainStatusBar.SetStatusText('Configuration File: '+self.confFile)
                self.config.ChangeValue(self.confFile)
                self.FillProgramList()
        finally:
            dlg.Destroy()

    def OnMenuFileExitMenu(self, event):
        self.Close()

    def OnMenuHelpAboutMenu(self, event):
        info = wx.AboutDialogInfo()
        info.Name = 'DAB+ Receiver'
        info.Version = '0.0.1'
        info.Copyright = '(C) Kyle Zhou'
        info.Description = 'DAB+ receiver according to ETSI EN 300 401 V1.4.1 and ETSI TS 102 563 V1.1.1'
        wx.AboutBox(info)

    def OnButtonRecordButton(self, event):
        if self.grblk != None:
            self.mainStatusBar.SetStatusText('Playing in progress. Please stop first!')
            return
        
        dlg = wx.FileDialog(self, 'Choose a file', '.', '', '*.aac', wx.SAVE|wx.OVERWRITE_PROMPT)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.recordFile = dlg.GetPath()
                
        finally:
            dlg.Destroy()
        
        if os.path.isfile(self.confFile):
            idx = self.progListView.GetFirstSelected()
            subchid = int(self.progListView.GetItemText(idx))
            self.grblk = dabp_msc.dabp_msc(self.dabMode, self.dabFreq*1e6, self.dabGain, self.confFile, subchid, self.recordFile, self.rbUsrp.GetValue())
            self.mainStatusBar.SetStatusText('Recording the program to '+self.recordFile+' ...')
            self.grblk.start()
            
        else:
            self.mainStatusBar.SetStatusText('Configuration File does not exist!')

    def OnModeTextEnter(self, event):
        self.dabMode = int(self.mode.GetValue())
        self.mainStatusBar.SetStatusText('Set DAB mode '+str(self.dabMode))

    def OnFreqTextEnter(self, event):
        self.dabFreq = float(self.freq.GetValue())
        self.mainStatusBar.SetStatusText('Set DAB frequency to '+str(self.dabFreq)+' MHz')

    def OnGainTextEnter(self, event):
        self.dabGain = int(self.gain.GetValue())
        self.mainStatusBar.SetStatusText('Set USRP gain to '+str(self.dabGain)+' dB')

    def OnButtonSampleButton(self, event):
        dlg = wx.FileDialog(self, 'Select a sample file', '.', '', '*.dat', wx.OPEN)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.sampFile = dlg.GetPath()
                self.mainStatusBar.SetStatusText('Sample File: '+self.sampFile)
                self.sample.ChangeValue(self.sampFile)
        finally:
            dlg.Destroy()

    def OnRbUsrpRadiobutton(self, event):
        self.freq.Enable(True)
        self.gain.Enable(True)
        self.sample.Enable(False)
        self.buttonSample.Enable(False)
        self.mainStatusBar.SetStatusText('USRP source selected')

    def OnRbSampleRadiobutton(self, event):
        self.freq.Enable(False)
        self.gain.Enable(False)
        self.sample.Enable(True)
        self.buttonSample.Enable(True)
        self.mainStatusBar.SetStatusText('Sample file source selected')

    def OnButtonScanButton(self, event):
        grblk = dabp_fic.dabp_fic(self.dabMode, self.dabFreq*1e6, self.dabGain, self.confFile, self.sampFile, self.rbUsrp.GetValue())
        self.mainStatusBar.SetStatusText('Scanning for programs... Please wait...')
        grblk.run()
        grblk.dst.save_subch(self.confFile)
        self.mainStatusBar.SetStatusText('Scanning Completed!')
        self.FillProgramList()

    def OnButtonPlayButton(self, event):
        if os.path.isfile(self.confFile):
            idx = self.progListView.GetFirstSelected()
            subchid = int(self.progListView.GetItemText(idx))
            self.grblk = dabp_msc.dabp_msc(self.dabMode, self.dabFreq*1e6, self.dabGain, self.confFile, subchid, None, self.rbUsrp.GetValue())
            self.mainStatusBar.SetStatusText('Playing...')
            self.grblk.start()
            
        else:
            self.mainStatusBar.SetStatusText('Configuration File does not exist!')
            
    def OnButtonStopButton(self, event):
        if self.grblk == None:
            self.mainStatusBar.SetStatusText('Nothing to stop!')
        else:
            self.mainStatusBar.SetStatusText('Stopping... Please wait...')
            self.grblk.stop()
            self.grblk.wait()
            self.mainStatusBar.SetStatusText('Playing Stopped')
            self.grblk = None

