#!/usr/bin/env python
import sys,os,re,string
sys.path.append(".");
from PyQt4 import QtCore, QtGui;
from blockgen_ui import *;

class SimpleTable( QtCore.QAbstractTableModel ):
    def __init__(self, d, parent):
        QtCore.QAbstractTableModel.__init__(self,parent)
        self.d=d;

class BGM( QtGui.QMainWindow ):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent);
        self.constructorname = ":constructor:";
        self.last_method = "";
        self.ui=Ui_Blockgen();
        self.ui.setupUi(self);
        self.populate();
        self.setupSignals();
        self.setActiveMethod(self.constructorname);

    def populate(self):
        self.types = [ ("char", 1), ("unsigned char", 1), ("short", 2), ("unsigned short", 2), ("int", 4), ("unsigned int", 4), ("float",4), ("gr_complex",8), ("double",8), ("none", 0), ("void",0) ];
        for t in self.types:
            self.ui.selectInSig.addItem( t[0] );
            self.ui.selectOutSig.addItem( t[0] );
            self.ui.selectArgType.addItem( t[0] );
            self.ui.selectMethodReturnType.addItem( t[0] );
         
        self.inMdl = QtGui.QStringListModel(None);
        self.outMdl = QtGui.QStringListModel(None);
        self.methodMdl = QtGui.QStringListModel(None);
        self.argTypeMdl = QtGui.QStringListModel(None);
        self.argNameMdl = QtGui.QStringListModel(None);
        
        self.ui.inSigTable.setModel(self.inMdl);
        self.ui.outSigTable.setModel(self.outMdl);

        self.ui.methodTable.setModel(self.methodMdl);
        self.ui.argTypeTable.setModel(self.argTypeMdl);
        self.ui.argNameTable.setModel(self.argNameMdl);

        self.listAppend(self.methodMdl, self.constructorname);
        
        self.methodtable = {self.constructorname: ["none", [], []] };

    def typeIdxByName(self, name):
        i = 0;
        for p in self.types:
            if(p[0] == name):
                return i;
            i=i+1;
        return -1;



    def setActiveMethod(self, method):
        self.saveMethod();

#        print "restoring %s"%( self.methodtable[method] );

        # set the return value type
        self.ui.selectMethodReturnType.setCurrentIndex( self.typeIdxByName( self.methodtable[method][0] ));

        # Clear the tables
        self.listClear( self.argTypeMdl );
        self.listClear( self.argNameMdl );
 
        # Set Arg Types       
        for i in self.methodtable[method][1]:
            self.listAppend( self.argTypeMdl, i );

        # Set Arg Names
        for i in self.methodtable[method][2]:
            self.listAppend( self.argNameMdl, i );

        # Store Last Method to be edited for reference
        self.last_method = method;
        
    

    def setupSignals(self):
        QtCore.QObject.connect(self.ui.genButton, QtCore.SIGNAL("clicked()"), self.do_generate);
        QtCore.QObject.connect(self.ui.addInSig, QtCore.SIGNAL("clicked()"), self.addInSig);
        QtCore.QObject.connect(self.ui.addOutSig, QtCore.SIGNAL("clicked()"), self.addOutSig);
        QtCore.QObject.connect(self.ui.addMethod, QtCore.SIGNAL("clicked()"), self.addMethod);
        QtCore.QObject.connect(self.ui.addArg, QtCore.SIGNAL("clicked()"), self.addArg);
        QtCore.QObject.connect(self.ui.txtModDirBrowse, QtCore.SIGNAL("clicked()"), self.modBrowse);

        QtCore.QObject.connect(self.ui.txtModule, QtCore.SIGNAL("textChanged(const QString &)"), self.updateFn);
        QtCore.QObject.connect(self.ui.txtBlock, QtCore.SIGNAL("textChanged(const QString &)"), self.updateFn);

        QtCore.QObject.connect(self.ui.methodTable, QtCore.SIGNAL("pressed( const QModelIndex &)"), self.methodSelect );
    
        QtCore.QObject.connect(self.ui.selectMethodReturnType,  QtCore.SIGNAL("activated ( int )"), self.callSave);


    def modBrowse(self):
        op = QtGui.QFileDialog.ShowDirsOnly;
        path = QtGui.QFileDialog.getExistingDirectory(None,"Select Target GR Module Directory", "." , op);
        if(path != ""):
            self.ui.txtModDir.setText( str(path) );
            dirs = str(path).split("/");
            modname = dirs[-1];
            self.ui.txtModule.setText(modname);


    def addArg(self):
        arg_type = self.ui.selectArgType.currentText();
        arg_name = self.ui.argName.text();

        self.listAppend(self.argTypeMdl, arg_type);
        self.listAppend(self.argNameMdl, arg_name);

#        print "adding arg %s %s"%(arg_type, arg_name);
        self.saveMethod();


    def methodSelect(self, idx):
        selected_method = self.methodMdl.data( idx, 0 ).toString();
        self.setActiveMethod(str(selected_method));

    def callSave(self, val):
        self.saveMethod();
    
    def saveMethod(self):
        if(self.last_method != ""):
            return_type = self.ui.selectMethodReturnType.currentText();
            arg_types = self.argTypeMdl.stringList();
            arg_names = self.argNameMdl.stringList();
            self.methodtable[self.last_method] = [return_type, arg_types, arg_names];

    def str_methodlist(self, list_type):
        v = ""
        for methname in self.methodtable.keys():
            if(methname != self.constructorname):
                meth = self.methodtable[methname];
                return_type = meth[0];
                arglist = [];
                for j in range(0, len( meth[1] )):
                    arglist.append( "%s %s"%( meth[1][j], meth[2][j] ) );

                str_arglist = string.join( arglist, "," );

                if(list_type == "proto"):
                    v = "%s    %s %s(%s);\n"%( v, return_type, methname, str_arglist );
                elif(list_type == "defs"):
                    modulename = self.ui.txtModule.text();
                    blockname = self.ui.txtBlock.text();
                    fullname = "%s_%s"%(modulename,blockname);

                    content = "";
                    if(return_type != "void" and return_type != "none"):
                        content = "    %s rv;\n    return rv;\n"%(return_type);

                    v = "%s%s %s::%s(%s){\n%s\n}\n"%( v, return_type, fullname, methname, str_arglist, content);
                else:
                    print "invalid list_type value, misuse"

        return v;

    def str_conarglist(self,incl_types):
        meth = self.methodtable[self.constructorname];
        l = [];
        for i in range(0,len(meth[1])):
            atype = "";
            if(incl_types):
                atype = str(meth[1][i]);
            aname = str(meth[2][i]);
            l.append("%s %s"%(atype, aname));
        return string.join(l,",");
    
    def do_generate(self):
        cc_tmpl = open("template/cc.tmpl","r");
        self.cc = cc_tmpl.read();

        hh_tmpl = open("template/hh.tmpl","r");
        self.hh = hh_tmpl.read();

        ii_tmpl = open("template/ii.tmpl","r");
        self.ii = ii_tmpl.read();

        module = self.ui.txtModule.text();
        block = self.ui.txtBlock.text();

        self.set_key("HEADER", "%s_%s.hh"%(module,block));
        self.set_key("SPTR", "%s_%s_sptr"%(module,block));
        self.set_key("MAKE", "%s_make_%s"%(module,block));
        self.set_key("FULL", "%s_%s"%(module,block));
        self.set_key("SHORT", "%s"%(block));
        self.set_key("MODULE", "%s"%(module));

        self.set_key("MININ", "%s"%(self.ui.txtInMin.text()));
        self.set_key("MAXIN", "%s"%(self.ui.txtInMax.text()));
        self.set_key("MINOUT", "%s"%(self.ui.txtOutMin.text()));
        self.set_key("MAXOUT", "%s"%(self.ui.txtOutMax.text()));

        self.set_key("METHODPROTOLIST", "%s"%(self.str_methodlist("proto")) );
        self.set_key("METHODDEFLIST", "%s"%(self.str_methodlist("defs")) );
        self.set_key("CONARGLISTWITHTYPES", self.str_conarglist(True));
        self.set_key("CONARGLIST", self.str_conarglist(False));

        superClass = str(self.ui.selectSuper.currentText());
        if(superClass == "gr_sync_block"):
            self.set_key("WORK", "work (int noutput_items,\n        gr_vector_const_void_star &input_items,\n        gr_vector_void_star &output_items)");
            self.set_key("CONSUME", "");
        elif(superClass == "gr_block"):
            self.set_key("WORK", "general_work (int noutput_items,\n        gr_vector_int &ninput_items,\n        gr_vector_const_void_star &input_items,\n        gr_vector_void_star &output_items)");
            self.set_key("CONSUME", "consume_each(noutput_items);");
        else:
            print("INVALID SUPER CLASS");
            sys.exit(-1);

        self.set_key("SUPER", superClass);
        self.set_key("INDEFS", self.ioDefs());
        self.set_key("INPUT_SIGNATURE", self.genSig(self.inMdl,"IN"));
        self.set_key("OUTPUT_SIGNATURE", self.genSig(self.outMdl,"OUT"));
        self.set_key("DEFINE", string.upper("%s_%s_H"%(module,block)));

        cc_out = open(str(self.ui.txtFnCC.text()),"w");
        cc_out.writelines(self.cc);
        cc_out.close();
        hh_out = open(str(self.ui.txtFnH.text()),"w");
        hh_out.writelines(self.hh);
        hh_out.close();
        ii_out = open(str(self.ui.txtFnI.text()),"w");
        ii_out.writelines(self.ii);
        ii_out.close();

        self.ui.statusbar.showMessage("wrote %s, %s, and %s."%(str(self.ui.txtFnCC.text()),str(self.ui.txtFnH.text()),str(self.ui.txtFnI.text())));

    def ioDefs(self):
        ret = "";
        inlist = self.inMdl.stringList();
        for i in range(0,len(inlist)):
            type = inlist[i];
            ret = "%s        const %s* in%d = (const %s*) input_items[%d];\n"%(ret,type,i,type,i);
        outlist = self.outMdl.stringList();
        for i in range(0,len(outlist)):
            type = outlist[i];
            ret = "%s        %s* out%d = (%s*) output_items[%d];\n"%(ret,type,i,type,i);
        return ret;

    def set_key(self,key,repl):
        p = re.compile('####%s####'%(key));
        self.cc = p.sub(repl, self.cc, 0);
        self.hh = p.sub(repl, self.hh, 0);
        self.ii = p.sub(repl, self.ii, 0);

    def listClear(self, mdl):
        cnt = mdl.rowCount();
        mdl.removeRows(0, cnt);

    def listAppend(self, mdl, val):
        mdl.insertRows(mdl.rowCount(), 1);
        mdl.setData(mdl.index(mdl.rowCount()-1), QtCore.QVariant( val ) );

    def addInSig(self):
        self.listAppend(self.inMdl, self.ui.selectInSig.currentText());       
        self.ui.txtInMin.setText( str(self.inMdl.rowCount() ));
        self.ui.txtInMax.setText( str(self.inMdl.rowCount() ));

    def addOutSig(self):
        self.listAppend(self.outMdl, self.ui.selectOutSig.currentText());       
        self.ui.txtOutMin.setText( str(self.outMdl.rowCount() ));
        self.ui.txtOutMax.setText( str(self.outMdl.rowCount() ));

    def addMethod(self):
        method = str(self.ui.addMethodName.text());
        self.listAppend(self.methodMdl, method);
        self.methodtable[ method ] = ["void", [], []];
        self.setActiveMethod( method );

    
    def updateFn(self,nt):
        base = "%s_%s"%(self.ui.txtModule.text(), self.ui.txtBlock.text());
        self.ui.txtFnCC.setText("%s.cc"%(base));
        self.ui.txtFnH.setText("%s.hh"%(base));
        self.ui.txtFnI.setText("%s.i"%(base));
        self.ui.txtFnX.setText("%s.xml"%(base));

    def genSig(self,mdl,dir):
        l = mdl.stringList();
        count = ""
        if(len(l) > 1):
            count = str(len(l));
        
        if(len(l)==0):
            l = [0];

        sizelist = "";
        for i in l:
            sizelist = "%s,sizeof(%s)"%(sizelist,i);


        return "gr_make_io_signature%s(MIN_%s,MAX_%s%s)"%(count,dir,dir,sizelist);
    

app = QtGui.QApplication(sys.argv);
bg = BGM();
bg.show();
sys.exit(app.exec_());

    
