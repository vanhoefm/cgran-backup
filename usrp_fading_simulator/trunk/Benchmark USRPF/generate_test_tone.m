%**************************************************************************
% Quick script that creates a data file (C4FM 1011 Hz) and sends it to the
% USRPF server for playback.
%
% Author: Jonas Hodel
% Date: 08/05/07
%**************************************************************************

% Simulation constants.
%--------------------------------------------------------------------------
% USRPF Server address.
ip_address = '172.25.114.1';
fs = 192e3;
% Name of the file that is gnerated and then sent to the server.
file_name = 'c4fm1011test.dat';
%--------------------------------------------------------------------------

% Generate a C4FM 1011 Hz test tone.
%--------------------------------------------------------------------------
sig = C4FM_TX_1011Hz(1,0,150);
% Safe the tone to file
write_complex_binary(sig, file_name);
%--------------------------------------------------------------------------


% Send the test file to the server.
%--------------------------------------------------------------------------
USRPF_send_file(ip_address, file_name);
%--------------------------------------------------------------------------
