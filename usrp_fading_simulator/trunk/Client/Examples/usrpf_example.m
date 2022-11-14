%**************************************************************************
% This script is an exmple of how the USRPF server can be used from within
% Matlab. It creates a data file (C4FM 1011 Hz) and sends it to the USRPF
% server for playback.
%
% Author: Jonas Hodel
% Date: 01/05/07
%**************************************************************************

% Simulation constants.
%--------------------------------------------------------------------------
% USRPF Server address.
ip_address = '172.25.115.19';
rf_tx_freq = 440.0e6;
rf_rx_freq = 418.075e6;
rf_power = -47;
vehicle_speed = 100;
%fs = 192e3;
fs = 256e3;
% Name of the file that is gnerated and then sent to the server.
file_name = 'tone.dat';
%--------------------------------------------------------------------------

% % Generate a C4FM 1011 Hz test tone.
% %--------------------------------------------------------------------------
% sig = C4FM_TX_1011Hz(1,0,67);
% % Safe the tone to file
% write_complex_binary(sig, file_name);
% %--------------------------------------------------------------------------


% Communicate with the server (make sure that the server is running on the
% Linux USRP computer).
%--------------------------------------------------------------------------
% Open a connection to the server.
io = USRPF_open_connection(ip_address);

% Set simulation parrameters.
USRPF_set_rf_tx_freq(io, rf_tx_freq);
USRPF_set_rf_tx_power(io, rf_power);
USRPF_set_file_sample_rate(io, fs);
USRPF_set_receiver_speed(io,vehicle_speed);

USRPF_play_receiving_rf(io,rf_rx_freq)

% % % Send the file to the server.
% % USRPF_send_file(ip_address, file_name);
% 
% % Start playback, then terminate after 10 seconds
% disp('Starting playback')
% USRPF_play_file(io, file_name);
% pause(10)
pause
 USRPF_play_stop(io);
% disp('ending playback')
% 
% % Finally, close the connection to the server.
 USRPF_close_connection(io);
%--------------------------------------------------------------------------
