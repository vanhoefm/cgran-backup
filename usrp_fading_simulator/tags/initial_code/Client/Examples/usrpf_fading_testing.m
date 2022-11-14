%**************************************************************************
% This script is an exmple of how the USRPF server can be used from within
% Matlab. It creates a data file (C4FM 1011 Hz) and sends it to the USRPF
% server for playback.
%
% Author: Jonas Hodel
% Date: 01/05/07
%**************************************************************************

% Simulation constants.
%USRP is divided into side A and side B for plugging in daughter boards.
A = 0;  %With current hardware this effectively means the FLEX400 board
FLEX400 = A;
B = 1; %With current hardware this effectively means the LFTX
LFTX = B;

FROM_FILE = 2;
FROM_RF = 3;
%--------------------------------------------------------------------------
% USRPF Server address.
ip_address = '172.25.115.19';
dataSrcForTransmission = FROM_RF;
txDst = LFTX;       %FLEX400 or LFTX
flex400_vco_error = 3.3e3;
%flex400_vco_error = 0e3;
flex400_vco_rx_freq = 452.75e6;
rf_tx_freq = 440.0e6;

%Add a tweak factor that compensates for the inaccuracy of the flex400
%Tx/Rx board's receive VCO frequency. Typically it seems to be around
%2.7KHz after approx 45 minutes warm up time. Until an automatic tuning 
%method is created for the GNU Radio platform it is best to manually verify
%the offset and then enter the correction here.
rf_rx_freq = flex400_vco_rx_freq+flex400_vco_error;
rf_power = -30;
vehicle_speed = 0;
%fs = 192e3;
fs = 250e3;
% Name of the file that is generated, and then sent to the server, or
% pre-exists on the server, in the location /home/FTP-share/upload
file_name = 'data.dat';
file_name = 'thsd_nb_500ksps.dat';



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

USRPF_set_output_dboard(io,txDst);

if dataSrcForTransmission == FROM_RF
    USRPF_play_receiving_rf(io,rf_rx_freq)
elseif dataSrcForTransmission == FROM_FILE
    % Start playback
    % disp('Starting playback')
    USRPF_play_file(io, file_name);
    %USRPF_close_connection(io);
else
    disp('Invalid data source, stopping')
    USRPF_close_connection(io);
    break;
end

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
