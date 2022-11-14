%**************************************************************************
% USRPF_set_rf_tx_power(): Sets the RF transmit power of the USRP
% connected to the server PC.
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection. 
% - rf_power: The RF power in dB. Valid inputs are from -110 to 0 dB (these
% ranges need to be tested).
% 
% Outputs:
%
% An error if the RF power cannot be set.
%
% Example:
%
% connection = USRPF_set_rf_tx_freq(io, 440.1e6);
%
% Author: Jonas Hodel
% Date: 23/04/07
%**************************************************************************
function USRPF_set_rf_tx_power(connection, rf_power)
 
    command_id= 'set_rf_tx_power';
 
    if rf_power < -85 || rf_power > 0
        error('RF transmit power must be -85 to 0 dB')
    end
 
    warning('RF power control is very rough and at best only approximate between -40 and 0 dB')
    
    command = [command_id, ':', num2str(rf_power)];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to set RF transmit power')
    end
    
    % This assumes that the server will respond in a timely fashion. There
    % should really be a time-out but no time to implement at the moment.
    server_reply_str = MEX_USRPF_read_string(connection);
 
    % Validate the response from the server. 
    server_reply = interp_server_response(server_reply_str);
    if (strcmp(server_reply.id, command_id) ~= 1) || (strcmp(server_reply.success, 'success') ~= 1)
        print_server_response(server_reply);
        error('Unexpected server response')
    end
    
end