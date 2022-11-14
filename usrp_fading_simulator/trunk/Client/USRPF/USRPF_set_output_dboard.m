%**************************************************************************
% USRPF_set_rf_tx_freq(): Sets the RF transmit frequency of the USRP
% connected to the server PC.
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection. 
% - rf_freq: The RF frequency (Hz). The URSP is currently limited between
% 400 and 500 MHz.
% 
% Outputs:
%
% An error if the RF frequency cannot be set.
%
% Example:
%
% connection = USRPF_set_rf_tx_freq(io, 440.1e6);
%
% Author: Jonas Hodel
% Date: 23/04/07
%**************************************************************************
function USRPF_set_output_dboard(connection, dboard)
 
    command_id= 'set_output_dboard';
 
    if ~( (dboard == 0) || (dboard == 1) )
        error('Invalid USRP Side Selection')
        USRPF_close_connection(io);
    end
 
    command = [command_id, ':', num2str(dboard,'%d')];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to set USRP Side Selection')
        USRPF_close_connection(io);
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