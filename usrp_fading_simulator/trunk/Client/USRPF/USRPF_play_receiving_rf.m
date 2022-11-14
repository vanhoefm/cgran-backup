%**************************************************************************
% USRPF_play_receiving_rf(): Applies fading and then transmits on one
% frequency, what is simultaneously received on another. For best results
% the rx frequency should be well spaced from the tx frequency. To set the
% USRP transmit frequency, see USRPF_set_rf_tx_freq().
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection(). 
% - rf_rx_freq: The RF frequency of the receiver (Hz).
%
% Outputs:
%
% An error if the received signal cannot be transmitted.
%
% Example: 
%
% USRPF_play_receiving_rf(io, 451e6);
%
% Author: Jonas Hodel
% Date: 08/05/07
%**************************************************************************
function USRPF_play_receiving_rf(connection, rf_rx_freq)
 
    command_id = 'play_receiving_rf';
 
    command = [command_id, ':', num2str(rf_rx_freq)];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to transmit received RF.')
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