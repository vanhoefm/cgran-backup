%**************************************************************************
% USRPF_play_stop(): Stop the play back (RF transmission) of the USRP on the
% USRPF server computer. If playback is not in progress then nothing
% happens. 
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection. 
%
% Outputs:
%
% An error if playback cannot be stopped.
%
% Example: 
%
% USRPF_play_stop(io);
%
% Author: Jonas Hodel
% Date: 08/05/07
%**************************************************************************
function USRPF_play_stop(connection)
 
    command_id = 'play_stop';
 
    command = [command_id, ':'];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to stop playback')
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