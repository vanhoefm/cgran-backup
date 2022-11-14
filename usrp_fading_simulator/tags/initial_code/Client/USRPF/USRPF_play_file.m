%**************************************************************************
% USRPF_play_file(): Applies Rayleigh fading (if enabled) and transmits the
% specified file. If the file doesn't exist on the server computer, or it
% can't be transmitted then an error is thrown. See the related function
% USRPF_send_file.
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection. 
% - file_name: The local file that will be transmitted.
%
% Outputs:
%
% An error if the file cannot be played.
%
% Example: 
%
% USRPF_play_file(io, 'c4fm_1011.dat');
%
% Author: Jonas Hodel
% Date: 23/04/07
%**************************************************************************
function USRPF_play_file(connection, file_name)
 
    command_id = 'play_file';
 
    command = [command_id, ':', file_name];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to play file')
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