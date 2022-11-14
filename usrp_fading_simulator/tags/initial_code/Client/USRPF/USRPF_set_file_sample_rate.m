%**************************************************************************
% USRPF_set_file_sample_rate(): Sets the sample rate associated with the
% file source. 
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection. 
% - sample_rate: The sample rate (Samples/second)
% 
% Outputs:
%
% An error if the sample rate connont be set.
%
% Example:
%
% connection = USRPF_set_file_sample_rate(io, 440.1e6);
%
% Author: Jonas Hodel
% Date: 23/04/07
%**************************************************************************
function USRPF_set_file_sample_rate(connection, sample_rate)
 
    command_id= 'set_file_sample_rate';

    command = [command_id, ':', num2str(sample_rate)];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        warning('Unable to set sample rate of the source file')
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