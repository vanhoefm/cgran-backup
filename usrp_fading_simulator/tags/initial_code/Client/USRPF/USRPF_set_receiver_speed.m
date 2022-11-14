%**************************************************************************
% USRPF_set_receiver_speed(): Sets the receiver speed. Along with the
% receiver’s receive frequency (equal to the USRP transmit frequency), this
% affects Rayleigh fading.
% 
% Rayleigh fading is dependent on the Doppler frequency, which is defined
% as: 
%                    fd = (v*f)/c, 
% 
% where, v  = receiver speed (m/s), f = receiver frequency (Hz), and c is
% the speed of light (m/s).
%
% Inputs:
%
% - connection: an open connection to the server, see
% USRPF_open_connection.  
% - recv_speed: The receiver speed (km/h). If this is set to zero then
% Rayleigh fading is disabled. This is useful for straight through
% transmission of the signal source.
% 
% Outputs:
%
% An error if the receiver speed cannot be set.
%
% Example:
%
% connection = USRPF_set_receiver_speed(io, 100);
%
% Author: Jonas Hodel
% Date: 10/05/07
%**************************************************************************
function USRPF_set_receiver_speed(connection, recv_speed)
 
    command_id= 'set_recv_speed';
 
    if recv_speed < 0
        error('Receiver speed cannot be less than zero')
    elseif recv_speed == 0
        disp('Rayleigh fading has been DISABLED')
    else
        disp('Rayleigh fading has been ENABLED')
    end
 
    command = [command_id, ':', num2str(recv_speed)];
    success = MEX_USRPF_send_string(connection, command);
    if success == 0
        error('Unable to set receiver speed')
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