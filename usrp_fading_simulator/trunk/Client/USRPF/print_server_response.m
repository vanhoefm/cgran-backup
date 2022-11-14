% Low level function for USRPF. This function is used to print the
% reponse from the USRP server computer.
% Asumes that the server response has already been interprested by
% interp_server_response()
function print_server_response(response)
    disp('----------------------------------------------------------')
    disp('Server response:')
    disp (['--Instruction: ', response.id])
    disp (['--Success: ', response.success])
    disp (['--Information: ', response.info])
    disp('----------------------------------------------------------')
end