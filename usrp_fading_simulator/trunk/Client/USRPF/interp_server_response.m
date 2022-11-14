% Low level function for USRPF. This function is used to inerpret the
% reponse from the USRP server computer.
% Asumes the server will responde with the following string
% command:success:information. 
% This funciton will break the above string into id, success and
% info. The ID corresponds to the command that ther server recieved,
% success determines whether the command was succeefully executed by ther
% server info is any additional information provided by the server.
function response = interp_server_response(string)
    [response.id, remain] = strtok(string, ':');
    [response.success, remain] = strtok(remain, ':');
    response.info = strtok(remain, ':');
end