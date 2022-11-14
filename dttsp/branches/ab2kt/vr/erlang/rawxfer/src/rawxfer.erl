%%%-------------------------------------------------------------------
%%% File        : rawxfer.erl
%%% Author      : Frank Brickle <brickle@pobox.com>
%%% Copyright   : FSF
%%% License     : GPLv3
%%% Description : Simple, unsafe remote file copy
%%%               for cross-loading GNU Radio python scripts
%%%               via back-channel
%%%
%%% Created     : Dec 2008 by Frank Brickle <brickle@pobox.com>
%%% @author Frank Brickle <brickle@pobox.com>
%%% @copyright 2008 by the Free Software Foundation, license GPLv3
%%% @version 1.0.0
%%% @doc Simple, unsafe remote file copy.
%%%      For cross-loading GNU Radio python scripts via back channel.
%%% @end
%%%-------------------------------------------------------------------
-module(rawxfer).

-vsn("1").
-author('brickle@pobox.com').
-purpose("simple, unsafe remote file copy for backchannel cross-loading").
-copyright("FSF").
-license("GPLv3").

-export([get_file/3,put_file/3]).

%%-
%% fetch file named Remote_path
%%   living on Erlang node Node
%% save in local file named Local_path
%%-
%% @spec (Node::atom(),
%%        Remote_path::string(),
%%        Local_path::string()) -> ok | {error,Reason}
%% @doc Fetch file named Remote_path,
%%      living on Erlang node Node;
%%      save in local file named Local_path.
get_file(Node,Remote_path,Local_path) ->
    case rpc:call(Node,file,read_file,[Remote_path]) of
	{ok,Bin} -> file:write_file(Local_path,[Bin]),
		    ok;
	{error,Reason} -> {error,Reason}
    end.

%%-
%% send file named Local_path
%% to be saved as file named Remote_path
%%   living on Erlang node Node
%%-
%% @spec (Local_path::string(),
%%        Node::atom(),
%%        Remote_path::string()) -> ok | {error,Reason}
%% @doc Deposit file named Local_path,
%%      on remote file named Remote_path,
%%      living on Erlang node Node.
put_file(Local_path,Node,Remote_path) ->
    case file:read_file(Local_path) of
	{ok,Bin} -> rpc:call(Node,file,write_file,[Remote_path,Bin]);
	{error,Reason} -> {error,Reason}
    end.
