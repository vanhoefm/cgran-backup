%%%-------------------------------------------------------------------
%%% File        : ducpin.erl
%%% Author      : Frank Brickle <brickle@pobox.com>
%%% Copyright   : 2008 FSF
%%% License     : GPLv3
%%% Description : proxy for Erlang control of a running sdr-core
%%%               via datagram
%%% ducpin == DttSP Update Command Proxy Implementation Nodule
%%%
%%% TODO        : * start breaking out individual update commands
%%%
%%% Created : Nov 2008 by Frank Brickle <brickle@pobox.com>
%%%
%%% @author Frank Brickle <brickle@pobox.com>
%%% @copyright 2008 by the Free Software Foundation, license GPLv3
%%% @doc UDP proxy for Erlang control of a running sdr-core via datagram.
%%% @TODO <p> start breaking out individual update commands </p>
%%% @version 1.0.0
%%%-------------------------------------------------------------------
-module(ducpin).

-vsn("1").
-author('brickle@pobox.com').
-purpose("UDP proxy to a single sdr-core").
-copyright("FSF").
-license("GPLv3").

-behavior(gen_server).

%% API
-export([start_link/0,
	 start_link/2,
	 start_link/3,
	 stop/0,
	 stop/1,
	 raw_command/1,
	 raw_command/2,
	 spill/0,
	 spill/1]).

%% gen_server callbacks
-export([init/1,
	 handle_call/3,
	 handle_cast/2,
	 handle_info/2,
	 terminate/2,
	 code_change/3]).

-define(SERVER,?MODULE).

-define(CMD_PORT,19001).
-define(CMD_HOST,"localhost").
-define(TIMEOUT,1000).

-record(state,
	{sock = null,
	 serv = ?SERVER,
	 host = ?CMD_HOST,
	 port = ?CMD_PORT,
	 wait = ?TIMEOUT}).

%%====================================================================
%% API
%%====================================================================

%%--------------------------------------------------------------------
%% Function: start_link() -> {ok,Pid} | ignore | {error,Error}
%%           start_link(Host,Port) ->
%%                           {ok,Pid} | ignore | {error,Error}
%%           start_link(ServHost,Port) ->
%%                           {ok,Pid} | ignore | {error,Error}
%% Description: Starts the server,
%%              under default or stipulated Name,
%%              on default or stipulated Host & Port
%%--------------------------------------------------------------------
%% @spec () -> {ok,State}
%% @doc Start the server with default name, host, and port.    
start_link() ->
    gen_server:start_link({local,?SERVER},?MODULE,[],[]).

%% @spec (Host::term(),Port::integer()) -> {ok,State}
%% @doc Start the server with specified host and port, default name.
start_link(Host,Port) ->
    gen_server:start_link({local,?SERVER},?MODULE,[?SERVER,Host,Port],[]).

%% @spec (Serv::term(),Host::term(),Port::integer()) -> {ok,State}
%% @doc Start the server with specified name, host, and port.
start_link(Serv,Host,Port) ->
    gen_server:start_link({local,Serv},?MODULE,[Serv,Host,Port],[]).

%% @spec () -> {ok,State}
%% @doc Stop the default server.
stop() ->
    gen_server:cast(?SERVER,stop).

%% @spec (Serv::term()) -> {ok,State}
%% @doc Stop the named server.
stop(Serv) ->
    gen_server:cast(Serv,stop).

%% @spec (Serv::term(),CmdStr::string()) -> {ok,State}
%% @doc Send a complete command expressed as a string via the named server.
raw_command(Serv,CmdStr) ->
    gen_server:call(Serv,{update,CmdStr}).

%% @spec (CmdStr::string()) -> {ok,State}
%% @doc Send a complete command expressed as a string via the default server.
raw_command(CmdStr) ->
    raw_command(?SERVER,CmdStr).

%% @spec () -> {noreply,State}
%% @doc Tell default server to print state to output.
spill() ->
    gen_server:cast(?SERVER,spill).

%% @spec (Serv::term()) -> {noreply,State}
%% @doc Tell named server to print state to output.
spill(Serv) ->
    gen_server:cast(Serv,spill).

%%====================================================================
%% gen_server callbacks
%%====================================================================

%%--------------------------------------------------------------------
%% Function: init(Args) -> {ok, State} |
%%                         {ok, State, Timeout} |
%%                         ignore               |
%%                         {stop, Reason}
%% Description: Initiates the server
%%              opens socket to (presumptive) sdr-core
%%              on default or specified Host & Port
%%--------------------------------------------------------------------
init([Serv,Host,Port]) ->
    case gen_udp:open(0,[binary]) of
	{ok,Sock} ->
	    State = #state{serv = Serv,
			   sock = Sock,
			   host = Host,
			   port = Port},
	    {ok,State};
	{error,Reason} ->
	    {stop,Reason}
    end;

init([]) ->
    init([?SERVER,?CMD_HOST,?CMD_PORT]).

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request,From,State) -> 
%%                                      {reply, Reply,State} |
%%                                      {reply, Reply,State,Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State,Timeout} |
%%                                      {stop, Reason,Reply,State} |
%%                                      {stop, Reason,State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
%%-
%% send already-formed command
%%     ie, client knows form of command and parameters
%%     & so constructs command directly,
%%     passed on directly to port and gets simple ack
%% NB CmdStr is exactly that, an Erlang string
%%-
handle_call({update,CmdStr},_From,State) ->
    Sock = State#state.sock,
    Host = State#state.host,
    Port = State#state.port,
    Wait = State#state.wait,
    Req = CmdStr++"\n", % just in case
    Reply = ask(Req,Sock,Host,Port,Wait),
    {reply,Reply,State};

handle_call(_Request,_From,State) ->
    Reply = ok,
    {reply,Reply,State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg,State) -> {noreply, State} |
%%                                     {noreply, State, Timeout} |
%%                                     {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
%%-
%% stop: bring to a graceful conclusion
%%-
handle_cast(stop,State) ->
    {stop,normal,State};

%%-
%% spill: dump current State for casual diagnosis
%%-
handle_cast(spill,State) ->
    io:format("spill ~p~n",[State]),
    {noreply,State};

handle_cast(_Msg,State) ->
    {noreply,State}.

%%--------------------------------------------------------------------
%% Function: handle_info(Info,State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling all non call/cast messages
%%--------------------------------------------------------------------
handle_info(_Info,State) ->
    {noreply,State}.

%%--------------------------------------------------------------------
%% Function: terminate(Reason, State) -> void()
%% Description: This function is called by a gen_server when it is about to
%% terminate. It should be the opposite of Module:init/1 and do any necessary
%% cleaning up. When it returns, the gen_server terminates with Reason.
%% The return value is ignored.
%%--------------------------------------------------------------------
%%-
%% nothing to do but close up the socket
%%-
terminate(normal,State) ->
    gen_udp:close(State#state.sock),
    ok;

terminate(_Reason,_State) ->
    ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
%%-
%% we ain't there yet
%%-
code_change(_OldVsn,State,_Extra) ->
    {ok,State}.

%%--------------------------------------------------------------------
%%% Internal functions
%%--------------------------------------------------------------------
%%-
%% send a command string as datagram, wait for response
%% expects an Erlang string (list), converts to packed binary
%%-
ask(Req,Sock,Host,Port,Wait) ->
    % io:format("ask sending ~p to ~p,~p~n",[Req,Host,Port]),
    Pkt = list_to_binary(Req),
    gen_udp:send(Sock,Host,Port,[Pkt]),
    receive
    	{udp,_Serv,_Addr,_Port,<<"ok">>} ->
	    % io:format("ask gets ok from server~n",[]),
	    ok;
    	{udp,_Serv,_Addr,_Port,<<"ok",More/binary>>} ->
	    % io:format("ask gets ok+stuff from server~n",[]),
	    {ok,More};
	{udp,_Serv,_Addr,_Port,<<"error">>} ->
	    % io:format("ask gets error from server~n",[]),
	    error;
	_Other ->
	    % io:format("ask sees Other ~p~n",[_Other]),
	    error
    after
	Wait ->
	    {error,timeout}
    end.
