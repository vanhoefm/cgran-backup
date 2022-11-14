%%%-------------------------------------------------------------------
%%% File        : grape.erl
%%% Author      : Frank Brickle <brickle@lambda>
%%% Copyright   : FSF
%%% License     : GPLv3
%%% Description : GNU Radio Access Proxy in Erlang
%%%
%%% Created : Nov 2008 by Frank Brickle <brickle@lambda>
%%% @doc TCP proxy to a single GNU Radio remote instance.
%%% @end
%%% @author Frank Brickle <brickle@pobox.com>
%%% @copyright 2008 by the Free Software Foundation, license GPLv3
%%% @version 1.0.0
%%%-------------------------------------------------------------------
-module(grape).

-vsn("1").
-author('brickle@pobox.com').
-purpose("TCP proxy to a single GNU Radio remote instance").
-copyright("FSF").
-license("GPLv3").

-behavior(gen_server).

%% API
-export([start_link/0,
	 start_link/2,
	 start_link/3,
	 stop/0,
	 stop/1,
	 expression/1,
	 expression/2,
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

-define(DFLT_HOST,"localhost").
-define(DFLT_PORT,18617).
-define(TIMEOUT,1000).

-record(state,
	{sock = null,
	 serv = ?SERVER,
	 host = ?DFLT_HOST,
	 port = ?DFLT_PORT,
	 wait = ?TIMEOUT}).

%%====================================================================
%% API
%%====================================================================
%%--------------------------------------------------------------------
%% Function: start_link() -> {ok,Pid} | ignore | {error,Error}
%%           start_link(Host,Port) ->
%%                           {ok,Pid} | ignore | {error,Error}
%%           start_link(Serv,Host,Port) ->
%%                           {ok,Pid} | ignore | {error,Error}
%% Description: Starts the server,
%%              with default or stipulated name
%%              on default or stipulated Host & Port
%%--------------------------------------------------------------------
%% @spec () -> {ok,State}
%% @doc Start the server with default name and port.
start_link() ->
    gen_server:start_link({local,?SERVER},?MODULE,[],[]).

%% @spec (Host::term(),
%%        Port::integer()) -> {ok,State}
%% @doc Start the server as specified host and port.
start_link(Host,Port) ->
    gen_server:start_link({local,?SERVER},?MODULE,[?SERVER,Host,Port],[]).

%% @spec (Serv::term(),
%%        Host::term(),
%%        Port::integer()) -> {ok,State}
%% @doc Start the server under name as specified host and port.
start_link(Serv,Host,Port) ->
    gen_server:start_link({local,Serv},?MODULE,[Serv,Host,Port],[]).

%% @spec () -> {reply,Reply,State}
%% @doc Stop the default server.
stop() ->
    gen_server:cast(?SERVER,stop).

%% @spec (Serv::term()) -> {reply,Reply,State}
%% @doc Stop the named server.
stop(Serv) ->
    gen_server:cast(Serv,stop).

%% @spec (Data::term()) -> {reply,Reply,State}
%% @doc Send Data to GNU Radio exec_server for execution via default server.
expression(Data) ->
    gen_server:call(?SERVER,{expression,Data}).

%% @spec (Serv::term(),Data::term()) -> {reply,Reply,State}
%% @doc Send Data to GNU Radio exec_server for execution via named server.
expression(Serv,Data) -> 
    gen_server:call(Serv,{expression,Data}).

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
%%                         {ok, State, Wait} |
%%                         ignore               |
%%                         {stop, Reason}
%% Description: Initiates the server
%%              opens connected socket to python executive
%%              on default or specified Host & Port
%%--------------------------------------------------------------------
init([Serv,Host,Port]) ->
    case gen_tcp:connect(Host,Port,[binary,{active,false},{packet,0}]) of
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
    init([?SERVER,?DFLT_HOST,?DFLT_PORT]).

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) -> {reply, Reply, State} |
%%                                      {reply, Reply, State, Wait} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Wait} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
%%-
%% send a string capable of being exec()ed by python,
%%     ie, a complete expression or sequence of expressions
%%-
handle_call({expression,Data},_From,State) ->
    Sock = State#state.sock,
    Wait = State#state.wait,
    Reply = xsend(Sock,Data,Wait),
    {reply,Reply,State};

handle_call(_Request,_From,State) ->
    Reply = ok,
    {reply,Reply,State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Wait} |
%%                                      {stop, Reason, State}
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

handle_cast(_Msg, State) ->
    {noreply,State}.

%%--------------------------------------------------------------------
%% Function: handle_info(Info, State) -> {noreply, State} |
%%                                       {noreply, State, Wait} |
%%                                       {stop, Reason, State}
%% Description: Handling all non call/cast messages
%%--------------------------------------------------------------------
handle_info(_Info, State) ->
    {noreply, State}.

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
    gen_tcp:close(State#state.sock),
    ok;

terminate(_Reason, _State) ->
    ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%--------------------------------------------------------------------
%%% Internal functions
%%--------------------------------------------------------------------
%%-
%% send an expression over connected TCP port
%% get minimal ack/nak
%% any other results show up elsewhere
%% NB be careful to distinguish ok/error returns!
%%    some are from connection, some are from python
%%-
xsend(Sock,Data,Wait) ->
    gen_tcp:send(Sock,Data),
    case gen_tcp:recv(Sock,0,Wait) of
	{ok,<<"ok">>} ->
	    ok;
	{ok,<<"error: ",Rest/binary>>} ->
	    {error,Rest};
	{error,timeout} ->
	    {error,{error,timeout}};
	{error,closed} ->
	    {error,{error,closed}};
	_Other ->
	    error
    end.
