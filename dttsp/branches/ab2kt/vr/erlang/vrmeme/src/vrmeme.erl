%%%-------------------------------------------------------------------
%%% File        : vrmeme.erl
%%% Author      : Frank Brickle <brickle@pobox.com>
%%% Copyright   : FSF
%%% License     : GPLv3
%%% Description : semi-persistent and persistent
%%%               memory functions for vrk and friends
%%%               store/retrieve running parm changes, etc.
%%%
%%% Created     : Dec 2008 by Frank Brickle <brickle@lambda>
%%%
%%% @author Frank Brickle <brickle@pobox.com>
%%% @copyright 2008 by the Free Software Foundation, license GPLv3
%%% @doc Semi-persistent and persistent memory functions
%%%      for vrk and friends.
%%%      Store/retrieve running parm changes, etc.
%%% @end
%%% @version 1.0.0
%%% @TODO <p>QA</p>
%%% @end
%%%-------------------------------------------------------------------
-module(vrmeme).

-vsn("1").
-author('brickle@pobox.com').
-purpose("Semi-persistent and persistent memory for vrk and friends").
-copyright("FSF").
-license("GPLv3").

-behavior(gen_server).

-define(SERVER,?MODULE).
-define(DFLT_TBL_NAME,?MODULE).
-define(DFLT_TBL_OPTS,[]).
-define(DFLT_FILE_NAME,"./vrmeme.ets").
-define(DFLT_FILE_OPTS,[]).

%% API
-export([start_link/0,
	 start_link/2,
	 start_link/3,
	 lookup/1,
	 lookup/2,
	 insert/2,
	 insert/3,
	 update/2,
	 update/3,
	 delete/1,
	 delete/2,
	 get_direct/0,
	 get_direct/1,
	 dump/2,
	 dump/1,
	 dump/0,
	 load/2,
	 load/1,
	 load/0]).

%% gen_server callbacks
-export([init/1,
	 handle_call/3,
	 handle_cast/2,
	 handle_info/2,
	 terminate/2,
	 code_change/3]).

-record(state,
	{tid       = undefined,
	 servename = ?SERVER,
	 tablename = ?DFLT_TBL_NAME,
	 tableopts = ?DFLT_TBL_OPTS,
	 filename  = ?DFLT_FILE_NAME,
	 fileopts  = ?DFLT_FILE_OPTS}).

%%====================================================================
%% API
%%====================================================================
%%--------------------------------------------------------------------
%% Function: start_link() -> {ok,Pid} | ignore | {error,Error}
%% Description: Starts the server
%%--------------------------------------------------------------------
%% @spec (Srv_name::atom(),
%%        Tbl_name::atom(),
%%        Tbl_opts::list()) -> {ok,State}
%% @doc Fire up server
start_link(Srv_name,Tbl_name,Tbl_opts) ->
    gen_server:start_link({local,Srv_name},
			  ?MODULE,
			  [Srv_name,Tbl_name,Tbl_opts],
			  []).

%% @spec (Tbl_name::atom(),
%%        Tbl_opts::list()) -> {ok,State}
%% @doc Fire up server
start_link(Tbl_name,Tbl_opts) ->
    start_link(?SERVER,Tbl_name,Tbl_opts).

%% @spec () -> {ok,State}
%% @doc Fire up server
start_link() ->
    gen_server:start_link({local,?SERVER},?MODULE,[],[]).

%% @spec (Serv::atom(),Key::term()) -> [{Key,Value}]
%% @doc Retrieve Key, Value
lookup(Serv,Key) ->
    gen_server:call(Serv,{lookup,Key}).

%% @spec (Key::term()) -> [{Key,Value}]
%% @doc Retrieve Key, Value
lookup(Key) ->
    lookup(?SERVER,Key).

%% @spec (Serv::atom(),Key::term(),Val::term()) -> true
%% @doc Add Key, Val
insert(Serv,Key,Val) ->
    gen_server:call(Serv,{insert,{Key,Val}}).

%% @spec (Key::term(),Val::term()) -> true
%% @doc Add Key, Val
insert(Key,Val) ->
    insert(?SERVER,Key,Val).

%% @spec (Serv::atom(),Key::term(),Val::term()) -> true
%% @doc New Val for Key
update(Serv,Key,Val) ->
    gen_server:call(Serv,{insert,{Key,Val}}).

%% @spec (Key::term(),Val::term()) -> true
%% @doc New Val for Key
update(Key,Val) ->
    update(?SERVER,Key,Val).

%% @spec (State::atom(),Key::term()) -> true
%% @doc Remove Key and Val
delete(Serv,Key) ->
    gen_server:call(Serv,{delete,Key}).

%% @spec (Key::term()) -> true
%% @doc Remove Key and Val
delete(Key) ->
    delete(?SERVER,Key).

%% @spec (Serv::atom()) -> Tid
%% @doc Get handle to whole table
get_direct(Serv) ->
    gen_server:call(Serv,{table,get_direct}).

%% @spec () -> Tid
%% @doc Get handle to whole table
get_direct() ->
    get_direct(?SERVER).

%% @spec (State::atom(),Name::string()) -> true
%% @doc Save table on Serv to file Name
dump(Serv,Name) ->
    gen_server:call(Serv,{table,{dump,Name}}).

%% @spec (Name::string()) -> true
%% @doc Save default table to file Name
dump(Name) ->
    dump(?SERVER,Name).

%% @spec () -> true
%% @doc Save default table to default file
dump() ->
    dump(?SERVER,?DFLT_FILE_NAME).

%% @spec (Serv::atom(),Name::string()) -> true
%% @doc Load table in file Name to server Serv
load(Serv,Name) ->
    gen_server:call(Serv,{table,{load,Name}}).

%% @spec (Name::string()) -> true
%% @doc Load table in file Name to default server
load(Name) ->
    load(?SERVER,Name).

%% @spec () -> true
%% @doc Load table in default file to default server
load() ->
    load(?SERVER,?DFLT_FILE_NAME).

%%====================================================================
%% gen_server callbacks
%%====================================================================
%%--------------------------------------------------------------------
%% Function: init(Args) -> {ok, State} |
%%                         {ok, State, Timeout} |
%%                         ignore               |
%%                         {stop, Reason}
%% Description: Initiates the server
%%--------------------------------------------------------------------
init([Srv_name,Tbl_name,Tbl_opts]) ->
    State = #state{servename = Srv_name,
		   tablename = Tbl_name,
		   tableopts = Tbl_opts},
    T = ets:new(State#state.tablename,State#state.tableopts),
    {ok,State#state{tid = T}};

init([]) ->
    State = #state{},
    T = ets:new(State#state.tablename,State#state.tableopts),
    {ok,State#state{tid = T}}.

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) ->
%%                                      {reply, Reply, State} |
%%                                      {reply, Reply, State, Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
handle_call({lookup,Key},_From,State) ->
    Table = State#state.tid,
    Reply = ets:lookup(Table,Key),
    {reply,Reply,State};

%% NB handles update also, per ets/dets API
handle_call({insert,{Key,Val}},_From,State) ->
    Table = State#state.tid,
    Reply = ets:insert(Table,{Key,Val}),
    {reply,Reply,State};

handle_call({delete,Key},_From,State) ->
    Table = State#state.tid,
    Reply = ets:delete(Table,Key),
    {reply,Reply,State};

handle_call({table,get_direct},_From,State) ->
    Reply = State#state.tid,
    {reply,Reply,State};

handle_call({table,{dump,Name}},_From,State) ->
    Table = State#state.tid,
    Reply = ets:tab2file(Table,Name),
    {reply,Reply,State};

handle_call({table,{load,Name}},_From,State) ->
    {ok,T} = ets:file2tab(Name),
    Reply = ets:delete(State#state.tid),
    {reply,Reply,State#state{tid = T}};

handle_call(_Request,_From,State) ->
    Reply = ok,
    {reply,Reply,State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
handle_cast({stop},State) ->
    {noreply,State};

handle_cast(_Msg,State) ->
    {noreply, State}.

%%--------------------------------------------------------------------
%% Function: handle_info(Info, State) -> {noreply, State} |
%%                                       {noreply, State, Timeout} |
%%                                       {stop, Reason, State}
%% Description: Handling all non call/cast messages
%%--------------------------------------------------------------------
handle_info(_Info,State) ->
    {noreply, State}.

%%--------------------------------------------------------------------
%% Function: terminate(Reason, State) -> void()
%% Description: This function is called by a gen_server when it is about to
%% terminate. It should be the opposite of Module:init/1 and do any necessary
%% cleaning up. When it returns, the gen_server terminates with Reason.
%% The return value is ignored.
%%--------------------------------------------------------------------
terminate(_Reason,_State) ->
    ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn,State,_Extra) ->
    {ok,State}.

%%--------------------------------------------------------------------
%%% Internal functions
%%--------------------------------------------------------------------
