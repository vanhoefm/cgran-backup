%%%-------------------------------------------------------------------
%%% File        : vrk.erl
%%% Author      : Frank Brickle <brickle@pobox.com>
%%% Copyright   : FSF 2008
%%% License     : GPLv3
%%% Description : launcher for simplest implementation
%%%               that can possibly work
%%%               of a vr kernel
%%% Created     : Dec 2008 by Frank Brickle <brickle@pobox.com>
%%%
%%% TODO        : * associated vrmeme
%%%               * multiple virtual yaws servers 
%%%               * QA
%%%
%%% @author Frank Brickle <brickle@pobox.com>
%%% @copyright 2008 by the Free Software Foundation, license GPLv3
%%% @doc Launch and control the simplest VR-Kernel that can possibly work.
%%%      Handles individual named instances of this server
%%%      and the embedded yaws server.
%%% @end
%%% @version 1.0.0
%%% @TODO <p>associated vrmeme</p>
%%%       <p>multiple virtual yaws servers</p>
%%%       <p>QA</p>
%%% @end
%%%-------------------------------------------------------------------
-module(vrk).

-vsn("1").
-author('brickle@pobox.com').
-purpose("Launch simplest vr kernel that can possibly work.").
-copyright("FSF").
-license("GPLv3").

-behavior(gen_server).

%% API directly concerning server operation
-export([start_link/0,
	 start_link/1,
	 stop_yaws/0,
	 stop_yaws/1,
	 get_yaws_parms/0,
	 get_yaws_parms/1,
	 set_yaws_parms/1,
	 set_yaws_parms/2]).

%% conveniences, shortcuts, etc.
-export([execute/2,
	 decode_as_float/1,
	 lkup/3,
	 replace_elem/3,
	 http_meth/1,
	 dump/1]).

%% gen_server callbacks
-export([init/1,
	 handle_call/3,
	 handle_cast/2,
	 handle_info/2,
	 terminate/2,
	 code_change/3]).

-define(SERVER,?MODULE).

-define(DFLT_YSRV,"localhost").
-define(DFLT_PORT,18464).
-define(DFLT_ROOT,"tmp/vrk").
-define(DFLT_LOGS,"tmp/logs").
-define(DFLT_APPM,[]).
%-define(DFLT_REWM,yaws).
-define(DFLT_REWM,rest_rew).
-define(DFLT_OPAQ,[]).
-define(DFLT_EXEC,{node(),?MODULE,dump}).

-include("/usr/local/lib/yaws/include/yaws.hrl").
-include("/usr/local/lib/yaws/include/yaws_api.hrl").

-record(vconf,{serv = ?SERVER,
	       ysrv = ?DFLT_YSRV,
	       port = ?DFLT_PORT,
	       root = ?DFLT_ROOT,
	       appm = ?DFLT_APPM,
	       rewm = ?DFLT_REWM,
	       opaq = ?DFLT_OPAQ}).

-record(state,
	{vc = #vconf{},
	 gc = undefined,
	 sc = undefined}).

%%====================================================================
%% API
%%====================================================================
%%--------------------------------------------------------------------
%% Function: start_link() -> {ok,Pid} | ignore | {error,Error}
%% Description: Starts the server
%%--------------------------------------------------------------------
%% @spec () -> {ok,State}
%% @doc Launch the server with all default settings.
start_link() ->
    gen_server:start_link({local,?SERVER},?MODULE,[],[]).

%% @spec (List::list()) -> {ok,State}
%%
%% @doc Launch the server with all settings spelled out.
%% <p>Vname is name for this vrk instance.</p>
%% <p>Yname is name for embedded yaws instance.</p>
%% <p>Port is which port yaws listens on.</p>
%% <p>Appm is a list of appmods passed to yaws.
%%    These are the basic resource handlers.</p>
%% @end
start_link([Vname,Yname,Port,Root,Appm,Rewm,Opaq]) ->
    gen_server:start_link({local, Vname},
			  ?MODULE,
			  [Vname,Yname,Port,Root,Appm,Rewm,Opaq],
			  []);

%% debugging & testing only
%% for example
%% vrk:start_link([{"resource",showarg}]).
start_link(Appm) ->
    gen_server:start_link({local, ?SERVER},
			  ?MODULE,
			  [?SERVER,
			   ?DFLT_YSRV,
			   ?DFLT_PORT,
			   ?DFLT_ROOT,
			   Appm,
			   ?DFLT_REWM,
			   ?DFLT_OPAQ],
			  []).

%% @spec () -> {reply,Reply,State}
%% @doc Halt the default embedded yaws server.
stop_yaws() ->
    stop_yaws(?SERVER).

%% @spec (Vname::atom()) -> {reply,Reply,State}
%% @doc Halt the named embedded yaws server.
stop_yaws(Vname) ->
    gen_server:call(Vname,stop).

%% @spec (Vname::atom()) -> {reply,Reply,State}
%% @doc Halt the named embedded yaws server.
get_yaws_parms(Vname) ->
    gen_server:call(Vname,get_yaws_parms).

%% @spec () -> {reply,Reply,State}
%% @doc Return the parameters of the default embedded yaws server.
get_yaws_parms() ->
    get_yaws_parms(?SERVER).

%% @spec (Vname::term(),
%%        Parms::term()) -> {reply,Reply,State}
%% @doc Reconfigure the named server with supplied parms.
set_yaws_parms(Vname,Parms) ->
    gen_server:call(Vname,{set_yaws_parms,Parms}).

%% @spec (Parms::term()) -> {reply,Reply,State}
%% @doc Reconfigure the default server with supplied parms.
set_yaws_parms(Parms) ->
    set_yaws_parms(?SERVER,Parms).

%%-
%% Dispatch execution for this particular Arg
%% (other obvious default is {node(),ducpin,raw_comand}
%%-
%% @spec (Arg::term(),Data::term()) -> term()
%% @doc Indirect execution, mostly from an out/1.
%%      NMF is taken from opaque list in Arg,
%%      or uses default hardwired here (!)
%% @end
execute(Arg,Data) ->
    {Node,Mod,Fun} = lkup(where,
			  Arg#arg.opaque,
			  ?DFLT_EXEC),
    rpc:call(Node,Mod,Fun,[Data]).

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
init([Vname,Yname,Port,Root,Appm,Rewm,Opaq]) ->
    Id = "vrk_id",
    Debug = false,
    ok = application:load(yaws),
    ok = application:set_env(yaws,embedded,true),
    ok = application:set_env(yaws,id,Id),
    application:start(yaws),
    DGC = yaws_config:make_default_gconf(Debug,Id),
    GC = DGC#gconf{logdir = ?DFLT_LOGS},
    yaws:mkdir(GC#gconf.logdir),
    SC = #sconf{port            = Port,
		servername      = Yname,
		listen          = {0,0,0,0},
		appmods         = Appm, % eg [{"resource",Module},...]
		docroot         = Root,
		arg_rewrite_mod = Rewm,
		opaque          = Opaq},
    _Res = yaws_api:setconf(GC,[[SC]]),
    VC = #vconf{serv = Vname,	% name of *this* vrk server
		ysrv = Yname,	% name of yaws server
		port = Port,	% yaws server port
		root = Root,	% yaws server docroot
		appm = Appm,	% list of appmods
		rewm = Rewm,	% rewrite module
		opaq = Opaq},	% opaque data
    State = #state{vc = VC,	% vrk conf
		   gc = GC,	% yaws server gconf
		   sc = SC},	% yaws server sconf
%%  error_logger:info_msg("vrk yaws setconf ~p with state ~p~n",[_Res,State]),
    {ok,State};

init([]) ->
    init([?SERVER,
	  ?DFLT_YSRV,
	  ?DFLT_PORT,
	  ?DFLT_ROOT,
	  ?DFLT_APPM,
	  ?DFLT_REWM,
	  ?DFLT_OPAQ]).

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) -> {reply, Reply, State} |
%%                                      {reply, Reply, State, Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
handle_call(stop,_From,State) ->
%%  error_logger:info_msg("vrk stopping yaws ~p~n",[]),
    application:stop(yaws),
    {reply,ok,State};

handle_call(get_yaws_parms,_From,State) ->
    VC = State#state.vc,
    Reply = [VC#vconf.root,
	     VC#vconf.appm,
	     VC#vconf.rewm,
	     VC#vconf.opaq],
    {reply,Reply,State};

%%-
%% Reconfigure the server
%% NB only Root,Appm,Rewm,Opaq can be modified
%%-
handle_call({set_yaws_parms,Parms},_From,State) ->
    case Parms of
	[Root,Appm,Rewm,Opaq] ->
	    SC = State#state.sc,
	    VC = State#state.vc,
	    NewSC = SC#sconf{appmods         = Appm,
			     docroot         = Root,
			     arg_rewrite_mod = Rewm,
			     opaque          = Opaq},
	    Reply = gen_server:call(yaws_server,{update_sconf,NewSC}),
%	    io:format("Reply from update is ~p~n",[Reply]),
	    NewVC = VC#vconf{root = Root,
			     appm = Appm,
			     rewm = Rewm,
			     opaq = Opaq},
	    NewState = State#state{vc = NewVC,
				   sc = NewSC},
	    {reply,Reply,NewState};
	_ ->
	    {reply,{bardarg,Parms},State}
    end;

handle_call(_Request,_From,State) ->
    {reply,ok,State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
handle_cast(_Msg,State) ->
    {noreply,State}.

%%--------------------------------------------------------------------
%% Function: handle_info(Info, State) -> {noreply, State} |
%%                                       {noreply, State, Timeout} |
%%                                       {stop, Reason, State}
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
terminate(_Reason,_State) ->
    ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn,State,_Extra) ->
    {ok,State}.

%%--------------------------------------------------------------------
%%% Internal and/or convenience functions
%%--------------------------------------------------------------------

%%-
%% Erlang is finicky about turning digits into a float
%%-
%% @spec (Digits::string()) -> float() | {badarg,Digits}
%% @doc Convert string of digits to float.
%%      Gives back a float even if string represents an integer.
%% @end
decode_as_float(Digits) ->
    case io_lib:fread("~f",Digits) of
	{ok,[Float],_} ->
	    Float;
	_ ->
	    case io_lib:fread("~d",Digits) of
		{ok,[Int],[]} ->
		    float(Int);
		_ ->
		    exit({badarg,Digits})
	    end
    end.

%%-
%% lifted wholesale from yaws.erl
%%-
%% @spec (Key::term(),
%%        List::list(),
%%        Def::term()) -> term()
%% @doc Does no-fail keysearch
lkup(Key,List,Def) ->
    case lists:keysearch(Key,1,List) of
        {value,{_,Value}} -> Value;
        _                 -> Def
    end.

%%-
%% Replace all occurrences of Elem in List with Item
%%   mostly to service rewrites and/or appmods
%%-
%% @spec (List::list(),
%%        Elem::term(),
%%        Item::term()) -> list()
%% @doc Replace all occurrences of Elem in List with Item.
%%      Mostly to service rewrites and/or appmods.
%% @end
replace_elem([],_Elem,_Item) -> [];
replace_elem(List,_Elem,[])  -> List;
replace_elem(List,Elem,Item) ->
    replace_elem(List,Elem,Item,[]).

replace_elem([],_Elem,_Item,Acc) ->
    lists:reverse(Acc);

replace_elem([H|T],Elem,Item,Acc) when H == Elem ->
    replace_elem(T,Elem,Item,[Item|Acc]);

replace_elem([H|T],Elem,Item,Acc) ->
    replace_elem(T,Elem,Item,[H|Acc]).

%%-
%% save some typing and editing screen real estate
%%-
%% @spec (Arg::term()) -> term()
%% @doc Pull out http request method from an Arg.
%%      Save some typing and editing screen real estate.
%% @end
http_meth(Arg) ->
    (Arg#arg.req)#http_request.method.

dump(Data) ->
    io:format("*** ~p~n",[Data]).
