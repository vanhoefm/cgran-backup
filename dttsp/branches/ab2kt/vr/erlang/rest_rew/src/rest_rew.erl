%%%-------------------------------------------------------------------
%%% File    : grest_rew.erl
%%% Author  : Frank Brickle <brickle@lambda>
%%% Description : 
%%%
%%% Created : Dec 2008 by Frank Brickle <brickle@lambda>
%%%-------------------------------------------------------------------
-module(rest_rew).

-vsn("1").
-author('brickle@pobox.com').
-purpose("Extremely simple RESTful interface example.").
-copyright("FSF").
-license("GPLv3").

%% API
-export([arg_rewrite/1]).

-include("/usr/local/lib/yaws/include/yaws.hrl").
-include("/usr/local/lib/yaws/include/yaws_api.hrl").

%%====================================================================
%% API
%%====================================================================
%%--------------------------------------------------------------------
%% Function: arg_rewrite(Arg) -> NewArg
%% Description: Tiny RESTful interface via arg rewrite
%% (This version forces remote execution to
%%  what would be the default anyway.)
%%--------------------------------------------------------------------

arg_rewrite(Arg) ->
    Req = Arg#arg.req,
    Opq = Arg#arg.opaque,
    case Req#http_request.path of
	{abs_path,URL} ->
	    case (catch yaws_api:url_decode_q_split(URL)) of
		{'EXIT',_} -> % borked; punt
		    Arg;
		{Path,Query} ->
		    Sure   = ensure_yaws(Path),
		    NewURL = maybe_query(Sure,Query),
		    NewReq = Req#http_request{path = {abs_path,NewURL}},
		    NewOpq = reroute_execute(Opq),
		    Arg#arg{req    = NewReq,
			    opaque = NewOpq}
	    end;
	_ ->
	    Arg
    end.

%%====================================================================
%% Internal functions
%%====================================================================

ensure_yaws(Path) ->
    case string:rstr(Path,".yaws") of
	0 -> Path++".yaws";
	_ -> Path
    end.

maybe_query(Path,Query) ->
    case Query of
	[] -> Path;	    
	_  -> Path++"?"++Query
    end.

reroute_execute(Opq) ->
    Bare = proplists:delete(where,Opq),
    [{where,{node(),grape,expression}}|Bare].

