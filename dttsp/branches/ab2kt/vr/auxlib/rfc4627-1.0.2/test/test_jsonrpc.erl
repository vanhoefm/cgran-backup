%% An example JSON-RPC service
%%---------------------------------------------------------------------------
%% Copyright (c) 2007 Tony Garnock-Jones <tonyg@kcbbs.gen.nz>
%% Copyright (c) 2007 LShift Ltd. <query@lshift.net>
%%
%% Permission is hereby granted, free of charge, to any person
%% obtaining a copy of this software and associated documentation
%% files (the "Software"), to deal in the Software without
%% restriction, including without limitation the rights to use, copy,
%% modify, merge, publish, distribute, sublicense, and/or sell copies
%% of the Software, and to permit persons to whom the Software is
%% furnished to do so, subject to the following conditions:
%%
%% The above copyright notice and this permission notice shall be
%% included in all copies or substantial portions of the Software.
%%
%% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
%% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
%% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
%% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
%% BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
%% ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
%% CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%% SOFTWARE.
%%---------------------------------------------------------------------------

-module(test_jsonrpc).

-include("rfc4627.hrl").
-include("mod_jsonrpc.hrl").

-behaviour(gen_server).

-export([start/0, start_httpd/0]).
-export([init/1, terminate/2, code_change/3, handle_call/3, handle_cast/2, handle_info/2]).

start() ->
    {ok, Pid} = gen_server:start(?MODULE, [], []),
    mod_jsonrpc:register_service
      (Pid,
       mod_jsonrpc:service(<<"test">>,
			   <<"urn:uuid:afe1b4b5-23b0-4964-a74a-9168535c96b2">>,
			   <<"1.0">>,
			   [#service_proc{name = <<"test_proc">>,
					  idempotent = true,
					  params = [#service_proc_param{name = <<"value">>,
									type = <<"str">>}]}])).

start_httpd() ->
    httpd:start("test/server_root/conf/httpd.conf"),
    mod_jsonrpc:start(),
    start().

%---------------------------------------------------------------------------

init(_Args) ->
    {ok, no_state}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    State.

handle_call({jsonrpc, <<"test_proc">>, _ModData, [Value]}, _From, State) ->
    {reply, {result, <<"ErlangServer: ", Value/binary>>}, State}.

handle_cast(Request, State) ->
    error_logger:error_msg("Unhandled cast in test_jsonrpc: ~p", [Request]),
    {noreply, State}.

handle_info(Info, State) ->
    error_logger:error_msg("Unhandled info in test_jsonrpc: ~p", [Info]),
    {noreply, State}.
