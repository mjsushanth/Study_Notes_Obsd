

## Stage 1: Local processes and streams

- processes, stdin stdout stder, pipes, byte vs text, 
- streams vs messages, serialization, framing.
Goal: Understand how two local programs communicate before introducing networking.

## Stage 2: Networking foundations

- IP address, port, socket, listening socket versus connected socket
- TCP connection, byte streams, why TCP does not preserve application-message boundaries
- latency, interruption, timeout, and reconnection
Goal: Understand what the operating system provides—and what it does not provide.

## Stage 3: HTTP foundations

- request and response, method, URL, headers, and body, status codes, `Content-Type`
- one HTTP request versus one underlying TCP connection
- connection reuse
- HTTP being stateless at the semantic level
- authentication and cookies as state layered over HTTP
Goal: Stop equating HTTP request, TCP connection, user session, and application transaction.

## Stage 4: Streaming techniques

- buffered response versus streaming response, chunks, SSE, WebSocket, long polling
- request-scoped streaming, long-lived subscription streams, backpressure and cancellation
Goal: Understand that “streaming” is a behavior, not a single protocol.

## Stage 5: RPC and JSON-RPC

- local function call versus remote procedure call, method, parameters, result, and error
- request IDs, notifications, JSON serialization, transport independence
Goal: Recognize MCP messages as structured RPC messages, not mysterious “AI packets.”

## Stage 6: MCP’s semantic layer

- host, client, server, tools, resources, prompts, capabilities, tool schemas
- discovery, tool invocation, results and errors
Goal: Understand MCP without caring whether it uses stdio or HTTP.

## Stage 7: MCP transports and revisions

1. MCP over stdio
2. Legacy HTTP+SSE
3. Streamable HTTP, including the 2025 sessionful and 2026 stateless variants
Goal: Understand why apparently contradictory tutorials may each describe a different revision.

## Stage 8: Production system design

- reverse proxies, load balancers, horizontal scaling, sticky sessions, shared session stores
- retries and idempotency, resumability, 
- authentication and authorization, observability, partial failures
Goal: Understand why MCP’s transport design changed—not merely what changed.


-----

### JSON RCP
1. Simply a json request file that might have `id, methods, params, tools` : such that one machine wants to talk, request, execute a function on another machine. a process on one machine wishes to execute a process on another. 
2. JSON-RPC defines requests, responses, errors, and notifications, but deliberately does not require a particular transport.
3. A transport determines how those messages travel.

### Layer stack, remote call MCP.

User or model intention
        ↓
MCP semantics
"Call the tool named add"
        ↓
JSON-RPC envelope
method + params + id
        ↓
JSON serialization
A sequence of textual bytes
        ↓
Framing
Where does one message begin and end?
        ↓
Transport
stdio or Streamable HTTP
        ↓
Operating-system/network mechanisms
pipes, sockets, TCP, TLS


| Layer     | Question                                             |
| --------- | ---------------------------------------------------- |
| MCP       | What operation does this message mean?               |
| JSON-RPC  | Is this a request, response, error, or notification? |
| JSON      | How are the data structures represented as text?     |
| Framing   | Where does this message end?                         |
| Transport | How is it delivered?                                 |
| TCP/pipe  | How are bytes moved between processes?               |

### Stream

1. Stream is just a flow of bytes. `{ "id": 1 } { "id": 2 }`
2. receiver may not receive those bytes in the same pieces in which the sender wrote them. chunks might break, including breaks at the braces level, etc. wont be clean.
3. byte stream preserves order, but it does not necessarily preserve your application’s message boundaries.
4. **framing**: rules for identifying where one message ends. 
5. example: one JSON object per line, a length prefix, HTTP’s body-length or chunking rules, SSE’s blank-line-separated events, WebSocket’s message frames. 

### Socket, Comms stdin, out, err.

1. socket is an operating-system object used by a process for network communication. **server generally creates a listening socket associated with an address and port.** 
	127.0.0.1:8000
2. operating systems then create a connected communication path.
3. client process ⇄ client socket ⇄ network ⇄ server socket ⇄ server process
4. It is a lower-level communication endpoint.
5. One socket connection can carry many HTTP requests. One logical application interaction can also survive across several socket connections. Therefore socket lifetime and application lifetime are not necessarily identical.

- stdin: bytes coming in
- stdout: normal bytes going out
- stderr: diagnostics going out

```
MCP host process                 MCP server process
      │                                  │
      ├──── requests via server stdin ──→│
      │                                  │
      │←── responses via server stdout ──┤
      │                                  │
      │←── diagnostics via stderr ───────┤
```



### HTTP is a request-response protocol

```
POST /mcp HTTP/1.1
Host: example.com
Content-Type: application/json

{"jsonrpc":"2.0","id":17,"method":"tools/list"}
```

```
HTTP/1.1 200 OK
Content-Type: application/json

{"jsonrpc":"2.0","id":17,"result":{"tools":[]}}
```


1. HTTP req response means: it would start with a method ( post or get ) so forth, a target, headers/metadata, a body - the actual payload of the request, a status - code, content type mentioned. 
2. **HTTP is defined as a stateless application-level protocol.** 
3. This means the meaning of a request does not inherently depend on a hidden HTTP conversation object. It does **not** mean servers cannot implement login sessions, shopping carts, or workflows on top of it.
4. An HTTP body is just bytes. The `Content-Type` header tells the recipient how to interpret those bytes. ( `Content-Type: application/json, Content-Type: text/event-stream`)
5. Changing the content type does not necessarily change the TCP connection or HTTP endpoint. It changes the grammar used to interpret the response body.
6. ```
	   POST /mcp
	    ↓
		server chooses response representation
		    ├── application/json       one completed response
		    └── text/event-stream      a sequence of events over time
   ```
7. The MCP operation is still `tools/call`. Only the response-delivery style differs.

### Buffered response versus streaming response


1. A normal buffered response behaves like receiving an entire package.
2. A streaming response behaves like receiving pages while the document is being produced.

```
request ───────────────→
        server works
response ←──────────────
```

```
request ───────────────→
event 1  ←──────────────
event 2  ←──────────────
event 3  ←──────────────
final    ←──────────────
close
```


### SSE versus WebSocket

Server-Sent Events is a **textual event format delivered through an HTTP response.** ( text/event-stream ).  

```
client ──opens HTTP request──→ server
client ←──events over response── server
```

WebSocket establishes a persistent, message-oriented, bidirectional channel. After setup, either side can send a WebSocket message independently. It is **standardized separately from ordinary HTTP request-response semantics.**


### Critical: why mcp benefits from HTTP ?

Because MCP generally benefits from ordinary HTTP infrastructure:
- reverse proxies already understand HTTP
- authentication standards already work with HTTP
- load balancers route HTTP well
- firewalls commonly allow HTTPS
- individual requests can be logged, traced, retried, or routed
- servers need less connection-specific machinery

WebSocket is useful, but persistent bidirectional communication brings additional connection ownership, scaling, reconnection, and routing complexity.


### Connection, Handshake, Session, State.

Connection - currently usable communication path. TCP, or pipe, or WebSocket.
Handshake - initial exchange used to establish or negotiate something. TCP handshake, TLS, HTTP exchange, MCP initialize handshake (earlier version).
Session - logical period of continuity during which some party remembers context across operations. login session, browser session, db session, model conv, MCP protocol session.
State - Information remembered over time. 

A system may use:
- a connection without application state
- state without a persistent connection
- a session identifier across many connections
- one connection containing several logical sessions

|Concept|Can exist without the others?|
|---|---|
|Connection|Yes|
|Handshake|Yes|
|Session|Yes|
|Application state|Yes|


Stateless MCP? in 2026, request carries the protocol it needs so any loadbalances + server 1, 2, 3 .. can process easily. 'sessionful design' earlier caused a routing problem. a shared session store cache or something needed; - a shared session store accessible by A and B.

The second design makes routing and scaling easier, but it pushes more responsibility into explicit request data, databases, signed tokens, or tool handles.

Era wise changes. Last one is 2026.

```
Host launches server process
        ↓
JSON-RPC MCP messages over stdin/stdout

------------------------------------------------------------------------

GET /sse
    establishes server → client event channel

POST /messages
    sends client → server messages
    
------------------------------------------------------------------------

POST /mcp
    request or response can use JSON or SSE

GET /mcp
    optional long-lived server event stream

Mcp-Session-Id
    relates requests to an MCP session
    
------------------------------------------------------------------------

POST /mcp
    self-describing request

response:
    application/json
    or
    text/event-stream

```

A request can be self-contained and its response can remain open for a long time.
New server: "I do not implement initialize; each request must describe itself." 




# The modern compromise

Most scalable systems are neither completely stateful nor completely stateless. They separate the layers:

```
Stateless computation layer
    Requests can reach any worker
             ↓
Explicit state identifier
    cart_id, user_id, job_id, browser_id
             ↓
Stateful storage or resource owner
    database, cache, stream processor, actor, game server
```

The request is explicit, but the whole world is not stuffed into its payload.

That is also the proper interpretation of modern MCP:
- MCP no longer creates an invisible protocol session for every client.
- A tool can still create a genuinely stateful resource.
- The tool returns an explicit handle.
- Later requests pass that handle back.
- The application decides how to store, route, expire, and authorize that state.