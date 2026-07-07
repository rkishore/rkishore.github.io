---
title: "MCP Transports: stdio, SSE & Streamable HTTP"
description: "How MCP messages actually travel: stdio for local servers, the legacy HTTP+SSE transport, and the Streamable HTTP transport that replaced it — and why the switch happened."
---

*MCP series — 1. [Why MCP?](/2026/06/28/why-mcp-scales-as-n-plus-m.html) · 2. [Key MCP Primitives](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html) · 3. MCP Transports (you're here)*

**Objective:** A plain-language tour of how [Model Context Protocol (MCP)](https://modelcontextprotocol.io) messages actually move between client and server &mdash; the local **stdio** transport, the legacy **HTTP + SSE** transport, and the **Streamable HTTP** transport that replaced it &mdash; and a clear picture of why that last swap happened.

## The layer below the primitives

The [first post](/2026/06/28/why-mcp-scales-as-n-plus-m.html) drew MCP as three layers: the primitives (tools, resources, prompts) at the top, encoded as [JSON-RPC 2.0](https://www.jsonrpc.org/specification) messages in the middle, riding some *transport* at the bottom. JSON-RPC is deliberately transport-agnostic &mdash; it says nothing about *how* the bytes get delivered. That's the transport's whole job.

MCP defines two standard transports, and a third one you'll still bump into:

- **stdio** &mdash; for a server running locally as a subprocess.
- **Streamable HTTP** &mdash; for a server reachable over the network. This is the current remote transport.
- **HTTP + SSE** &mdash; the *older* remote transport, now deprecated but still out there in the wild.

The nice part: the server code above the transport doesn't change. As you'll see, picking a transport is a one-line decision.

## stdio: a subprocess and two pipes

For a server on your own machine, there's no need for HTTP at all. The host **launches the server as a subprocess** and they talk over the two pipes every process already has: the client writes JSON-RPC messages to the server's `stdin`, and reads the server's replies from its `stdout`. Messages are newline-delimited, one per line. Anything the server prints to `stderr` is just logs.

![The stdio transport: the host launches the MCP server as a subprocess and they exchange newline-delimited JSON-RPC over stdin and stdout pipes, with stderr reserved for logs.](/images/mcp/mcp-transport-stdio.svg)

On the server, this is the default &mdash; one line picks it:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("realestate")

@mcp.tool()
def schedule_viewing(address: str, when: str) -> str:
    ...

if __name__ == "__main__":
    mcp.run(transport="stdio")          # local: talk over stdin/stdout
```

The client launches that subprocess and wraps its pipes in a session:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()     # same list-then-fetch as always
```

No ports, no URLs, no network. This is what a desktop host uses when you point it at a local server.

## The old way: HTTP + SSE

Remote servers can't use pipes, so MCP needs HTTP. The first design for this &mdash; the **HTTP + SSE** transport &mdash; split the conversation across **two endpoints**:

1. The client opens a long-lived **SSE** ([Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)) stream with an HTTP `GET`. The stream's *first* event is a special `endpoint` event that tells the client which URL to post to.
2. The client sends each of its own messages as a separate HTTP `POST` to that URL.
3. Everything coming *back* &mdash; every response, every server notification &mdash; travels down the one SSE stream from step 1.

![The legacy HTTP+SSE transport: the client opens a long-lived SSE stream with GET (whose first event names the POST URL), sends each message as a separate POST, and receives every server response back down the single persistent stream.](/images/mcp/mcp-transport-http-sse.svg)

On the server, again, it's a one-line change:

```python
mcp.run(transport="sse")                # legacy remote: an SSE endpoint + a POST endpoint
```

The client just needs the SSE URL; it handles the `GET`, reads the `endpoint` event, and POSTs there for the rest of the session:

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("https://example.com/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

It works &mdash; but notice the shape: one SSE connection has to **stay open for the entire session**, because it's the only way the server can reach the client.

## Why HTTP + SSE didn't last

That always-open stream is the catch. The design has a few awkward consequences:

- **It's stateful by nature.** Each client needs its own long-lived SSE connection held open on the server for as long as the session lasts.
- **It fights modern infrastructure.** Behind a load balancer, every request from a client has to land on the exact server holding its stream &mdash; so you need sticky sessions. On serverless platforms, which don't keep long-lived connections around at all, it barely fits.
- **A drop means starting over.** If the SSE connection breaks, there's no way to pick up where it left off &mdash; the session is gone and the client reconnects from scratch.
- **Two endpoints** to route, secure, and keep in sync, instead of one.

None of these are bugs; they're just the cost of "hold one stream open forever." Fixing them meant rethinking the shape.

## The new way: Streamable HTTP

**Streamable HTTP** collapses everything onto a **single endpoint** (say, `/mcp`) that speaks both `POST` and `GET`. The key change: the server no longer *needs* a permanently open stream.

- The client sends every message as an HTTP `POST` to `/mcp`.
- For each POST, the server replies with **one of two things**: a single `application/json` response when one message is enough, or a `text/event-stream` (SSE) stream when it needs to send several messages back for that request. The server picks per request.
- If the server ever needs to push a message to the client *unprompted*, the client can open an optional `GET` stream &mdash; but for plain request/response work, it's never needed.

![The Streamable HTTP transport: the client POSTs every message to a single /mcp endpoint, and the server answers each POST with either a single JSON response or an SSE stream; an optional GET opens a server-push stream, with sessions carried by MCP-Session-Id and resumption via Last-Event-ID.](/images/mcp/mcp-transport-streamable-http.svg)

Two optional features close the gap the old transport left open:

- **Sessions.** On initialize, the server may hand back an `MCP-Session-Id` header; the client echoes it on every later request. That's what lets a stateful server recognize a returning client &mdash; without holding a socket open.
- **Resumability.** The server can tag each SSE event with an `id`. If a stream drops, the client reconnects and sends the `Last-Event-ID` header, and the server replays what was missed. No more starting from scratch.

Because a basic exchange is now just a normal POST-and-response, the same server works cleanly behind load balancers and on serverless platforms. On the server, you already know the drill:

```python
mcp.run(transport="streamable-http")    # remote: one /mcp endpoint
```

And the client swaps in the matching helper &mdash; everything above the transport is identical:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("https://example.com/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
```

Every server example in this post has been the *same* server &mdash; only the `transport=` argument changed. That's the layering from post 1 paying off: the primitives and JSON-RPC on top don't care what's underneath.

## Side by side

Laid out together, the trade is easy to see: Streamable HTTP keeps SSE around for when streaming genuinely helps, but stops *requiring* a permanent stream for everything else.

![A row-by-row comparison: HTTP+SSE uses two endpoints, an always-on stream, is stateful, has no resumption, and needs sticky sessions; Streamable HTTP uses one endpoint, streams per request, is optionally stateless, resumes via Last-Event-ID, and is serverless-friendly.](/images/mcp/mcp-transport-comparison.svg)

## Talking to older servers

The two remote transports can coexist during the changeover. A server that wants to support older clients simply keeps hosting the old SSE and POST endpoints alongside the new `/mcp` one. A client that wants to support older servers probes: it first tries to `POST` an `initialize` request to the URL &mdash; if that works, the server speaks Streamable HTTP; if it comes back with a `4xx`, the client falls back to opening a `GET` stream and waiting for the old `endpoint` event. So "which transport?" can be sorted out automatically, without the user having to know.

## The one-liner that ties it together

The transport only decides *how* the JSON-RPC bytes travel, never *what* they say: **stdio for a local subprocess, Streamable HTTP for anything remote, and HTTP + SSE as the legacy remote path you'll still meet in older servers.** Same primitives, same messages &mdash; just a different bottom layer.

And that closes the series. [Why MCP?](/2026/06/28/why-mcp-scales-as-n-plus-m.html) covered the N+M win, [Key MCP Primitives](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html) covered what the model, host, user, and server each get to drive, and this post covered the wire it all rides on.
