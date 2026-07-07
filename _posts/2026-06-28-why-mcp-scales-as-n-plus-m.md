---
description: "A short, picture-first explanation of why MCP turns integration from multiplication into addition, and where MCP sits in the protocol stack."
---

# Why MCP? N+M, not N&times;M

*MCP series — 1. Why MCP? (you're here) · 2. [Key MCP Primitives](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html) · 3. MCP Transports (coming soon)*

**Objective:** A short, picture-first explanation of why the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) turns the integration problem from multiplication into addition, and where MCP actually sits in the protocol stack.

## The integration math

An *integration* is the adapter that lets one app talk to one tool. Without a shared standard, every app has to build a custom adapter for every tool it wants to reach. Three apps each wiring to three tools means 3 &times; 3 = 9 adapters &mdash; and it grows multiplicatively, so adding one new tool means new work in *every* app.

MCP replaces those bespoke adapters with one common protocol. Each app implements MCP once (it becomes a *client*) and each tool implements MCP once (it becomes a *server*). Any client can then talk to any server through the shared contract, so you build N + M integrations instead of N &times; M. Adding a new tool is now one MCP server, not a patch to every app.

![MCP turns an N-by-M grid of point-to-point integrations into an N-plus-M hub-and-spoke](/images/mcp/mcp-n-plus-m-vs-n-times-m.svg)

That is the whole win in one line: MCP converts a grid of point-to-point connections into hub-and-spoke, and multiplication into addition.

## Is it over HTTP?

Sometimes &mdash; but HTTP is just one of two transports MCP can ride on. MCP is deliberately layered. At the top it defines *what* is exchanged (tools, resources, prompts). Every interaction is encoded as a [**JSON-RPC 2.0**](https://www.jsonrpc.org/specification) message, and that message format is the real heart of MCP &mdash; it doesn't care how the bytes are delivered.

Underneath JSON-RPC sit two standard transports:

- **stdio** &mdash; for local servers. The host launches the server as a subprocess and they exchange JSON-RPC over stdin/stdout pipes. No HTTP, no ports. This is the common setup for desktop tools.
- **Streamable HTTP** &mdash; for remote servers. The client sends JSON-RPC over HTTP POST, and the server can stream responses back using Server-Sent Events (SSE), all over normal TCP+TLS. (This consolidated the older "HTTP + SSE" transport from earlier versions of the spec.)

![The MCP protocol stack: MCP primitives over JSON-RPC 2.0 over either stdio or Streamable HTTP](/images/mcp/mcp-protocol-stack.svg)

So "is MCP over HTTP?" depends on the server: a local server you point a desktop app at is almost certainly talking over stdio pipes, while a hosted server talks over Streamable HTTP. The protocol semantics are identical either way &mdash; only the bottom transport layer changes.
