# Beyond Tools: MCP Resources, Templates &amp; Prompts

Jun 30, 2026

**Objective:** A first-person field guide to the three [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server primitives you meet *after* tools &mdash; resources, resource templates, and prompts &mdash; built around one question that made them all click for me.

I'd already built an MCP tool before any of this clicked. Tools felt obvious: the model wants to do something, it calls a function, it gets a result. But then I kept seeing "resources" and "prompts" in the spec and couldn't tell why they weren't just more tools. They *look* like functions. Why three boxes instead of one?

The thing that finally sorted it out wasn't a definition. It was a question.

## Who's in the driver's seat?

Ask, for each primitive: **who decides when this fires?** That single question splits the three cleanly.

- **Tools are model-controlled.** The LLM decides to call them, mid-conversation, based on what it's trying to accomplish. Side effects are fine &mdash; that's the point.
- **Resources are host-controlled.** The *host* (not the model) decides when to attach a resource as context. They're like GET endpoints: data the model can read, no side effects. Useful for "here's reference material the agent should always have" &mdash; the host might pull one in because the user opened a file, or because it always wants some config in context.
- **Prompts are user-controlled.** The *user* explicitly invokes them &mdash; think slash commands &mdash; and the server hands back a ready-made set of messages to seed the conversation.

![Three cards showing who controls each MCP primitive: tools are model-controlled (what the model can do), resources are host-controlled (what the model can read), prompts are user-controlled (what the user can invoke)](/images/mcp/mcp-driver-seat.svg)

Once I had the driver's seat in my head, the FastMCP decorators stopped looking redundant. They're not three flavors of the same thing &mdash; they're three different *triggers*:

```python
@mcp.tool()
def schedule_viewing(address: str, when: str) -> str:
    ...                     # model calls this; it has side effects

@mcp.resource("property://featured")
def featured() -> str:
    ...                     # host reads this; pure data, no side effects

@mcp.prompt()
def compare_homes() -> str:
    ...                     # user invokes this; returns seed messages
```

Here's a fuller resource from a seller agent in a demo real-estate agentic app &mdash; a read-only catalog of floor prices the host can drop into the system prompt, so the agent never has to *call* a tool just to know its own constraints:

```python
# ─── MCP Resource: read-only directory of seller floor prices ─────────────────
#
# Why a Resource instead of a Tool?
#   - Tools are *actions* the model chooses to invoke (with arguments).
#   - Resources are *documents* the host can attach as context up front.
#   The floor-prices catalog never takes arguments and is naturally a doc.

@mcp.resource("inventory://floor-prices")
def floor_prices_resource() -> str:
    """JSON catalog of seller floor prices for every known property.

    Hosts can fetch this resource and inject it into the system prompt
    so the seller agent never needs to call a tool just to know its own
    constraints.
    """
    catalog = {
        pid: {
            "address": data["display_address"],
            "list_price": data["list_price"],
            "minimum_acceptable_price": data["minimum_acceptable_price"],
            "ideal_price": data["ideal_price"],
            "motivation": data["seller_motivation_level"],
        }
        for pid, data in SELLER_CONSTRAINTS.items()
    }
    return json.dumps(catalog, indent=2)
```

And here's the other half &mdash; the *host* side. Notice the model isn't involved in the decision at all: the host lists resources, reads the one it wants, and bakes the contents into the system prompt before the conversation even starts.

```python
# Host-side setup (runs once, before the agent starts talking)
async with mcp_client(server) as session:
    # discovery: list, then fetch — same rails as everything else
    resources = await session.list_resources()                 # resources/list
    floor_prices = await session.read_resource("inventory://floor-prices")  # resources/read

    system_prompt = (
        "You are a seller's agent negotiating on the owner's behalf.\n"
        "Here are the floor prices you must never go below:\n"
        f"{floor_prices.contents[0].text}"
    )

# the model sees the catalog as plain context — it never called a tool to get it
agent = Agent(model="claude-opus-4-8", system_prompt=system_prompt)
```

## Same rails: list, then fetch

Here's the part that made learning the other two cheap: **all three ride the same rails.**

After the handshake &mdash; `initialize`, then the `initialized` notification, then you're ready &mdash; discovery is just *list, then fetch*. Every primitive works this way. Tools call `tools/list` to enumerate, then `tools/call` to invoke. Prompts call `prompts/list`, then `prompts/get`. There's no new protocol to learn for resources and prompts; it's the same shape with different nouns.

![After the handshake, each primitive uses a list-then-fetch pattern: tools/list to tools/call, prompts/list to prompts/get, and resources splitting across two lists feeding resources/read](/images/mcp/mcp-list-then-fetch.svg)

Resources are the one place this gets a wrinkle &mdash; and it's a wrinkle worth understanding, because it bit me.

## Resource vs. resource template

A resource comes in two shapes:

- A **plain resource** has a *fixed URI* &mdash; `property://featured`. The client reads it directly; there's nothing to fill in.
- A **resource template** has a *parameterized URI* with placeholders &mdash; `property://{address}`. The client has to supply the value, and the handler takes that value as a parameter.

```python
@mcp.resource("property://featured")
def featured() -> str:
    return load_featured()

@mcp.resource("property://{address}")     # template: {address} is a parameter
def by_address(address: str) -> str:
    return lookup(address)
```

And here's the catch that tripped me up: **they live in different discovery lists.** Plain resources show up under `resources/list`. Templates show up under `resources/templates/list`. Two lists, not one.

> **The "Method not found" trap.** I registered exactly one resource &mdash; the template `property://{address}` &mdash; then pointed my client at `resources/list` to see it. I got back `Method not found`. I assumed my server was broken.
>
> It wasn't. A template isn't a concrete resource, so it never appears under `resources/list` &mdash; it lives under `resources/templates/list`. My server had *nothing concrete* to return, so that endpoint had nothing to say. The fix was either register a plain resource too, or call `list_resource_templates()` instead.
>
> The takeaway reframed the whole split for me: the two lists aren't bureaucratic. A template is *useless* until someone fills in the parameter &mdash; `property://{address}` isn't addressable until you know the address. It's a recipe, not a dish. That's different enough from a concrete resource that it earns its own channel.

## Prompts: messages the user reaches for

Prompts are the one primitive the *user* drives directly. In a chat host they usually surface as slash commands; pick one and the server hands back a ready-made list of messages to seed the conversation. The model only sees the result *after* the user chooses &mdash; that's what makes prompts user-controlled rather than model- or host-controlled.

Server side, a prompt returns *messages* instead of a blob of data, and it can take arguments just like a tool:

```python
@mcp.prompt()
def draft_counteroffer(address: str, buyer_offer: int) -> list[Message]:
    """A ready-made negotiation opener the user can fire off."""
    return [
        UserMessage(
            f"A buyer offered ${buyer_offer:,} on {address}. "
            "Draft a firm but friendly counteroffer that stays above my floor price."
        )
    ]
```

When the user picks it &mdash; say they type `/draft_counteroffer` in the host &mdash; discovery is the same list-then-fetch you've seen all along:

```python
prompts = await session.list_prompts()      # prompts/list → populate the slash-command menu

msg = await session.get_prompt(             # prompts/get  → fill in the arguments
    "draft_counteroffer",
    arguments={"address": "12 Oak St", "buyer_offer": 720_000},
)
# msg.messages is now seeded into the conversation, exactly as the user asked
```

The contrast with the resource is the whole point: a resource is data the *host* quietly attaches up front; a prompt is a message set the *user* deliberately reaches for.

## Making that slash command real

Everything above is the *protocol*. The part that made it click for me was how little stood between that decorator and a slash command I could actually type. Claude Code reads a `.mcp.json` at the root of your project and launches any server listed there &mdash; the same stdio server, no code changes:

```json
{
  "mcpServers": {
    "realestate": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

That's the whole hookup. Reopen the project, run `/mcp` to approve the server once &mdash; project-scoped servers are gated the first time they connect &mdash; and the prompt surfaces under a name Claude Code assembles from the config key and the prompt's own name: `/mcp__realestate__draft_counteroffer`. Type it, pass the same `address` and `buyer_offer` the server declared, and the messages it returns drop straight into the conversation *as my turn* &mdash; the model only sees them because I reached for them.

![Flow of a prompt invocation across four actors: you pick the slash command and send it to the host (Claude Code), the host calls prompts/get on the server, the server returns rendered messages, the host seeds them into the conversation as your turn and sends it to the model, and the model replies. The model never fetches the prompt itself. A tool runs the other way, with the model firing the first arrow.](/images/mcp/mcp-prompt-flow.svg)

The order is the whole story: *you* fire arrow 1, the host fetches the messages from the server, and the model only meets them at the end, as your turn. A tool inverts it &mdash; the model fires the first arrow and calls `tools/call` &mdash; which is exactly why one is user-controlled and the other model-controlled.

The tools on that same server show up too, but through the other door: those the *model* calls when it decides to, while the slash command waits for *me*. One server, two audiences &mdash; the whole tools-vs-prompts split, now sitting in my editor instead of a spec.

## The one-liner that ties it together

If I could distill all of this into one sentence, it'd be this: **tools are what the model can *do*, resources are what the model can *read*, prompts are what the user can *invoke*.** Everything else &mdash; the decorators, the two lists, the handshake &mdash; hangs off that.

I could adapt an MCP *client* and feel like I understood the protocol. I didn't, really. What actually made it click was building a *server*, registering the wrong kind of resource, and watching it return `Method not found` &mdash; then figuring out why the protocol was right and I was wrong.
