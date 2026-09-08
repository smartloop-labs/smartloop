# Agent Flow (front-of-DAG agents)

How a user prompt is routed through the front-of-DAG agents
(SkillAgent, WebSearchAgent, MCPAgent, FetchAgent) and executed as DAG steps.

## High-level flow

```mermaid
flowchart TD
    U([User prompt]) --> EX["Orchestrator.execute()"]
    EX --> PLAN["plan() — seed DAG"]

    subgraph DAG["DAG (DAGExecutor: parallel, LLM nodes serialized)"]
        direction TB
        SM["SkillMatchNode<br/>= SkillAgent"]
        CO["CollectNode<br/>vector / workspace / feedback search"]
        RT["RouteNode<br/>planner: classify + plan"]
        SM --> CO --> RT

        RT -->|"_extend_dag (the loop)"| STEPS
        SM -. "matched skill can spawn<br/>agent steps (loop)" .-> STEPS

        subgraph STEPS["Dynamically-added step nodes (by StepType)"]
            direction TB
            FE["FETCH → FetchAgent<br/>URL → markdown"]
            WS["WEB_SEARCH → WebSearchAgent<br/>latest info/events (stub)"]
            MC["TOOL → MCPAgent<br/>resolve + execute MCP tool"]
            OT["EXPLORE / ANALYZE /<br/>CONVERSATIONAL"]
        end

        STEPS --> RE["REASON (TransformNode)<br/>aggregate step results"]
    end

    RE --> ANS([Final answer to LLM / user])

    %% triggers
    U -. "direct trigger" .-> STEPS
```

## MCPAgent detail — "discovery is the catalog, the model resolves the call"

```mermaid
flowchart LR
    subgraph REG["MCP registration (side path — non-blocking)"]
        direction TB
        ADD["add_local/remote_mcp_server<br/>persist + return (status=pending)"] --> QUEUE["MCPTask queue"]
        QUEUE --> PROV["_provision_mcp_server<br/>discover → enrich (LLM)"]
        PROV --> TOOLS[("server.tools (config.json)<br/>name · description · input_schema")]
    end

    subgraph RESOLVE["MCP step resolution (resolve_tool_args)"]
        direction TB
        STEP["MCP TOOL step<br/>(instruction + prior results)"] --> RESV["MCPAgent.resolve<br/>one call over every discovered tool<br/>thinking=True, 1024-token budget"]
        RESV --> CHOSE["{tool, arguments}<br/>required params filled"]
        CHOSE -->|"complete"| CALL["MCPToolCaller.call_tool"]
        CHOSE -->|"partial"| ARGS["MCPArgumentExtractor<br/>second pass for the gaps"]
        ARGS --> CALL
        CALL --> RES["result → LLM for the step"]
    end

    TOOLS -. "tool signatures" .-> RESV
```

## Notes

- **The loop:** both `RouteNode` (from the prompt) and a matched skill in
  `SkillMatchNode` add agent step nodes via each node's `_extend_dag` result,
  so a skill invoking an MCP is the same code path as a direct trigger.
- **MCPAgent** performs *resolution*: one call over the tools discovery already
  recorded, answering with the tool and its required arguments. There is no
  vector index over tool descriptions — the discovered list is the catalog —
  and the call asks for reasoning (`thinking=True`) with an output budget big
  enough to hold the block, on a model declaring `ModelCapability.THINKING`
  when the workspace has one (see `ModelRouter.reasoning_llm` and
  `smartloop/agents/mcp/mcp_agent.py`). The resolve→execute handlers still call
  the tool and feed the result back (mirrors `FetchAgent`), and both emit the
  call — `Resolved notion.notion-search(query='…')` — on the progress stream.
- **Registration never blocks**: `add_*_mcp_server` persists the connection and
  returns `status="pending"`. Tool discovery (a process launch or a remote
  handshake) and LLM enrichment run on `MCPTask`, which records
  `ready`/`failed` + `last_error` for clients to poll. Nothing is indexed:
  discovery's tool list is the catalog, so a connection is usable the moment it
  is discovered.
- **A connected server is a source, asked live**: `Explorer.collect_parallel`
  resolves and runs a read-only tool for the question alongside documents,
  workspace files, feedback and the web. Content used to be crawled and
  vectorized into `mcp_documents` ahead of time, and the live call was skipped
  whenever that index matched; that is gone, so an answer about a Notion page
  comes from Notion rather than from a snapshot.
- **WebSearchAgent** is a stub — the search backend is still to be defined.
- `DAGExecutor` runs nodes with satisfied dependencies in parallel; the single
  local llama.cpp model instance serializes `LLMDecisionNode`s via a lock.
