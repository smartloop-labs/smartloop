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

## MCPAgent detail — "vector retrieves, LLM picks"

```mermaid
flowchart LR
    subgraph REG["MCP registration (side path — non-blocking)"]
        direction TB
        ADD["add_local/remote_mcp_server<br/>persist + return (status=pending)"] --> QUEUE["MCPTask queue"]
        QUEUE --> PROV["_provision_mcp_server<br/>discover → enrich (LLM) → index → crawl"]
        PROV --> IDX["MCPToolIndexer.index_server"]
        IDX --> VEC[("mcp_tools<br/>vector collection<br/>name · description · schema · tool-call")]
        PROV --> CRAWL["MCPDocumentIndexer.crawl_server<br/>(crawl plan: list → detail → fan-out)"]
        CRAWL --> DOCS[("mcp_documents<br/>vector collection<br/>crawled pages · issues · cards")]
        CYCLE["MCPTask refresh cycle<br/>(start + every slp_mcp_crawl_interval_s)"] --> CRAWL
        DEL["delete_mcp_server"] -->|remove_server| VEC
        DEL -->|remove_server| DOCS
    end

    subgraph RESOLVE["MCP step resolution (_resolve_tool_args)"]
        direction TB
        STEP["MCP TOOL step<br/>(instruction)"] --> Q["MCPToolIndexer.search<br/>(top-K candidate tools)"]
        Q --> PICK["MCPServerSelector<br/>LLM picks 1 of K"]
        PICK --> ARGS["MCPArgumentExtractor<br/>resolve arguments"]
        ARGS --> CALL["MCPToolCaller.call_tool"]
        CALL --> RES["result → LLM for the step"]
    end

    VEC -. "candidate tools" .-> Q
    DOCS -. "crawled content (Explorer.collect_parallel)" .-> STEP
```

## Notes

- **The loop:** both `RouteNode` (from the prompt) and a matched skill in
  `SkillMatchNode` add agent step nodes via each node's `_extend_dag` result,
  so a skill invoking an MCP is the same code path as a direct trigger.
- **MCPAgent** only performs *selection*; the existing resolve→execute handlers
  call the tool and feed the result back (mirrors `FetchAgent`).
- **Registration never blocks**: `add_*_mcp_server` persists the connection and
  returns `status="pending"`. Tool discovery (a process launch or a remote
  handshake), LLM enrichment, tool indexing and the first content crawl all run
  on `MCPTask`, which records `ready`/`failed` + `last_error` for clients to poll.
- **Document-type servers are crawled, not just called**: a crawl plan
  (`smartloop/mcp/crawl_plans.py`) walks a Notion/Linear/Trello connection's
  listing tools — with per-record detail calls and parent→child fan-out — and
  vectorizes the results into `mcp_documents`. `Explorer.collect_parallel`
  searches that collection alongside documents/workspace/feedback, so the
  content is in context without depending on the router picking the right live
  call. The cycle re-crawls at startup and on an interval; hashes in
  `.mcp_documents_manifest.json` keep a refresh to what actually changed.
- **WebSearchAgent** is a stub — the search backend is still to be defined.
- `DAGExecutor` runs nodes with satisfied dependencies in parallel; the single
  local llama.cpp model instance serializes `LLMDecisionNode`s via a lock.
