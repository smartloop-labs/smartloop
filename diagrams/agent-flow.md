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
    subgraph REG["MCP registration (side path)"]
        direction TB
        ADD["add_local/remote_mcp_server"] --> IDX["MCPToolIndexer.index_server"]
        IDX --> VEC[("mcp_tools<br/>vector collection<br/>name · description · schema · tool-call")]
        DEL["delete_mcp_server"] -->|remove_server| VEC
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
```

## Notes

- **The loop:** both `RouteNode` (from the prompt) and a matched skill in
  `SkillMatchNode` add agent step nodes via each node's `_extend_dag` result,
  so a skill invoking an MCP is the same code path as a direct trigger.
- **MCPAgent** only performs *selection*; the existing resolve→execute handlers
  call the tool and feed the result back (mirrors `FetchAgent`).
- **WebSearchAgent** is a stub — the search backend is still to be defined.
- `DAGExecutor` runs nodes with satisfied dependencies in parallel; the single
  local llama.cpp model instance serializes `LLMDecisionNode`s via a lock.
