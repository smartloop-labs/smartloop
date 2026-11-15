#!/usr/bin/env bash
# Sums up an SLP_TIMING log: one row per chat turn with where its time
# went, the models loaded for it and how often, then the models that
# loaded more than once. Reads the default log, or FILE if given; with
# REQ, also lists every line of that request from every process.
#
#   scripts/timing-summary.sh
#   scripts/timing-summary.sh ~/.smartloop/logs/slp-timing.jsonl <req>
set -euo pipefail

FILE=${1:-${SLP_HOME:-$HOME/.smartloop}/logs/slp-timing.jsonl}
REQ=${2:-}

jq -rs '
  def s: if . == null then "-" else ((. / 1000 * 100 | round) / 100 | tostring) + "s" end;
  def p($name): .phases[$name].ms | s;
  (map(select(.ev == "phase" and .name == "model.load" and .req != null)) | group_by(.req)
    | map({key: .[0].req, value: length}) | from_entries) as $loads
  | (["ended", "req", "total", "1st tok", "attach", "index", "orch", "announce", "max ctx",
      "prep", "stream", "cites", "unload", "llm calls", "mcp", "loads"] | @tsv),
    (.[] | select(.ev == "request.end" and .kind == "chat")
      | [(.ts / 1e6 | floor | strflocaltime("%H:%M:%S")), .req, (.ms | s), (.marks.first_token | s),
         p("chat.attachments"), p("chat.index_wait"), p("chat.orchestrate"), p("chat.announce_model"),
         p("chat.max_context"), p("chat.preparing"), p("chat.streaming"), p("chat.citations"),
         p("chat.unload"),
         ([.phases | to_entries[] | select(.key | startswith("llm.")) | "\(.key[4:])×\(.value.n)"]
           | if length == 0 then "-" else join(" ") end),
         (.phases["mcp.call"].n // 0), ($loads[.req] // 0)]
      | map(tostring) | @tsv)
' "$FILE" | column -t -s $'\t'

echo
echo "Models loaded more than once:"
jq -rs '
  map(select(.ev == "phase" and .name == "model.load"))
  | group_by(.model // .slot)
  | map(select(length > 1))
  | if length == 0 then "  none" else
      .[] | "  \(.[0].model // .[0].slot)  \(length) loads in \(map(.req) | unique | length) request(s), \((map(.ms) | add / 1000 * 10 | round) / 10)s in all"
    end
' "$FILE"

if [ -n "$REQ" ]; then
  echo
  echo "Request $REQ:"
  jq -rs --arg req "$REQ" '
    map(select(.req == $req)) | sort_by(.ts)
    | (.[0].ts // 0) as $t0
    | .[]
    | "\(((.ts - $t0) / 1000) | floor)ms\t\(.proc)\t\(.ev)\t\(.name // "-")\t\(if .ms then "\(.ms)ms" else "-" end)\t\(del(.ts, .pid, .proc, .req, .ev, .name, .ms, .phases, .marks, .counts) | tostring)"
  ' "$FILE" | column -t -s $'\t'
fi
