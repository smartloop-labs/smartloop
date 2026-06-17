"""
MCP Argument Extractor Agent - Extracts tool arguments from user input based on input schema.

Ported from core/core/agents/mcp/argument_extractor.py with simplified dependencies
(no conversation summary, no content extraction from attachments).
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.utils.json import parse_json_markdown

from smartloop.llms import BaseLLM

logger = logging.getLogger(__name__)


# URL pattern for identifying URL-type parameters
_URL_PARAM_KEYWORDS = ('url', 'uri', 'link', 'href', 'endpoint')

# Regex to extract http/https URLs from free text
_URL_REGEX = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

# Patterns for identifying ID-type parameters
ID_PARAM_PATTERNS = [
    r'.*[iI][dD]$',           # ends with "id" or "Id" (e.g., issueId, userId)
    r'.*[iI][dD]s$',          # ends with "ids" or "Ids" (e.g., issueIds)
    r'^id$',                   # exactly "id"
    r'.*_id$',                 # ends with "_id" (e.g., issue_id)
    r'.*_ids$',                # ends with "_ids"
    r'.*[uU][uU][iI][dD]$',   # ends with "uuid" or "UUID"
]

# Cursor/pagination parameters that are server-generated and must never be fabricated
CURSOR_PARAM_PATTERNS = [
    r'^after$',           # GraphQL forward cursor (e.g., Linear searchIssues)
    r'^before$',          # GraphQL backward cursor
    r'^cursor$',          # generic cursor
    r'.*[cC]ursor$',      # ends with "cursor" or "Cursor"
    r'^pageToken$',       # Google API page token
    r'^nextToken$',       # AWS-style next page token
    r'^continuationToken$',
]

# Common identifier patterns in user messages
IDENTIFIER_PATTERNS = [
    r'[A-Z]{2,10}-\d+',                          # JIRA/Linear style: LIN-123, PROJ-456
    r'#\d+',                                      # GitHub style: #123
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',  # UUID with dashes
    r'[0-9a-fA-F]{24,64}',                       # Hex IDs (MongoDB ObjectId, Linear IDs, etc.)
]


def normalize_uuid(value: str) -> str:
    """Normalize a hex string to UUID format with dashes if it's 32 hex chars."""
    clean = value.replace('-', '').lower()
    if len(clean) == 32 and all(c in '0123456789abcdef' for c in clean):
        return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"
    return value


class MCPArgumentExtractor:
    """
    Agent that extracts tool arguments from user input based on the tool's input schema.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _is_query_parameter(self, param_name: str, param_info: Dict[str, Any]) -> bool:
        """Check if a parameter is a search/query parameter that should be filled directly from user input."""
        name_lower = param_name.lower()
        if name_lower in ('query', 'search', 'search_query', 'q'):
            return True
        desc = param_info.get('description', '').lower()
        return any(kw in desc for kw in ('search query', 'search term', 'query string'))

    def _is_url_parameter(self, param_name: str, param_info: Dict[str, Any]) -> bool:
        """Check if a parameter expects a URL value."""
        name_lower = param_name.lower()
        if any(kw in name_lower for kw in _URL_PARAM_KEYWORDS):
            return True
        fmt = param_info.get('format', '').lower()
        if fmt in ('uri', 'url', 'iri'):
            return True
        desc = param_info.get('description', '').lower()
        return any(kw in desc for kw in ('url', 'uri', 'web address', 'link to'))

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract all http/https URLs from user text."""
        return _URL_REGEX.findall(text)

    def _is_id_parameter(self, param_name: str, param_info: Dict[str, Any]) -> bool:
        """Check if a parameter is an ID-type parameter based on name and description."""
        for pattern in ID_PARAM_PATTERNS:
            if re.match(pattern, param_name):
                return True

        description = param_info.get('description', '').lower()
        id_keywords = ['identifier', 'uuid', 'unique id', 'object id', 'reference']
        if any(keyword in description for keyword in id_keywords):
            return True

        return False

    def _is_cursor_parameter(self, param_name: str) -> bool:
        """Check if a parameter is a cursor/pagination token that must not be fabricated."""
        for pattern in CURSOR_PARAM_PATTERNS:
            if re.match(pattern, param_name):
                return True
        return False

    def _extract_identifiers_from_text(self, text: str) -> List[str]:
        """Extract potential identifiers from user text."""
        identifiers = []
        for pattern in IDENTIFIER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            identifiers.extend(matches)
        return identifiers


    def _is_valid_id_value(self, value: Any) -> bool:
        """Check if a value looks like a valid identifier."""
        if not isinstance(value, str):
            return False

        value = value.strip()
        if ' ' in value:
            return False

        for pattern in IDENTIFIER_PATTERNS:
            if re.fullmatch(pattern, value, re.IGNORECASE):
                return True

        if re.fullmatch(r'^[0-9a-fA-F]{16,64}$', value):
            return True

        if re.fullmatch(r'^[a-zA-Z0-9_-]{8,64}$', value):
            common_words = ['the', 'and', 'for', 'with', 'this', 'that', 'create', 'update',
                          'delete', 'list', 'get', 'add', 'show', 'find', 'search', 'make',
                          'issue', 'task', 'comment', 'team', 'user', 'project', 'assign',
                          'me', 'my', 'i', 'you', 'it', 'we', 'they', 'all', 'new', 'bug', 'fix']
            value_lower = value.lower()
            if value_lower in common_words:
                return False
            if len(value) > 20 and any(word in value_lower for word in common_words):
                return False
            return True

        return False

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Lowercase, collapse whitespace, drop trailing punctuation."""
        return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".!? ")

    def _is_schema_echo(self, value: Any, param_info: Dict[str, Any]) -> bool:
        """True when the extracted value just echoes the field's schema description.

        Small models frequently copy a parameter's description/guidance text
        (e.g. "Content as Markdown. Do not escape the string ...") into the
        value instead of producing a real value.  Such values are not real
        data and must be dropped so the param is treated as missing.

        Short values (e.g. ticker symbols, codes) are never considered echoes
        regardless of whether they appear in the description as examples.
        """
        if not isinstance(value, str):
            return False
        desc = param_info.get("description") or ""
        if not desc:
            return False
        nv, nd = self._normalize_text(value), self._normalize_text(desc)
        # Short values are almost certainly real data, not echoed descriptions.
        # A ticker like "AAPL" or a country code "US" should never be dropped.
        if len(nv) < 20:
            return False
        # Require a substantial description to avoid dropping legit short values.
        if len(nd) < 20:
            return False
        # Only treat as echo when the value matches the whole description or
        # contains most of it — not the reverse (value appearing as an example
        # in the description is fine).
        if nv == nd or nd in nv:
            logger.warning(
                "Dropping echoed schema description as value: %r", value[:80]
            )
            return True
        return False

    def _parse_json(self, text: str) -> Any:
        """Parse JSON from LLM response text, handling markdown fences.

        Strips fences with any language tag (``tool_code``, ``json``, etc.)
        before parsing, since langchain's parser only handles ``json``.
        """
        stripped = re.sub(r'```\w*\s*', '', text).strip().rstrip('`')
        return parse_json_markdown(stripped)

    def _build_previous_results_context(self, previous_tool_results: List[Dict[str, Any]]) -> str:
        """Build a formatted context string from previous tool execution results."""
        if not previous_tool_results:
            return ""

        context_parts = ["\nPREVIOUS TOOL RESULTS:"]

        for idx, tool_result in enumerate(previous_tool_results, 1):
            tool_name = tool_result.get('tool_name', 'unknown')
            result_data = tool_result.get('result', {})

            context_parts.append(f"\n{idx}. Tool: {tool_name}")

            if isinstance(result_data, dict):
                extracted_info = self._extract_key_fields(result_data)
                if extracted_info:
                    context_parts.append(f"   Data: {json.dumps(extracted_info, indent=2)}")
                else:
                    result_str = json.dumps(result_data, indent=2)
                    if len(result_str) < 500:
                        context_parts.append(f"   Result: {result_str}")
                    else:
                        context_parts.append(f"   Result: (Large result - {len(str(result_data))} chars)")
            elif isinstance(result_data, list):
                context_parts.append(f"   Items count: {len(result_data)}")
                for item_idx, item in enumerate(result_data[:3], 1):
                    extracted = self._extract_key_fields(item) if isinstance(item, dict) else item
                    context_parts.append(f"   Item {item_idx}: {json.dumps(extracted, indent=2) if extracted else str(item)[:100]}")
            else:
                result_str = str(result_data)
                if len(result_str) < 200:
                    context_parts.append(f"   Result: {result_str}")
                else:
                    context_parts.append(f"   Result: {result_str[:200]}...")

        return "\n".join(context_parts)

    def _extract_key_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields from a data dictionary (IDs, names, titles, etc.)"""
        if not isinstance(data, dict):
            return {}

        key_fields = {}
        priority_fields = [
            'id', '_id', 'teamId', 'issueId', 'projectId', 'userId', 'commentId',
            'identifier', 'key', 'number',
            'name', 'title', 'label', 'description'
        ]

        for field in priority_fields:
            if field in data:
                key_fields[field] = data[field]

        for key, value in data.items():
            if key not in key_fields and self._is_id_parameter(key, {}):
                key_fields[key] = value

        return key_fields

    def invoke(
        self,
        user_input: str,
        tool_name: str,
        input_schema: Dict[str, Any],
        previous_tool_results: List[Dict[str, Any]] = None,
        llm: Optional[BaseLLM] = None,
    ) -> Dict[str, Any]:
        """
        Extract tool arguments from user input based on the tool's input schema.

        Args:
            user_input: The user's message/request
            tool_name: Name of the tool being executed
            input_schema: JSON Schema describing the tool's input parameters
            previous_tool_results: Results from previously executed tools (for chained calls)

        Returns:
            Dictionary of extracted arguments. If required ID parameters are missing,
            the dict will contain a '_missing_params' key with the list of missing params.
        """
        try:
            parameters_desc = self._build_parameters_description(input_schema)

            if not parameters_desc:
                logger.info(f"Tool '{tool_name}' has no input parameters")
                return {}

            effective_llm = llm or self.llm

            required_params = input_schema.get('required', [])

            # Build previous tool results context (for chained calls)
            previous_results_context = ""
            if previous_tool_results:
                previous_results_context = self._build_previous_results_context(previous_tool_results)

            chained_instructions = ""
            if previous_results_context:
                chained_instructions = """
CHAINED TOOL CALL INSTRUCTIONS:
- Previous step results are provided — use them to infer parameter values
- Extract values from anywhere in the input: instructions, data, or original request
- If a value is clearly stated or derivable from context, use it
"""

            required_params = input_schema.get('required', [])
            properties = input_schema.get('properties', {})

            # schema.required is the source of truth. When the schema
            # declares required params, only those are extracted. When
            # required is empty (common with MCP servers like Linear),
            # fall back to non-ID properties — IDs are server-generated
            # and the LLM should never fabricate them.
            if required_params:
                accepted_params = required_params
            else:
                accepted_params = [
                    p for p in properties.keys()
                    if not self._is_id_parameter(p, properties.get(p, {}))
                    and not self._is_cursor_parameter(p)
                ]
            allowed_keys = ", ".join(f'"{p}"' for p in accepted_params)
            required_list = ", ".join(required_params) if required_params else "None"

            prompt = f"""\
<system>
You are a precise argument extraction assistant.
Given a tool schema and user input, extract parameter values as a flat JSON object.
Use "MISSING" for any required value that cannot be determined.
Do NOT invent or fabricate values — only use what is stated or derivable from the user input.
The text after each parameter is a DESCRIPTION of the field — it is guidance, never a value. NEVER copy a parameter's description text as its value.
For parameters where no value can be derived from the user input, omit them entirely.
Respond with ONLY raw JSON. No markdown fences, no explanation.
</system>
<user>
Extract arguments for tool '{tool_name}'.

The JSON response MUST only use these keys: {allowed_keys}

Tool Parameters:
{parameters_desc}

Required: {required_list}
{chained_instructions}
Input:
{user_input}
Results from steps for context in chained calls:
{previous_results_context}
Respond with raw JSON only, no markdown.
</user>"""

            prompt_tokens = effective_llm.get_num_tokens(prompt)
            logger.info("Argument extraction prompt (%d tokens) for tool '%s'", prompt_tokens, tool_name)

            response = effective_llm.invoke(prompt)

            if isinstance(response, dict):
                response_text = response.get('text', str(response))
            else:
                response_text = str(response) if response else '{}'

            response_tokens = effective_llm.get_num_tokens(response_text)
            logger.info("Argument extraction response (%d tokens): %s", response_tokens, response_text[:200])

            try:
                arguments = self._parse_json(response_text)

                if not isinstance(arguments, dict):
                    logger.warning(f"Response is not a dict: {type(arguments)}")
                    return {}

                # Keep only accepted parameters (required when declared,
                # all properties when required is empty). Filter out
                # null/empty/fabricated values.
                filtered_arguments = {
                    k: v for k, v in arguments.items()
                    if k in accepted_params
                    and v is not None and v != "" and v != "null" and v != [] and v != "MISSING"
                    and not isinstance(v, (dict, list))
                    and not self._is_schema_echo(v, properties.get(k, {}))
                }

                # Only flag params as missing if they are actually required
                missing_from_llm = [
                    k for k, v in arguments.items()
                    if v == "MISSING" and k in required_params
                ]

                # Ensure all required parameters are present
                filtered_arguments, missing_params = self._ensure_required_params(
                    filtered_arguments, input_schema, user_input
                )

                all_missing = list(set(missing_from_llm + missing_params))

                if all_missing:
                    logger.warning(f"Missing required parameters for '{tool_name}': {all_missing}")
                    filtered_arguments['_missing_params'] = all_missing

                logger.info(f"Extracted arguments for '{tool_name}': {filtered_arguments}")
                return filtered_arguments

            except Exception as e:
                logger.error(f"Failed to parse arguments: {e}")
                return {}

        except Exception as e:
            logger.error(f"Argument extraction failed: {str(e)}")
            return {}

    def _ensure_required_params(
        self,
        arguments: Dict[str, Any],
        input_schema: Dict[str, Any],
        user_input: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Ensure all required parameters are present in arguments.

        Every required parameter is treated uniformly — schema.required is
        the sole source of truth.  Missing string params fall back to the
        user input; other types fall back to schema defaults.
        """
        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})
        missing_params: List[str] = []

        for param_name in required_params:
            param_info = properties.get(param_name, {})
            param_type = param_info.get('type', 'string')

            current_value = arguments.get(param_name)
            is_missing = current_value is None or current_value == ""

            if is_missing:
                default_value = param_info.get('default')

                if default_value is None:
                    if param_type == 'string':
                        # For string params without a default, flag as missing
                        # so the caller can decide what to do.
                        missing_params.append(param_name)
                        continue
                    elif param_type == 'integer':
                        default_value = 0
                    elif param_type == 'number':
                        default_value = 0.0
                    elif param_type == 'boolean':
                        default_value = False
                    elif param_type == 'array':
                        default_value = []
                    elif param_type == 'object':
                        default_value = {}

                if default_value is not None:
                    arguments[param_name] = default_value

        return arguments, missing_params

    def _build_parameters_description(self, input_schema: Dict[str, Any]) -> str:
        """Build a human-readable description of parameters from JSON Schema.

        Only describes accepted parameters (required when declared, all
        properties when required is empty). This prevents the LLM from
        seeing and fabricating values for optional params like 'id'.
        """
        if not input_schema or 'properties' not in input_schema:
            return ""

        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])

        if not properties:
            return ""

        # Only show the params we actually want extracted — matches the
        # accepted_params logic in invoke().
        if required:
            accepted_params = required
        else:
            accepted_params = [
                p for p in properties.keys()
                if not self._is_id_parameter(p, properties.get(p, {}))
                and not self._is_cursor_parameter(p)
            ]

        descs = []
        for param_name in accepted_params:
            param_info = properties.get(param_name, {})
            param_type = param_info.get('type', 'any')
            param_desc = param_info.get('description', 'No description')

            param_str = f"- {param_name} ({param_type}): {param_desc}"

            if 'enum' in param_info:
                param_str += f"\n  Allowed values: {', '.join(str(v) for v in param_info['enum'])}"
            if 'default' in param_info:
                param_str += f"\n  Default: {param_info['default']}"

            descs.append(param_str)

        if required:
            return "REQUIRED parameters (must always be provided):\n" + "\n".join(descs)
        return "Parameters:\n" + "\n".join(descs)

    def invoke_with_resolved_ids(
        self,
        user_input: str,
        tool_name: str,
        input_schema: Dict[str, Any],
        resolved_ids: Dict[str, str],
        llm: Optional[BaseLLM] = None,
    ) -> Dict[str, Any]:
        """
        Extract arguments with pre-resolved IDs.
        Only uses LLM for non-ID parameters.

        Args:
            user_input: The user's message/request
            tool_name: Name of the tool being executed
            input_schema: JSON Schema describing the tool's input parameters
            resolved_ids: Pre-resolved ID parameters from MCPResultMapper

        Returns:
            Dictionary of extracted arguments (resolved_ids + LLM-extracted content)
        """
        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})
        if required_params:
            accepted_params = required_params
        else:
            accepted_params = [
                p for p in properties.keys()
                if not self._is_id_parameter(p, properties.get(p, {}))
                and not self._is_cursor_parameter(p)
            ]

        arguments = dict(resolved_ids)
        logger.info(f"invoke_with_resolved_ids: Starting with pre-resolved: {list(resolved_ids.keys())}")

        # Find accepted params that still need extraction (not already resolved)
        remaining = [
            p for p in accepted_params if p not in resolved_ids
        ]

        if remaining:
            logger.info(f"invoke_with_resolved_ids: Extracting remaining params via LLM: {remaining}")
            llm_extracted = self._extract_non_id_params(
                user_input=user_input,
                non_id_params=remaining,
                properties=properties,
                llm=llm,
            )
            arguments.update(llm_extracted)
        else:
            logger.info("invoke_with_resolved_ids: All params already resolved")

        # Flag missing required params (only when schema declares them)
        if required_params:
            missing_params = [
                p for p in required_params
                if p not in arguments or arguments[p] is None or arguments[p] == ""
            ]
            if missing_params:
                arguments['_missing_params'] = missing_params

        logger.info(f"invoke_with_resolved_ids: Final arguments for '{tool_name}': {list(arguments.keys())}")
        return arguments

    def _extract_non_id_params(
        self,
        user_input: str,
        non_id_params: List[str],
        properties: Dict[str, Any],
        llm: Optional[BaseLLM] = None,
    ) -> Dict[str, Any]:
        """Extract only non-ID parameters using LLM."""
        if not non_id_params:
            return {}

        param_descriptions = []
        for param_name in non_id_params:
            param_info = properties.get(param_name, {})
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', 'No description')
            param_str = f"- {param_name} ({param_type}): {param_desc}"

            if 'enum' in param_info:
                param_str += f"\n  Allowed values: {', '.join(str(v) for v in param_info['enum'])}"
            if 'default' in param_info:
                param_str += f"\n  Default: {param_info['default']}"

            param_descriptions.append(param_str)

        params_desc = "\n".join(param_descriptions)

        prompt = f"""\
<system>
You are a precise parameter extraction assistant.
Extract values for the requested parameters from the user input.
Do NOT make up values that aren't present in the input.
The text after each parameter is a DESCRIPTION of the field — never copy a parameter's description text as its value.
Respond with ONLY raw JSON. No markdown fences, no explanation.
</system>
<user>
Extract values for the following parameters from the input.

Input: {user_input}

Parameters to extract:
{params_desc}

Respond with raw JSON only, no markdown.
</user>"""

        try:
            effective_llm = llm or self.llm
            prompt_tokens = effective_llm.get_num_tokens(prompt)
            logger.info("Content extraction prompt (%d tokens)", prompt_tokens)

            response = effective_llm.invoke(prompt)

            if isinstance(response, dict):
                response_text = response.get('text', str(response))
            else:
                response_text = str(response) if response else '{}'

            response_tokens = effective_llm.get_num_tokens(response_text)
            logger.info("Content extraction response (%d tokens): %s", response_tokens, response_text[:200])

            extracted = self._parse_json(response_text)

            if not isinstance(extracted, dict):
                return {}

            result = {
                k: v for k, v in extracted.items()
                if k in non_id_params and v is not None and v != ""
                and not self._is_schema_echo(v, properties.get(k, {}))
            }

            logger.info(f"Extracted non-ID params: {list(result.keys())}")
            return result

        except Exception as e:
            logger.error(f"Non-ID param extraction failed: {e}")
            return {}
