"""
MCP Result Mapper - Creates structured ID maps from list tool results for direct lookup.

Ported from core/core/agents/mcp/result_mapper.py. Pure utility, no LLM.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MCPResultMapper:
    """
    Maps results from list tools to enable direct ID lookup for dependent tools.
    """

    ID_FIELDS = [
        'id', '_id', 'uuid', 'identifier', 'key', 'number', 'uid', 'guid',
        'Id', 'ID', 'UUID', 'resourceId', 'objectId', 'entityId'
    ]

    NAME_FIELDS = [
        'name', 'title', 'label', 'displayName', 'display_name', 'key',
        'identifier', 'slug', 'handle', 'username', 'email', 'Name', 'Title'
    ]

    def __init__(self):
        self._entity_maps: Dict[str, Dict[str, str]] = {}
        self._first_ids: Dict[str, str] = {}
        self._entity_items: Dict[str, List[Dict[str, Any]]] = {}

    def extract_entity_from_tool_name(self, tool_name: str) -> Optional[str]:
        """
        Extract the entity type from a tool name.
        E.g., 'list-teams' -> 'team', 'getProjects' -> 'project'
        """
        tool_name_lower = tool_name.lower()

        prefixes = [
            'list-', 'list_', 'list', 'get-', 'get_', 'get',
            'search-', 'search_', 'search', 'find-', 'find_', 'find',
            'fetch-', 'fetch_', 'fetch', 'query-', 'query_', 'query',
            'update-', 'update_', 'update',
            'create-', 'create_', 'create',
            'add-', 'add_', 'add',
            'delete-', 'delete_', 'delete',
            'remove-', 'remove_', 'remove',
            'edit-', 'edit_', 'edit',
            'modify-', 'modify_', 'modify',
        ]

        for prefix in prefixes:
            if tool_name_lower.startswith(prefix):
                entity = tool_name_lower[len(prefix):]
                if entity.endswith('s') and len(entity) > 1:
                    entity = entity[:-1]
                return entity

        return None

    def extract_param_entity(self, param_name: str) -> Optional[str]:
        """
        Extract entity type from a parameter name.
        E.g., 'teamId' -> 'team', 'issue_id' -> 'issue'
        """
        param_lower = param_name.lower()
        suffixes = ['id', 'ids', '_id', '_ids', 'uuid']

        for suffix in suffixes:
            if param_lower.endswith(suffix) and len(param_lower) > len(suffix):
                entity = param_lower[:-len(suffix)]
                entity = entity.replace('_', '').replace('-', '')
                return entity if entity else None

        entity = param_lower.replace('_', '').replace('-', '')
        return entity if entity else None

    def process_list_result(
        self,
        tool_name: str,
        result: Any,
        entity_override: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Process a list tool result and build the entity map.

        Args:
            tool_name: Name of the tool that produced this result
            result: The result data from the tool
            entity_override: Optional entity type override

        Returns:
            Dict mapping names to IDs for this entity type
        """
        entity_type = entity_override or self.extract_entity_from_tool_name(tool_name)

        if not entity_type:
            logger.warning(f"Could not determine entity type from tool: {tool_name}")
            return {}

        logger.info(f"Processing result for {tool_name}, result type: {type(result).__name__}")

        items = self._extract_items_from_result(result)

        if not items:
            logger.warning(f"No items found in result for {tool_name}. Raw result: {str(result)[:500]}")
            return {}

        entity_map: Dict[str, str] = {}
        first_id = None

        for item in items:
            if not isinstance(item, dict):
                continue

            item_id = self._extract_id(item)
            item_name = self._extract_name(item)

            if item_id:
                if first_id is None:
                    first_id = item_id

                if item_name:
                    entity_map[item_name.lower()] = item_id
                    entity_map[item_name] = item_id

                entity_map[item_id] = item_id

        self._entity_maps[entity_type] = entity_map
        self._entity_items[entity_type] = items

        if first_id:
            self._first_ids[entity_type] = first_id

        logger.info(f"Built entity map for '{entity_type}': {len(entity_map)} entries, first_id={first_id}")

        return entity_map

    def _extract_items_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Extract list of items from various result formats."""
        if isinstance(result, list):
            if not result:
                return []

            first = result[0]
            is_mcp_content = (isinstance(first, dict)
                              and first.get('type') == 'text'
                              and 'text' in first)

            if is_mcp_content:
                for block in result:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        try:
                            text_data = json.loads(block.get('text', '{}'))
                            nested_items = self._extract_items_from_result(text_data)
                            if nested_items:
                                return nested_items
                        except (json.JSONDecodeError, TypeError):
                            continue
                return []

            if all(isinstance(item, dict) for item in result):
                return result

            return []

        if isinstance(result, dict):
            # Handle MCP content format: {"content": [{"type": "text", "text": "..."}]}
            if 'content' in result:
                content = result['content']
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            try:
                                text_data = json.loads(block.get('text', '{}'))
                                nested_items = self._extract_items_from_result(text_data)
                                if nested_items:
                                    return nested_items
                            except (json.JSONDecodeError, TypeError):
                                pass

            # Check all fields that might contain a list of items
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        return value

                if isinstance(value, dict):
                    nested_items = self._extract_items_from_result(value)
                    if nested_items:
                        return nested_items

            if self._extract_id(result):
                return [result]

        return []

    def _extract_id(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract ID from an item dictionary."""
        for field in self.ID_FIELDS:
            if field in item:
                value = item[field]
                if value and isinstance(value, str):
                    return self._normalize_id(value)

        for key, value in item.items():
            if key.endswith('Id') or key.endswith('_id'):
                if value and isinstance(value, str):
                    return self._normalize_id(value)

        return None

    def _extract_name(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract name/label from an item dictionary."""
        for field in self.NAME_FIELDS:
            if field in item:
                value = item[field]
                if value and isinstance(value, str):
                    return value
        return None

    def _normalize_id(self, value: str) -> str:
        """Normalize an ID value — converts 32-char hex strings to UUID format."""
        clean = value.replace('-', '').lower()
        if len(clean) == 32 and all(c in '0123456789abcdef' for c in clean):
            return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"
        return value

    def get_id(
        self,
        entity_type: str,
        name_hint: Optional[str] = None,
        user_input: Optional[str] = None
    ) -> Optional[str]:
        """Get an ID for the given entity type."""
        entity_type_lower = entity_type.lower()
        entity_map = self._entity_maps.get(entity_type_lower, {})

        if name_hint:
            if name_hint in entity_map:
                return entity_map[name_hint]
            if name_hint.lower() in entity_map:
                return entity_map[name_hint.lower()]

        if user_input and entity_map:
            for name, item_id in entity_map.items():
                if name.lower() in user_input.lower():
                    logger.info(f"Found '{name}' in user input, using ID: {item_id}")
                    return item_id

        if entity_type_lower in self._first_ids:
            first_id = self._first_ids[entity_type_lower]
            logger.info(f"Using first ID for '{entity_type}': {first_id}")
            return first_id

        return None

    def get_id_for_param(
        self,
        param_name: str,
        user_input: Optional[str] = None
    ) -> Optional[str]:
        """Get an ID for a parameter based on its name."""
        entity_type = self.extract_param_entity(param_name)
        if not entity_type:
            return None
        return self.get_id(entity_type, user_input=user_input)

    def get_resolved_arguments(
        self,
        required_params: List[str],
        properties: Dict[str, Any],
        user_input: Optional[str] = None
    ) -> Dict[str, str]:
        """Get resolved arguments for ID parameters from the map."""
        resolved = {}

        for param_name in required_params:
            if self._is_id_param(param_name, properties.get(param_name, {})):
                resolved_id = self.get_id_for_param(param_name, user_input)
                if resolved_id:
                    resolved[param_name] = resolved_id
                    logger.info(f"Resolved {param_name} -> {resolved_id}")

        return resolved

    def _is_id_param(self, param_name: str, param_info: Dict[str, Any]) -> bool:
        """Check if a parameter is an ID-type parameter."""
        id_patterns = [
            r'.*[iI][dD]$',
            r'.*[iI][dD]s$',
            r'^id$',
            r'.*_id$',
            r'.*_ids$',
        ]

        for pattern in id_patterns:
            if re.match(pattern, param_name):
                return True

        description = param_info.get('description', '').lower()
        id_keywords = ['identifier', 'uuid', 'unique id', 'object id', 'id of', ' id', 'team id', 'project id']
        return any(keyword in description for keyword in id_keywords)

    def has_entity(self, entity_type: str) -> bool:
        """Check if we have data for an entity type."""
        return entity_type.lower() in self._first_ids

    def clear(self):
        """Clear all stored maps."""
        self._entity_maps.clear()
        self._first_ids.clear()
        self._entity_items.clear()
