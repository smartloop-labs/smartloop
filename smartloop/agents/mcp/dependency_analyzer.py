"""
MCP Dependency Analyzer Agent - Analyzes tool dependencies and determines execution order.

Ported from core/core/agents/mcp/dependency_analyzer.py.
Pure logic with minimal LLM — examines input schemas to identify ID dependencies
and finds prerequisite list tools to resolve them.
"""
import logging
import re
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DependencyInfo(BaseModel):
    """Information about a tool dependency."""
    param_name: str = Field(description="Name of the parameter that has a dependency")
    param_type: str = Field(description="Type of the parameter")
    description: str = Field(description="Description of the parameter")
    prerequisite_tool: Optional[str] = Field(default=None, description="Name of the tool to execute first")
    extraction_field: Optional[str] = Field(default=None, description="Field name to extract from prerequisite result")
    reason: str = Field(default="", description="Reason for the dependency")


class ToolDependencyAnalysis(BaseModel):
    """Result of dependency analysis for a tool."""
    has_dependencies: bool = Field(description="Whether the tool has dependencies that need resolution")
    dependencies: List[DependencyInfo] = Field(default_factory=list, description="List of identified dependencies")
    execution_order: List[str] = Field(default_factory=list, description="Ordered list of tools to execute")
    can_execute: bool = Field(description="Whether the tool can be executed with current information")
    missing_info: List[str] = Field(default_factory=list, description="Parameters that cannot be resolved")


# Common ID parameter patterns
ID_PARAM_PATTERNS = [
    r'.*[iI][dD]$',
    r'.*[iI][dD]s$',
    r'^id$',
    r'.*_id$',
    r'.*_ids$',
]


class MCPDependencyAnalyzer:
    """
    Analyzes MCP tool input schemas to identify dependencies
    and determines the execution order for chained tool calls.
    """

    def _is_id_parameter(self, param_name: str, param_info: Dict[str, Any] = None) -> bool:
        """Check if a parameter indicates it needs an ID/reference value."""
        for pattern in ID_PARAM_PATTERNS:
            if re.match(pattern, param_name, re.IGNORECASE):
                return True

        if param_info:
            description = param_info.get('description', '').lower()
            if any(word in description for word in ['identifier', 'uuid', 'id of', 'unique id', ' id', 'team id', 'project id']):
                return True

        return False

    def _extract_entity_from_param(self, param_name: str) -> Optional[str]:
        """
        Extract the entity name from a parameter.
        E.g., 'teamId' -> 'team', 'issueIds' -> 'issue', 'project_id' -> 'project'
        """
        entity = param_name
        suffixes = ['Id', 'Ids', 'ID', 'IDs', '_id', '_ids', 'id', 'ids']
        for suffix in suffixes:
            if entity.endswith(suffix) and len(entity) > len(suffix):
                entity = entity[:-len(suffix)]
                break

        entity = entity.replace('_', '').replace('-', '').lower()
        return entity if entity else param_name.lower()

    def _find_list_tool_for_entity(self, entity: str, available_tools: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find a tool that can list/get the given entity type.
        Uses deterministic pattern matching.
        """
        entity_lower = entity.lower()
        entity_plural = entity_lower + 's' if not entity_lower.endswith('s') else entity_lower

        tool_names = [t.get('name', '') for t in available_tools]
        logger.info(f"Looking for list tool for entity '{entity}'. Available tools: {tool_names}")

        # Priority 1: Exact patterns
        exact_patterns = [
            f'list_{entity_plural}', f'list-{entity_plural}',
            f'list_{entity_lower}', f'list-{entity_lower}',
            f'get_{entity_plural}', f'get-{entity_plural}',
            f'list{entity_plural}', f'get{entity_plural}',
        ]

        for tool in available_tools:
            tool_name = tool.get('name', '')
            if tool_name.lower() in exact_patterns:
                logger.info(f"Found exact match list tool '{tool_name}' for entity '{entity}'")
                return tool_name

        # Priority 2: Tools starting with list/get that contain entity and have no required params
        for tool in available_tools:
            tool_name = tool.get('name', '')
            tool_name_lower = tool_name.lower()
            input_schema = tool.get('input_schema', {})
            required_params = input_schema.get('required', []) if input_schema else []

            has_list_prefix = tool_name_lower.startswith(('list', 'get'))
            contains_entity = entity_lower in tool_name_lower or entity_plural in tool_name_lower
            no_required_params = len(required_params) == 0

            is_mutation = any(word in tool_name_lower for word in ['create', 'update', 'delete', 'remove', 'add', 'set'])

            if has_list_prefix and contains_entity and no_required_params and not is_mutation:
                logger.info(f"Found list tool '{tool_name}' for entity '{entity}' (prefix match)")
                return tool_name

        # Priority 3: Fallback - any tool with entity name and list-like characteristics
        for tool in available_tools:
            tool_name = tool.get('name', '')
            tool_name_lower = tool_name.lower()
            tool_desc = tool.get('description', '').lower()
            input_schema = tool.get('input_schema', {})
            required_params = input_schema.get('required', []) if input_schema else []

            contains_entity = entity_lower in tool_name_lower or entity_plural in tool_name_lower
            is_mutation = any(word in tool_name_lower for word in ['create', 'update', 'delete', 'remove', 'add', 'set'])
            desc_suggests_list = any(word in tool_desc for word in ['list', 'get all', 'fetch all', 'returns all', 'retrieve'])

            if contains_entity and not is_mutation and (desc_suggests_list or len(required_params) == 0):
                logger.info(f"Found list tool '{tool_name}' for entity '{entity}' (fallback match)")
                return tool_name

        logger.warning(f"No list tool found for entity '{entity}' in available tools: {tool_names}")
        return None

    def invoke(
        self,
        user_input: str,
        tool_name: str,
        input_schema: Dict[str, Any],
        available_tools: List[Dict[str, Any]],
    ) -> ToolDependencyAnalysis:
        """
        Analyze the tool's input schema to identify dependencies and determine execution order.

        Args:
            user_input: The user's message/request
            tool_name: Name of the tool to analyze
            input_schema: JSON Schema describing the tool's input parameters
            available_tools: All available tools from the same server

        Returns:
            ToolDependencyAnalysis with dependency information and execution order
        """
        try:
            required_params = input_schema.get('required', [])
            properties = input_schema.get('properties', {})

            if not required_params:
                return ToolDependencyAnalysis(
                    has_dependencies=False,
                    dependencies=[],
                    execution_order=[tool_name],
                    can_execute=True,
                    missing_info=[]
                )

            dependencies = []
            prerequisite_tools = []

            logger.info(f"Checking {len(required_params)} required params for dependencies: {required_params}")

            for param_name in required_params:
                param_info = properties.get(param_name, {})
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '')

                is_id_param = self._is_id_parameter(param_name, param_info)

                if is_id_param:
                    has_id_in_input = self._check_for_id_in_context(param_name, user_input)

                    if not has_id_in_input:
                        entity = self._extract_entity_from_param(param_name)

                        if entity:
                            prerequisite_tool = self._find_list_tool_for_entity(entity, available_tools)

                            if prerequisite_tool:
                                dependencies.append(DependencyInfo(
                                    param_name=param_name,
                                    param_type=param_type,
                                    description=param_desc,
                                    prerequisite_tool=prerequisite_tool,
                                    extraction_field='id',
                                    reason=f"Need to list {entity}s to get the {param_name}"
                                ))

                                if prerequisite_tool not in prerequisite_tools:
                                    prerequisite_tools.append(prerequisite_tool)
                            else:
                                dependencies.append(DependencyInfo(
                                    param_name=param_name,
                                    param_type=param_type,
                                    description=param_desc,
                                    prerequisite_tool=None,
                                    extraction_field=None,
                                    reason=f"No list tool found for {entity}"
                                ))

            execution_order = prerequisite_tools + [tool_name]

            missing_info = [
                d.param_name for d in dependencies
                if d.prerequisite_tool is None
            ]
            can_execute = len(missing_info) == 0
            has_dependencies = len(dependencies) > 0

            result = ToolDependencyAnalysis(
                has_dependencies=has_dependencies,
                dependencies=dependencies,
                execution_order=execution_order,
                can_execute=can_execute,
                missing_info=missing_info
            )

            logger.info(f"Dependency analysis for '{tool_name}': "
                       f"dependencies={len(dependencies)}, "
                       f"execution_order={execution_order}, "
                       f"can_execute={can_execute}")

            return result

        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}")
            return ToolDependencyAnalysis(
                has_dependencies=False,
                dependencies=[],
                execution_order=[tool_name],
                can_execute=True,
                missing_info=[]
            )

    def _check_for_id_in_context(self, param_name: str, user_input: str) -> bool:
        """Check if an ID value for the parameter exists in user input."""
        id_patterns = [
            r'[A-Z]{2,10}-\d+',
            r'#\d+',
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            r'[0-9a-fA-F]{24,64}',
        ]

        for pattern in id_patterns:
            if re.search(pattern, user_input):
                logger.info(f"Found ID pattern in user input for {param_name}")
                return True

        return False
