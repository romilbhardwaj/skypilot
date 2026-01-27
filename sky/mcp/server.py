"""SkyPilot MCP Server implementation.

This module provides an MCP (Model Context Protocol) server that exposes
SkyPilot functionality to AI assistants like Claude.

The server exposes the following capabilities:
- Cluster management: status, launch, stop, down, start, autostop
- Job management: queue, cancel, tail_logs
- Managed jobs: jobs.launch, jobs.queue, jobs.cancel
- Services: serve.up, serve.down, serve.status
- Storage: storage_ls, storage_delete
- Resource discovery: list_accelerators, cost_report

Usage:
    # Run the MCP server
    python -m sky.mcp.server

    # Or via the CLI
    sky mcp start
"""

import asyncio
import json
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Union

from sky import sky_logging
from sky.adaptors import common as adaptors_common
from sky.utils import common_utils
from sky.utils import ux_utils

# Lazy imports for MCP library
mcp = adaptors_common.LazyImport('mcp')
mcp_server = adaptors_common.LazyImport('mcp.server')
mcp_stdio = adaptors_common.LazyImport('mcp.server.stdio')
mcp_types = adaptors_common.LazyImport('mcp.types')

logger = sky_logging.init_logger(__name__)

# Server name and version
SERVER_NAME = 'skypilot-mcp'
SERVER_VERSION = '1.0.0'


def _format_result(result: Any) -> str:
    """Format a result for MCP response."""
    if result is None:
        return 'Operation completed successfully.'
    if isinstance(result, (list, dict)):
        return json.dumps(result, indent=2, default=str)
    return str(result)


def _get_sdk():
    """Get the SkyPilot SDK module."""
    # Import here to avoid circular imports and ensure proper initialization
    import sky  # pylint: disable=import-outside-toplevel
    return sky


def _get_jobs_sdk():
    """Get the SkyPilot jobs SDK module."""
    from sky.jobs.client import sdk as jobs_sdk  # pylint: disable=import-outside-toplevel
    return jobs_sdk


def _get_serve_sdk():
    """Get the SkyPilot serve SDK module."""
    from sky.serve.client import sdk as serve_sdk  # pylint: disable=import-outside-toplevel
    return serve_sdk


async def _execute_sdk_call(func, *args, **kwargs) -> Any:
    """Execute an SDK call and handle the async request pattern.

    SkyPilot SDK functions return a request_id that needs to be awaited
    using sky.get() to get the actual result.
    """
    sdk = _get_sdk()
    try:
        # SDK calls return request IDs
        request_id = func(*args, **kwargs)
        # Get the actual result
        result = sdk.get(request_id)
        return result
    except Exception as e:
        logger.error(f'Error executing SDK call: {e}')
        raise


def create_server():
    """Create and configure the MCP server with all SkyPilot tools."""
    from mcp.server import Server  # pylint: disable=import-outside-toplevel
    from mcp import types  # pylint: disable=import-outside-toplevel

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list:
        """List all available SkyPilot tools."""
        return [
            # Cluster Management Tools
            types.Tool(
                name='sky_status',
                description=
                'Get the status of SkyPilot clusters. Returns information about '
                'running clusters including their status, resources, and autostop '
                'configuration.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_names': {
                            'type':
                                'array',
                            'items': {
                                'type': 'string'
                            },
                            'description':
                                'List of cluster names to query. If not provided, '
                                'returns all clusters.'
                        },
                        'refresh': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to refresh cluster status from cloud providers.'
                        },
                        'all_users': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'Whether to show clusters from all users.'
                        }
                    },
                    'required': []
                }),
            types.Tool(
                name='sky_launch',
                description=
                'Launch a SkyPilot cluster with the specified task. This creates '
                'cloud resources and runs the specified commands.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'task_yaml': {
                            'type':
                                'string',
                            'description':
                                'YAML string defining the task to run. Should include '
                                'resources, setup, and run commands.'
                        },
                        'cluster_name': {
                            'type': 'string',
                            'description':
                                'Name for the cluster. Auto-generated if not provided.'
                        },
                        'idle_minutes_to_autostop': {
                            'type':
                                'integer',
                            'description':
                                'Minutes of idle time before auto-stopping the cluster.'
                        },
                        'down': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to tear down the cluster after the job finishes.'
                        },
                        'retry_until_up': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to retry launching until successful.'
                        },
                        'dryrun': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'If true, do not actually launch the cluster.'
                        }
                    },
                    'required': ['task_yaml']
                }),
            types.Tool(
                name='sky_exec',
                description=
                'Execute a task on an existing SkyPilot cluster. The cluster must '
                'already be running.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to execute on.'
                        },
                        'task_yaml': {
                            'type': 'string',
                            'description':
                                'YAML string defining the task to execute.'
                        },
                        'dryrun': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'If true, do not actually execute the task.'
                        }
                    },
                    'required': ['cluster_name', 'task_yaml']
                }),
            types.Tool(
                name='sky_stop',
                description=
                'Stop a SkyPilot cluster. The cluster can be restarted later with '
                'sky_start. Storage is preserved.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to stop.'
                        },
                        'purge': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Force stop even if in an error state.'
                        }
                    },
                    'required': ['cluster_name']
                }),
            types.Tool(
                name='sky_start',
                description=
                'Start a stopped SkyPilot cluster. Restores the cluster to its '
                'previous state.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to start.'
                        },
                        'idle_minutes_to_autostop': {
                            'type':
                                'integer',
                            'description':
                                'Minutes of idle time before auto-stopping.'
                        },
                        'down': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to tear down instead of stop on autostop.'
                        },
                        'retry_until_up': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to retry starting until successful.'
                        }
                    },
                    'required': ['cluster_name']
                }),
            types.Tool(
                name='sky_down',
                description=
                'Tear down a SkyPilot cluster completely. This terminates all '
                'resources and deletes the cluster.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to tear down.'
                        },
                        'purge': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Force tear down even if in an error state.'
                        }
                    },
                    'required': ['cluster_name']
                }),
            types.Tool(
                name='sky_autostop',
                description=
                'Configure autostop for a cluster. The cluster will automatically '
                'stop after the specified idle time.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to configure.'
                        },
                        'idle_minutes': {
                            'type':
                                'integer',
                            'description':
                                'Minutes of idle time before stopping. Use -1 to cancel.'
                        },
                        'down': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Whether to tear down instead of stop.'
                        }
                    },
                    'required': ['cluster_name', 'idle_minutes']
                }),
            # Job Management Tools
            types.Tool(
                name='sky_queue',
                description=
                'Get the job queue for a cluster. Shows all jobs running or '
                'pending on the cluster.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster to query.'
                        },
                        'all_users': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'Whether to show jobs from all users.'
                        },
                        'skip_finished': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Whether to skip finished jobs.'
                        }
                    },
                    'required': ['cluster_name']
                }),
            types.Tool(
                name='sky_cancel',
                description='Cancel jobs on a cluster.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'cluster_name': {
                            'type': 'string',
                            'description': 'Name of the cluster.'
                        },
                        'job_ids': {
                            'type': 'array',
                            'items': {
                                'type': 'integer'
                            },
                            'description': 'List of job IDs to cancel.'
                        },
                        'all': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Whether to cancel all jobs.'
                        }
                    },
                    'required': ['cluster_name']
                }),
            # Managed Jobs Tools
            types.Tool(
                name='sky_jobs_launch',
                description=
                'Launch a managed job. Managed jobs are automatically recovered '
                'from preemptions and failures.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'task_yaml': {
                            'type': 'string',
                            'description':
                                'YAML string defining the job to launch.'
                        },
                        'name': {
                            'type': 'string',
                            'description':
                                'Name for the managed job. Auto-generated if not provided.'
                        }
                    },
                    'required': ['task_yaml']
                }),
            types.Tool(
                name='sky_jobs_queue',
                description=
                'Get the status of managed jobs. Shows all managed jobs and '
                'their current state.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'skip_finished': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Whether to skip finished jobs.'
                        },
                        'all_users': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'Whether to show jobs from all users.'
                        },
                        'job_ids': {
                            'type': 'array',
                            'items': {
                                'type': 'integer'
                            },
                            'description':
                                'List of specific job IDs to query.'
                        }
                    },
                    'required': []
                }),
            types.Tool(
                name='sky_jobs_cancel',
                description='Cancel managed jobs.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'Name of the job to cancel.'
                        },
                        'job_ids': {
                            'type': 'array',
                            'items': {
                                'type': 'integer'
                            },
                            'description': 'List of job IDs to cancel.'
                        },
                        'all': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Whether to cancel all jobs.'
                        }
                    },
                    'required': []
                }),
            # Serve Tools
            types.Tool(
                name='sky_serve_up',
                description=
                'Launch a SkyServe service. Services provide auto-scaling and '
                'load balancing for model serving.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'task_yaml': {
                            'type': 'string',
                            'description':
                                'YAML string defining the service to launch.'
                        },
                        'service_name': {
                            'type': 'string',
                            'description': 'Name for the service.'
                        }
                    },
                    'required': ['task_yaml', 'service_name']
                }),
            types.Tool(
                name='sky_serve_status',
                description=
                'Get the status of SkyServe services. Shows all services and '
                'their replicas.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'service_names': {
                            'type':
                                'array',
                            'items': {
                                'type': 'string'
                            },
                            'description':
                                'List of service names to query. If not provided, '
                                'returns all services.'
                        }
                    },
                    'required': []
                }),
            types.Tool(
                name='sky_serve_down',
                description=
                'Tear down a SkyServe service. This terminates all replicas and '
                'deletes the service.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'service_names': {
                            'type':
                                'array',
                            'items': {
                                'type': 'string'
                            },
                            'description':
                                'List of service names to tear down.'
                        },
                        'all': {
                            'type': 'boolean',
                            'default': False,
                            'description':
                                'Whether to tear down all services.'
                        },
                        'purge': {
                            'type':
                                'boolean',
                            'default':
                                False,
                            'description':
                                'Force tear down even if in an error state.'
                        }
                    },
                    'required': []
                }),
            # Storage Tools
            types.Tool(
                name='sky_storage_ls',
                description=
                'List all SkyPilot storage objects. Storage objects are used '
                'for persistent data across clusters.',
                inputSchema={
                    'type': 'object',
                    'properties': {},
                    'required': []
                }),
            types.Tool(
                name='sky_storage_delete',
                description='Delete a SkyPilot storage object.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description':
                                'Name of the storage object to delete.'
                        }
                    },
                    'required': ['name']
                }),
            # Resource Discovery Tools
            types.Tool(
                name='sky_list_accelerators',
                description=
                'List available GPU/accelerator types across clouds. Useful for '
                'finding the right accelerator for your workload.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'accelerator_str': {
                            'type':
                                'string',
                            'description':
                                'Filter by accelerator name (e.g., "V100", "A100").'
                        },
                        'clouds': {
                            'type':
                                'array',
                            'items': {
                                'type': 'string'
                            },
                            'description':
                                'Filter by cloud providers (e.g., ["aws", "gcp"]).'
                        }
                    },
                    'required': []
                }),
            types.Tool(
                name='sky_cost_report',
                description=
                'Get cost report for clusters. Shows estimated costs for running '
                'and historical clusters.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'days': {
                            'type':
                                'integer',
                            'description':
                                'Number of days to include in the report.'
                        }
                    },
                    'required': []
                }),
            types.Tool(
                name='sky_check',
                description=
                'Check which cloud providers are enabled and configured. '
                'Returns the status of cloud credentials.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'clouds': {
                            'type':
                                'array',
                            'items': {
                                'type': 'string'
                            },
                            'description':
                                'List of clouds to check (e.g., ["aws", "gcp"]). '
                                'If not provided, checks all clouds.'
                        }
                    },
                    'required': []
                }),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        """Handle tool calls from MCP clients."""
        try:
            result = await _handle_tool_call(name, arguments)
            return [types.TextContent(type='text', text=_format_result(result))]
        except Exception as e:
            error_msg = f'Error executing {name}: {str(e)}\n{traceback.format_exc()}'
            logger.error(error_msg)
            return [types.TextContent(type='text', text=error_msg)]

    return server


async def _handle_tool_call(name: str, arguments: dict) -> Any:
    """Handle a tool call and return the result."""
    sdk = _get_sdk()
    jobs_sdk = _get_jobs_sdk()
    serve_sdk = _get_serve_sdk()

    # Cluster Management
    if name == 'sky_status':
        cluster_names = arguments.get('cluster_names')
        refresh = arguments.get('refresh', False)
        all_users = arguments.get('all_users', False)
        refresh_mode = (sdk.StatusRefreshMode.FORCE
                        if refresh else sdk.StatusRefreshMode.NONE)
        return await _execute_sdk_call(sdk.status,
                                       cluster_names=cluster_names,
                                       refresh=refresh_mode,
                                       all_users=all_users)

    elif name == 'sky_launch':
        task_yaml = arguments.get('task_yaml')
        task = sdk.Task.from_yaml_config(
            common_utils.decode_yaml_from_str(task_yaml))
        return await _execute_sdk_call(
            sdk.launch,
            task,
            cluster_name=arguments.get('cluster_name'),
            idle_minutes_to_autostop=arguments.get('idle_minutes_to_autostop'),
            down=arguments.get('down', False),
            retry_until_up=arguments.get('retry_until_up', False),
            dryrun=arguments.get('dryrun', False))

    elif name == 'sky_exec':
        cluster_name = arguments.get('cluster_name')
        task_yaml = arguments.get('task_yaml')
        task = sdk.Task.from_yaml_config(
            common_utils.decode_yaml_from_str(task_yaml))
        return await _execute_sdk_call(sdk.exec,
                                       task,
                                       cluster_name=cluster_name,
                                       dryrun=arguments.get('dryrun', False))

    elif name == 'sky_stop':
        return await _execute_sdk_call(sdk.stop,
                                       arguments.get('cluster_name'),
                                       purge=arguments.get('purge', False))

    elif name == 'sky_start':
        return await _execute_sdk_call(
            sdk.start,
            arguments.get('cluster_name'),
            idle_minutes_to_autostop=arguments.get('idle_minutes_to_autostop'),
            down=arguments.get('down', False),
            retry_until_up=arguments.get('retry_until_up', False))

    elif name == 'sky_down':
        return await _execute_sdk_call(sdk.down,
                                       arguments.get('cluster_name'),
                                       purge=arguments.get('purge', False))

    elif name == 'sky_autostop':
        return await _execute_sdk_call(sdk.autostop,
                                       arguments.get('cluster_name'),
                                       idle_minutes=arguments.get(
                                           'idle_minutes'),
                                       down=arguments.get('down', False))

    # Job Management
    elif name == 'sky_queue':
        return await _execute_sdk_call(
            sdk.queue,
            arguments.get('cluster_name'),
            all_users=arguments.get('all_users', False),
            skip_finished=arguments.get('skip_finished', False))

    elif name == 'sky_cancel':
        job_ids = arguments.get('job_ids')
        return await _execute_sdk_call(sdk.cancel,
                                       arguments.get('cluster_name'),
                                       job_ids=job_ids,
                                       all=arguments.get('all', False))

    # Managed Jobs
    elif name == 'sky_jobs_launch':
        task_yaml = arguments.get('task_yaml')
        task = sdk.Task.from_yaml_config(
            common_utils.decode_yaml_from_str(task_yaml))
        return await _execute_sdk_call(jobs_sdk.launch,
                                       task,
                                       name=arguments.get('name'))

    elif name == 'sky_jobs_queue':
        return await _execute_sdk_call(
            jobs_sdk.queue,
            refresh=False,
            skip_finished=arguments.get('skip_finished', False),
            all_users=arguments.get('all_users', False),
            job_ids=arguments.get('job_ids'))

    elif name == 'sky_jobs_cancel':
        return await _execute_sdk_call(jobs_sdk.cancel,
                                       name=arguments.get('name'),
                                       job_ids=arguments.get('job_ids'),
                                       all=arguments.get('all', False))

    # Serve
    elif name == 'sky_serve_up':
        task_yaml = arguments.get('task_yaml')
        task = sdk.Task.from_yaml_config(
            common_utils.decode_yaml_from_str(task_yaml))
        return await _execute_sdk_call(serve_sdk.up,
                                       task,
                                       service_name=arguments.get(
                                           'service_name'))

    elif name == 'sky_serve_status':
        service_names = arguments.get('service_names')
        return await _execute_sdk_call(serve_sdk.status,
                                       service_names=service_names)

    elif name == 'sky_serve_down':
        service_names = arguments.get('service_names')
        return await _execute_sdk_call(serve_sdk.down,
                                       service_names=service_names,
                                       all=arguments.get('all', False),
                                       purge=arguments.get('purge', False))

    # Storage
    elif name == 'sky_storage_ls':
        return await _execute_sdk_call(sdk.storage_ls)

    elif name == 'sky_storage_delete':
        return await _execute_sdk_call(sdk.storage_delete,
                                       arguments.get('name'))

    # Resource Discovery
    elif name == 'sky_list_accelerators':
        accelerator_str = arguments.get('accelerator_str')
        clouds = arguments.get('clouds')
        return await _execute_sdk_call(sdk.list_accelerators,
                                       accelerator_str=accelerator_str,
                                       clouds=clouds)

    elif name == 'sky_cost_report':
        days = arguments.get('days')
        return await _execute_sdk_call(sdk.cost_report, days=days)

    elif name == 'sky_check':
        clouds = arguments.get('clouds')
        infra_tuple = tuple(clouds) if clouds else None
        return await _execute_sdk_call(sdk.check,
                                       infra_list=infra_tuple,
                                       verbose=True)

    else:
        raise ValueError(f'Unknown tool: {name}')


async def run_server():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server  # pylint: disable=import-outside-toplevel

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream,
                         write_stream,
                         server.create_initialization_options())


def main():
    """Main entry point for the MCP server."""
    # Suppress SkyPilot logging output to stderr to avoid interfering with MCP
    import logging  # pylint: disable=import-outside-toplevel
    logging.getLogger('sky').setLevel(logging.WARNING)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error running MCP server: {e}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
