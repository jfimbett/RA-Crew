"""Utility functions for loading and interpolating YAML-based prompts.

Features:
- Caches parsed YAML files to avoid repeated disk I/O
- Provides safe placeholder interpolation with descriptive errors
- Allows retrieval of agent and task definitions by id
- Simple dependency expansion for task contexts (list of task ids)

Placeholders use Python str.format syntax. At runtime, you can pass values like:
        interpolate(template, metrics="['Total CEO compensation']", years="[2021,2022]")

Supported placeholders (extendable):
    metrics, ticker, years, filing_types, hint, sec_filing_content,
    derived_metrics, calculation_expressions, output_format

If a placeholder is missing from kwargs, interpolation raises a KeyError with guidance.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional
import yaml

_PROMPT_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE: Dict[str, Any] = {}

class PromptNotFoundError(Exception):
    pass

class InterpolationError(Exception):
    pass

def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompts(base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load agents.yaml and tasks.yaml from the prompts directory.

    Args:
        base_dir: Optional base directory; defaults to project root discovered relative to this file.
    Returns:
        Dict with keys 'agents' and 'tasks'.
    """
    if base_dir is None:
        # Assume structure project_root/ra_crew/utils/prompt_loader.py
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    prompts_dir = os.path.join(base_dir, 'prompts')

    agents_path = os.path.join(prompts_dir, 'agents.yaml')
    tasks_path = os.path.join(prompts_dir, 'tasks.yaml')

    cache_key = f"{agents_path}|{tasks_path}"
    with _PROMPT_CACHE_LOCK:
        if cache_key in _PROMPT_CACHE:
            return _PROMPT_CACHE[cache_key]
        data = {
            'agents': _load_yaml(agents_path).get('agents', []),
            'tasks': _load_yaml(tasks_path).get('tasks', []),
        }
        _PROMPT_CACHE[cache_key] = data
        return data

def get_agent_def(agent_id: str, prompts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prompts = prompts or load_prompts()
    for a in prompts['agents']:
        if a['id'] == agent_id:
            return a
    raise PromptNotFoundError(f"Agent id '{agent_id}' not found in agents.yaml")

def get_task_def(task_id: str, prompts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prompts = prompts or load_prompts()
    for t in prompts['tasks']:
        if t['id'] == task_id:
            return t
    raise PromptNotFoundError(f"Task id '{task_id}' not found in tasks.yaml")

def interpolate(template: str, **kwargs: Any) -> str:
    """Interpolate a template string using str.format with provided kwargs.

    Raises:
        InterpolationError if a required key is missing.
    """
    if not template:
        return template
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise InterpolationError(
            f"Missing placeholder '{missing}' for template interpolation. Provided keys: {list(kwargs.keys())}"
        ) from e

def build_agent_objects(model_name: str, verbose: bool, prompts: Optional[Dict[str, Any]] = None):
    """Return dict of agent_id -> kwargs for Agent construction (excluding llm)."""
    from crewai import Agent
    agents = {}
    prompts = prompts or load_prompts()
    for a in prompts['agents']:
        agents[a['id']] = Agent(
            name=a['name'],
            role=a['role'],
            goal=a['goal'],
            backstory=a['backstory'],
            llm=model_name,
            verbose=verbose,
            allow_delegation=a.get('allow_delegation', False),
            tools=[]  # future: map tool names
        )
    return agents

def build_task_objects(agent_objs: Dict[str, Any], verbose: bool, interpolation_kwargs: Dict[str, Any], prompts: Optional[Dict[str, Any]] = None):
    from crewai import Task
    tasks = {}
    prompts = prompts or load_prompts()
    # First pass create tasks without context linking
    temp_defs = {}
    for t in prompts['tasks']:
        desc = interpolate(t['description'], **interpolation_kwargs)
        expected = t.get('expected_output', '')
        task_obj = Task(
            description=desc,
            expected_output=expected,
            agent=agent_objs[t['agent']],
            verbose=verbose,
        )
        tasks[t['id']] = task_obj
        temp_defs[t['id']] = t
    # Second pass assign context
    for tid, t in temp_defs.items():
        ctx_ids = t.get('context', [])
        if ctx_ids:
            tasks[tid].context = [tasks[cid] for cid in ctx_ids if cid in tasks]
    return tasks
