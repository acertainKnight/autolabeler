#!/usr/bin/env python
"""Utility script to view debug prompts in a nice format."""

import json
import argparse
from pathlib import Path
from datetime import datetime
import textwrap


def view_debug_prompts(debug_file_path: Path, limit: int = None, full: bool = False):
    """View debug prompts from a JSON file.

    Args:
        debug_file_path: Path to the debug prompts JSON file
        limit: Maximum number of prompts to display
        full: Whether to show full prompt text or truncated
    """
    if not debug_file_path.exists():
        print(f"Error: File not found: {debug_file_path}")
        return

    with open(debug_file_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"Debug Prompts for: {data.get('dataset_name', 'Unknown')}")
    print(f"Service: {data.get('service', 'labeling')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    print(f"Total Prompts: {data.get('prompts_count', 0)}")
    print(f"{'='*80}\n")

    prompts = data.get('prompts', [])
    if limit:
        prompts = prompts[:limit]

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}] Prompt ID: {prompt.get('prompt_id', 'N/A')}")
        print(f"    Timestamp: {prompt.get('timestamp', 'N/A')}")

        # For labeling prompts
        if 'text' in prompt:
            print(f"    Input Text: {prompt['text'][:100]}{'...' if len(prompt['text']) > 100 else ''}")
            print(f"    Examples Used: {prompt.get('examples_count', 0)}")

        # For synthetic generation prompts
        if 'method' in prompt:
            print(f"    Method: {prompt['method']}")
            print(f"    Target Label: {prompt.get('target_label', 'N/A')}")
            print(f"    Strategy: {prompt.get('strategy', 'N/A')}")

        if prompt.get('ruleset'):
            print(f"    Ruleset: {prompt['ruleset']}")

        # Show rendered prompt
        rendered = prompt.get('rendered_prompt', '')
        if full:
            print(f"    Full Prompt:")
            wrapped = textwrap.fill(rendered, width=76, initial_indent='      ', subsequent_indent='      ')
            print(wrapped)
        else:
            preview_length = 200
            preview = rendered[:preview_length] + ('...' if len(rendered) > preview_length else '')
            print(f"    Prompt Preview:")
            wrapped = textwrap.fill(preview, width=76, initial_indent='      ', subsequent_indent='      ')
            print(wrapped)

        print(f"\n{'-'*80}\n")


def main():
    parser = argparse.ArgumentParser(description='View debug prompts in a nice format')
    parser.add_argument('file', type=Path, help='Path to debug prompts JSON file')
    parser.add_argument('-n', '--limit', type=int, help='Maximum number of prompts to display')
    parser.add_argument('-f', '--full', action='store_true', help='Show full prompt text')

    args = parser.parse_args()

    view_debug_prompts(args.file, args.limit, args.full)


if __name__ == '__main__':
    main()
