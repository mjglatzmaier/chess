#!/usr/bin/env python3
"""
Bake tuned parameters into parameters.hpp source code.

Reads a tuned_params.txt file (key = value format) and updates the default
values in include/havoc/parameters.hpp so the next build uses them.

Usage:
    python3 scripts/bake_params.py tuned_params.txt
    python3 scripts/bake_params.py tuned_params.txt --header include/havoc/parameters.hpp
"""

import re
import sys
import argparse
from pathlib import Path


def parse_param_file(path: str) -> dict[str, int]:
    """Parse key = value parameter file."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            params[key.strip()] = int(val.strip())
    return params


def update_header(header_path: str, params: dict[str, int]) -> int:
    """Update default values in parameters.hpp. Returns count of updates."""
    with open(header_path) as f:
        source = f.read()

    updated = 0

    # Match array params like: std::array<int, N> name = {v1, v2, ...};
    # Group indexed params by base name
    array_params: dict[str, dict[int, int]] = {}
    scalar_params: dict[str, int] = {}

    for key, val in params.items():
        # Check if it's an indexed param (e.g., "material_value_0")
        match = re.match(r'^(.+)_(\d+)$', key)
        if match:
            base_name = match.group(1)
            idx = int(match.group(2))
            if base_name not in array_params:
                array_params[base_name] = {}
            array_params[base_name][idx] = val
        else:
            scalar_params[key] = val

    # Update scalar params (e.g., "int open_file_bonus = 1;")
    for name, val in scalar_params.items():
        # Skip tempo (float, needs special handling)
        if name == 'tempo':
            continue

        # Match: int name = <old_val>;  or  = <old_val>,  etc.
        pattern = rf'(\b{re.escape(name)}\b\s*=\s*)-?\d+'
        new_source = re.sub(pattern, rf'\g<1>{val}', source)
        if new_source != source:
            source = new_source
            updated += 1

    # Update array params
    for base_name, indices in array_params.items():
        # Find the array initializer: name = {v1, v2, v3, ...};
        pattern = rf'(\b{re.escape(base_name)}\b\s*=\s*\{{)([^}}]+)(\}})'
        match = re.search(pattern, source)
        if match:
            old_values = match.group(2).split(',')
            new_values = list(old_values)  # preserve formatting
            for idx, val in indices.items():
                if idx < len(new_values):
                    new_values[idx] = str(val)
            new_init = ', '.join(v.strip() for v in new_values)
            replacement = match.group(1) + new_init + match.group(3)
            source = source[:match.start()] + replacement + source[match.end():]
            updated += 1

    with open(header_path, 'w') as f:
        f.write(source)

    return updated


def main():
    parser = argparse.ArgumentParser(description='Bake tuned params into source')
    parser.add_argument('param_file', help='Path to tuned_params.txt')
    parser.add_argument('--header', default='include/havoc/parameters.hpp',
                        help='Path to parameters.hpp')
    args = parser.parse_args()

    params = parse_param_file(args.param_file)
    print(f"Read {len(params)} parameters from {args.param_file}")

    updated = update_header(args.header, params)
    print(f"Updated {updated} values in {args.header}")
    print("Rebuild the engine to use the new defaults.")


if __name__ == '__main__':
    main()
