#!/usr/bin/env python3
"""
Convert PGN game files to EPD training data for Texel tuning.

Extracts quiet positions from games with known results.
Filters: skip first N moves, skip positions in check, skip captures.

Usage:
    python3 scripts/pgn_to_epd.py input.pgn -o training_data.epd
    python3 scripts/pgn_to_epd.py games/*.pgn -o training_data.epd --min-elo 2000
"""

import argparse
import re
import sys


def parse_result(result_str):
    """Convert PGN result to float: 1.0 (white wins), 0.0 (black wins), 0.5 (draw)."""
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return 0.0
    elif result_str == "1/2-1/2":
        return 0.5
    return None


def parse_pgn_games(filename):
    """Yield (headers, moves_text) for each game in a PGN file."""
    headers = {}
    moves_lines = []
    in_moves = False

    with open(filename, errors='replace') as f:
        for line in f:
            line = line.strip()

            if line.startswith('['):
                if in_moves and moves_lines:
                    yield headers, ' '.join(moves_lines)
                    headers = {}
                    moves_lines = []
                    in_moves = False

                # Parse header
                m = re.match(r'\[(\w+)\s+"(.*)"\]', line)
                if m:
                    headers[m.group(1)] = m.group(2)

            elif line:
                in_moves = True
                moves_lines.append(line)

        # Last game
        if moves_lines:
            yield headers, ' '.join(moves_lines)


def extract_fens_from_moves(moves_text, result, skip_moves=8, max_positions=50):
    """
    Extract FEN-like move numbers from PGN move text.
    Since we don't have a board engine here, we output move numbers
    and the result — the actual FEN extraction needs the engine.

    NOTE: This is a simplified approach. For proper FEN extraction,
    use the havoc engine's PGN parser or an external tool like pgn-extract.
    """
    # This function returns (move_number, result) pairs
    # The actual FEN extraction is done by the C++ tool below
    positions = []

    # Count moves
    move_numbers = re.findall(r'(\d+)\.\s', moves_text)
    total_moves = len(move_numbers)

    for i, mn in enumerate(move_numbers):
        move_num = int(mn)
        if move_num > skip_moves and move_num < total_moves - 5:
            positions.append((move_num, result))
            if len(positions) >= max_positions:
                break

    return positions


def main():
    parser = argparse.ArgumentParser(description='Convert PGN to EPD for Texel tuning')
    parser.add_argument('pgn_files', nargs='+', help='Input PGN files')
    parser.add_argument('-o', '--output', default='training_data.epd', help='Output EPD file')
    parser.add_argument('--min-elo', type=int, default=0, help='Minimum Elo filter')
    parser.add_argument('--skip-moves', type=int, default=8, help='Skip first N moves')
    parser.add_argument('--max-games', type=int, default=0, help='Max games to process (0=all)')
    args = parser.parse_args()

    print("NOTE: This script extracts game metadata only.")
    print("For full FEN extraction, use the C++ tool: havoc_pgn2epd")
    print(f"Processing {len(args.pgn_files)} file(s)...")
    print()

    total_games = 0
    filtered_games = 0
    results = {1.0: 0, 0.5: 0, 0.0: 0}

    for pgn_file in args.pgn_files:
        print(f"  {pgn_file}...", end='', flush=True)
        file_games = 0

        for headers, moves_text in parse_pgn_games(pgn_file):
            total_games += 1

            result = parse_result(headers.get('Result', '*'))
            if result is None:
                continue

            # Elo filter
            if args.min_elo > 0:
                white_elo = int(headers.get('WhiteElo', '0') or '0')
                black_elo = int(headers.get('BlackElo', '0') or '0')
                if white_elo < args.min_elo or black_elo < args.min_elo:
                    continue

            filtered_games += 1
            results[result] += 1
            file_games += 1

            if args.max_games > 0 and filtered_games >= args.max_games:
                break

        print(f" {file_games} games")

        if args.max_games > 0 and filtered_games >= args.max_games:
            break

    print(f"\nTotal: {total_games} games, {filtered_games} after filtering")
    print(f"Results: W={results[1.0]} D={results[0.5]} L={results[0.0]}")
    print(f"\nTo generate EPD with FENs, use the C++ tool:")
    print(f"  ./build/havoc_pgn2epd {' '.join(args.pgn_files)} -o {args.output}")


if __name__ == '__main__':
    main()
