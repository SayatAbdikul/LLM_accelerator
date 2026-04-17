#!/usr/bin/env python3
"""CLI: assemble .asm → .bin"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taccel.assembler import Assembler


def main():
    parser = argparse.ArgumentParser(description="TACCEL assembler: .asm → .bin")
    parser.add_argument("input", help="Input .asm file")
    parser.add_argument("-o", "--output", help="Output .bin file (default: <input>.bin)")
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.input)[0] + ".bin"

    with open(args.input, 'r') as f:
        source = f.read()

    asm = Assembler()
    prog = asm.assemble(source)

    with open(output, 'wb') as f:
        f.write(prog.to_bytes())

    print(f"Assembled {prog.insn_count} instructions → {output}")
    print(f"  Instructions: {len(prog.instructions)} bytes")
    print(f"  Data:         {len(prog.data)} bytes")


if __name__ == "__main__":
    main()
