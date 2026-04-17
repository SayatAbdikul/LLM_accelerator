#!/usr/bin/env python3
"""CLI: disassemble .bin → .asm"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taccel.assembler import Disassembler
from taccel.assembler.assembler import ProgramBinary


def main():
    parser = argparse.ArgumentParser(description="TACCEL disassembler: .bin → .asm")
    parser.add_argument("input", help="Input .bin file")
    parser.add_argument("-o", "--output", help="Output .asm file (default: stdout)")
    parser.add_argument("--no-offsets", action="store_true",
                        help="Don't show [0xNNNN] PC offsets in output")
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        raw = f.read()

    prog = ProgramBinary.from_bytes(raw)
    disasm = Disassembler()
    text = disasm.disassemble(prog)

    if args.no_offsets:
        lines = []
        for line in text.split('\n'):
            if '] ' in line:
                lines.append(line.split('] ', 1)[1])
            else:
                lines.append(line)
        text = '\n'.join(lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(text + '\n')
        print(f"Disassembled {prog.insn_count} instructions → {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
