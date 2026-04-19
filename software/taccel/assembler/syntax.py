"""Assembly syntax parser - converts text lines to instruction objects."""
import re
from typing import Optional, Tuple, List
from ..isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn, SoftmaxInsn, LayernormInsn, GeluInsn,
    SoftmaxAttnVInsn, MaskedSoftmaxInsn, MaskedSoftmaxAttnVInsn, DequantAddInsn,
    LoadInsn, StoreInsn, BufCopyInsn, SetAddrLoInsn, SetAddrHiInsn,
    ConfigTileInsn, ConfigAttnInsn, SetScaleInsn, SyncInsn, NopInsn, HaltInsn, Instruction,
)

BUFFER_NAME_TO_ID = {"ABUF": BUF_ABUF, "WBUF": BUF_WBUF, "ACCUM": BUF_ACCUM}

# Pattern: BUFNAME[0xOFFSET] or BUFNAME[OFFSET]
BUF_REF_RE = re.compile(r'(ABUF|WBUF|ACCUM)\[(?:0x)?([0-9a-fA-F]+)\]')

# Mnemonic aliases
MNEMONIC_MAP = {
    "NOP": Opcode.NOP,
    "HALT": Opcode.HALT,
    "SYNC": Opcode.SYNC,
    "CONFIG_TILE": Opcode.CONFIG_TILE,
    "SET_SCALE": Opcode.SET_SCALE,
    "SET_ADDR_LO": Opcode.SET_ADDR_LO,
    "SET_ADDR_HI": Opcode.SET_ADDR_HI,
    "LOAD": Opcode.LOAD,
    "STORE": Opcode.STORE,
    "BUF_COPY": Opcode.BUF_COPY,
    "MATMUL": Opcode.MATMUL,
    "REQUANT": Opcode.REQUANT,
    "REQUANT_PC": Opcode.REQUANT_PC,
    "SCALE_MUL": Opcode.SCALE_MUL,
    "VADD": Opcode.VADD,
    "SOFTMAX": Opcode.SOFTMAX,
    "LAYERNORM": Opcode.LAYERNORM,
    "GELU": Opcode.GELU,
    "SOFTMAX_ATTNV": Opcode.SOFTMAX_ATTNV,
    "DEQUANT_ADD": Opcode.DEQUANT_ADD,
    "CONFIG_ATTN": Opcode.CONFIG_ATTN,
    "MASKED_SOFTMAX": Opcode.MASKED_SOFTMAX,
    "MASKED_SOFTMAX_ATTNV": Opcode.MASKED_SOFTMAX_ATTNV,
}


def parse_int(s: str) -> int:
    """Parse integer from decimal or hex string."""
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    if s.startswith("0b") or s.startswith("0B"):
        return int(s, 2)
    return int(s)


def parse_buf_ref(s: str) -> Tuple[int, int]:
    """Parse 'ABUF[0x100]' → (buf_id, offset)."""
    m = BUF_REF_RE.match(s.strip())
    if not m:
        raise SyntaxError(f"Invalid buffer reference: {s}")
    buf_id = BUFFER_NAME_TO_ID[m.group(1)]
    offset = int(m.group(2), 16) if '0x' in s.lower() or any(c in 'abcdefABCDEF' for c in m.group(2)) else int(m.group(2))
    return buf_id, offset


def parse_sreg(s: str) -> int:
    """Parse 'S0'...'S15' → 0...15."""
    s = s.strip()
    if not s.upper().startswith("S"):
        raise SyntaxError(f"Expected scale register Sn, got: {s}")
    return int(s[1:])


def parse_line(line: str) -> Tuple[Optional[str], Optional[Instruction]]:
    """Parse a single assembly line.

    Returns (label_or_none, instruction_or_none).
    """
    # Strip comments
    for comment_char in [';', '#']:
        idx = line.find(comment_char)
        if idx >= 0:
            line = line[:idx]
    line = line.strip()
    if not line:
        return None, None

    # Check for label
    label = None
    if ':' in line:
        parts = line.split(':', 1)
        if not parts[0].strip().startswith(('ABUF', 'WBUF', 'ACCUM')):
            label = parts[0].strip()
            line = parts[1].strip()
            if not line:
                return label, None

    # Parse mnemonic
    tokens = line.split(None, 1)
    mnemonic = tokens[0].upper()
    args_str = tokens[1] if len(tokens) > 1 else ""

    if mnemonic not in MNEMONIC_MAP:
        raise SyntaxError(f"Unknown mnemonic: {mnemonic}")

    opcode = MNEMONIC_MAP[mnemonic]

    # Parse args based on mnemonic
    if opcode == Opcode.NOP:
        return label, NopInsn()
    elif opcode == Opcode.HALT:
        return label, HaltInsn()
    elif opcode == Opcode.SYNC:
        mask = parse_int(args_str) if args_str.strip() else 0
        return label, SyncInsn(resource_mask=mask)

    elif opcode == Opcode.CONFIG_TILE:
        # CONFIG_TILE M=13, N=13, K=4 (tile counts, 1-based)
        params = {}
        for part in args_str.split(','):
            k, v = part.strip().split('=')
            params[k.strip().upper()] = parse_int(v.strip())
        # Subtract 1 for encoding (tile_count → 0-based)
        return label, ConfigTileInsn(
            M=params['M'] - 1,
            N=params['N'] - 1,
            K=params['K'] - 1,
        )

    elif opcode == Opcode.CONFIG_ATTN:
        params = _parse_kv_args(args_str)
        return label, ConfigAttnInsn(
            query_row_base=parse_int(params.get('query_row_base', '0')),
            valid_kv_len=parse_int(params.get('valid_kv_len', '0')),
            mode=parse_int(params.get('mode', '0')),
        )

    elif opcode == Opcode.SET_SCALE:
        # SET_SCALE S0, imm=0x3000 or SET_SCALE S0, ABUF[0x100]
        parts = [p.strip() for p in args_str.split(',', 1)]
        sreg = parse_sreg(parts[0])
        rest = parts[1].strip()
        if rest.startswith("imm="):
            imm_val = parse_int(rest[4:])
            return label, SetScaleInsn(sreg=sreg, src_mode=0, imm16=imm_val)
        else:
            buf_id, offset = parse_buf_ref(rest)
            src_mode = {BUF_ABUF: 1, BUF_WBUF: 2, BUF_ACCUM: 3}[buf_id]
            return label, SetScaleInsn(sreg=sreg, src_mode=src_mode, imm16=offset)

    elif opcode in (Opcode.SET_ADDR_LO, Opcode.SET_ADDR_HI):
        # SET_ADDR_LO R0, 0x1234567
        parts = [p.strip() for p in args_str.split(',')]
        addr_reg = int(parts[0].strip().upper().replace('R', ''))
        imm = parse_int(parts[1])
        cls = SetAddrLoInsn if opcode == Opcode.SET_ADDR_LO else SetAddrHiInsn
        return label, cls(addr_reg=addr_reg, imm28=imm)

    elif opcode in (Opcode.LOAD, Opcode.STORE):
        # LOAD buf_id=WBUF, sram_off=0, xfer_len=9216, addr_reg=0, dram_off=0
        # Also accept: LOAD WBUF[0x0000], len=9216, R0, dram_off=0
        params = _parse_kv_args(args_str)
        buf_id = BUFFER_NAME_TO_ID.get(params.get('buf_id', '').upper(), 0)
        cls = LoadInsn if opcode == Opcode.LOAD else StoreInsn
        return label, cls(
            buf_id=buf_id,
            sram_off=parse_int(params.get('sram_off', '0')),
            xfer_len=parse_int(params.get('xfer_len', '0')),
            addr_reg=parse_int(params.get('addr_reg', '0')),
            dram_off=parse_int(params.get('dram_off', '0')),
            stride_log2=parse_int(params.get('stride_log2', '0')),
            flags=parse_int(params.get('flags', '0')),
        )

    elif opcode == Opcode.BUF_COPY:
        params = _parse_kv_or_positional_bufcopy(args_str)
        return label, BufCopyInsn(**params)

    elif opcode in (Opcode.MATMUL, Opcode.REQUANT, Opcode.REQUANT_PC, Opcode.SCALE_MUL, Opcode.VADD,
                    Opcode.SOFTMAX, Opcode.LAYERNORM, Opcode.GELU, Opcode.SOFTMAX_ATTNV,
                    Opcode.MASKED_SOFTMAX, Opcode.MASKED_SOFTMAX_ATTNV, Opcode.DEQUANT_ADD):
        return label, _parse_r_type(opcode, args_str)

    raise SyntaxError(f"Unhandled mnemonic: {mnemonic}")


def _parse_kv_args(args_str: str) -> dict:
    """Parse key=value argument pairs."""
    result = {}
    for part in args_str.split(','):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result


def _parse_kv_or_positional_bufcopy(args_str: str) -> dict:
    """Parse BUF_COPY arguments."""
    params = _parse_kv_args(args_str)
    return {
        'src_buf': BUFFER_NAME_TO_ID.get(params.get('src_buf', '').upper(), 0),
        'src_off': parse_int(params.get('src_off', '0')),
        'dst_buf': BUFFER_NAME_TO_ID.get(params.get('dst_buf', '').upper(), 0),
        'dst_off': parse_int(params.get('dst_off', '0')),
        'length': parse_int(params.get('length', '0')),
        'src_rows': parse_int(params.get('src_rows', '0')),
        'transpose': parse_int(params.get('transpose', '0')),
    }


def _parse_r_type(opcode: Opcode, args_str: str) -> Instruction:
    """Parse R-type instruction arguments.

    Format: MATMUL ABUF[0x100], WBUF[0x000], ACCUM[0x000], S0, acc=1
    Or key=value: src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=S0, flags=0
    """
    cls_map = {
        Opcode.MATMUL: MatmulInsn, Opcode.REQUANT: RequantInsn,
        Opcode.REQUANT_PC: RequantPcInsn,
        Opcode.SCALE_MUL: ScaleMulInsn, Opcode.VADD: VaddInsn,
        Opcode.SOFTMAX: SoftmaxInsn, Opcode.LAYERNORM: LayernormInsn,
        Opcode.GELU: GeluInsn, Opcode.SOFTMAX_ATTNV: SoftmaxAttnVInsn,
        Opcode.MASKED_SOFTMAX: MaskedSoftmaxInsn,
        Opcode.MASKED_SOFTMAX_ATTNV: MaskedSoftmaxAttnVInsn,
        Opcode.DEQUANT_ADD: DequantAddInsn,
    }

    if '=' in args_str and 'src1=' in args_str:
        # Key-value format
        params = _parse_kv_args(args_str)
        src1_buf, src1_off = parse_buf_ref(params.get('src1', 'ABUF[0]'))
        src2_buf, src2_off = parse_buf_ref(params.get('src2', 'ABUF[0]'))
        dst_buf, dst_off = parse_buf_ref(params.get('dst', 'ABUF[0]'))
        sreg = parse_int(params.get('sreg', '0'))
        flags = parse_int(params.get('flags', '0'))
    else:
        # Positional format: BUF[off], BUF[off], BUF[off], S0, acc=1
        parts = [p.strip() for p in args_str.split(',')]
        src1_buf, src1_off = parse_buf_ref(parts[0]) if len(parts) > 0 else (0, 0)
        src2_buf, src2_off = parse_buf_ref(parts[1]) if len(parts) > 1 else (0, 0)
        dst_buf, dst_off = parse_buf_ref(parts[2]) if len(parts) > 2 else (0, 0)
        sreg = 0
        flags = 0
        for p in parts[3:]:
            p = p.strip()
            if p.upper().startswith('S') and p[1:].isdigit():
                sreg = int(p[1:])
            elif 'acc=' in p.lower():
                flags = parse_int(p.split('=')[1])
            elif p.isdigit():
                sreg = int(p)

    return cls_map[opcode](
        src1_buf=src1_buf, src1_off=src1_off,
        src2_buf=src2_buf, src2_off=src2_off,
        dst_buf=dst_buf, dst_off=dst_off,
        sreg=sreg, flags=flags,
    )
