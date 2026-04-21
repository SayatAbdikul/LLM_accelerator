"""Runtime CONFIG_ATTN patch-site tests."""
import pytest

from taccel.assembler.assembler import ProgramBundle, RuntimeConfigAttnSite
from taccel.isa.encoding import decode, encode
from taccel.isa.instructions import ConfigAttnInsn, HaltInsn, NopInsn


def _bytes(*insns):
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return bytes(out)


def test_patch_config_attn_site_updates_decode_stream_fields():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(ConfigAttnInsn(query_row_base=0, valid_kv_len=1, mode=0b11), HaltInsn()),
        runtime_config_attn_sites=[
            RuntimeConfigAttnSite("decode", local_pc=0, absolute_pc=0, mode=0b11),
        ],
    )
    site = bundle.runtime_config_attn_sites[0]

    bundle.patch_config_attn_site(site, query_row_base=23, valid_kv_len=24)

    patched = decode(bundle.stream_bytes("decode")[:8])
    assert isinstance(patched, ConfigAttnInsn)
    assert patched.query_row_base == 23
    assert patched.valid_kv_len == 24
    assert patched.mode == 0b11
    assert site.absolute_pc == bundle.decode_pc


def test_patch_config_attn_site_faults_when_pc_is_not_config_attn():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(NopInsn(), HaltInsn()),
        runtime_config_attn_sites=[
            RuntimeConfigAttnSite("decode", local_pc=0, absolute_pc=0, mode=0b11),
        ],
    )

    with pytest.raises(ValueError, match="not CONFIG_ATTN"):
        bundle.patch_config_attn_site(bundle.runtime_config_attn_sites[0], query_row_base=1, valid_kv_len=2)
