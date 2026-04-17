"""Dynamic SRAM allocation with eviction scheduling."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ..isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE

UNIT = 16  # bytes per addressing unit


@dataclass
class Allocation:
    """A named region in an SRAM buffer."""
    name: str
    buf_id: int
    offset_units: int  # in 16-byte units
    size_units: int    # in 16-byte units
    size_bytes: int
    evictable: bool = True

    @property
    def end_units(self) -> int:
        return self.offset_units + self.size_units


class BufferAllocator:
    """Free-list SRAM allocator.

    Uses first-fit placement. When an allocation is freed, its slot is returned
    to the free list and can be reused by future allocations WITHOUT moving any
    live allocation. This is critical: already-emitted instructions reference
    ABUF/WBUF offsets by address; compacting (moving) live allocations would
    silently corrupt those references.
    """

    def __init__(self, buf_id: int, capacity_bytes: int):
        self.buf_id = buf_id
        self.capacity = capacity_bytes
        self.capacity_units = capacity_bytes // UNIT
        self.allocations: Dict[str, Allocation] = {}
        # Free list: sorted list of (start_units, size_units) free segments
        self._free: List[Tuple[int, int]] = [(0, capacity_bytes // UNIT)]
        self.high_water_units = 0

    def alloc(self, name: str, size_bytes: int, evictable: bool = True) -> Allocation:
        """First-fit allocation from free list."""
        size_units = (size_bytes + UNIT - 1) // UNIT
        for i, (start, avail) in enumerate(self._free):
            if avail >= size_units:
                alloc = Allocation(
                    name=name, buf_id=self.buf_id,
                    offset_units=start, size_units=size_units,
                    size_bytes=size_bytes, evictable=evictable,
                )
                self.allocations[name] = alloc
                # Update free list: consume size_units from this block
                remaining = avail - size_units
                if remaining > 0:
                    self._free[i] = (start + size_units, remaining)
                else:
                    self._free.pop(i)
                end = start + size_units
                self.high_water_units = max(self.high_water_units, end)
                return alloc
        # Compute live bytes for error message
        live = sum(a.size_units for a in self.allocations.values())
        raise MemoryError(
            f"Cannot allocate {size_bytes}B ({size_units} units) in buffer {self.buf_id}. "
            f"Capacity={self.capacity}B, live={live * UNIT}B, "
            f"largest free block={max((s for _, s in self._free), default=0) * UNIT}B"
        )

    def free(self, name: str):
        """Return allocation's region to the free list (no compaction)."""
        alloc = self.allocations.pop(name, None)
        if alloc is None:
            return
        start, size = alloc.offset_units, alloc.size_units
        # Insert and merge adjacent free segments
        self._free.append((start, size))
        self._free.sort()
        merged = []
        for seg_start, seg_size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == seg_start:
                merged[-1] = (merged[-1][0], merged[-1][1] + seg_size)
            else:
                merged.append([seg_start, seg_size])
        self._free = [(s, sz) for s, sz in merged]

    def free_all_evictable(self):
        to_remove = [n for n, a in self.allocations.items() if a.evictable]
        for n in to_remove:
            self.free(n)

    def reset(self):
        self.allocations.clear()
        self._free = [(0, self.capacity_units)]

    @property
    def next_free_units(self) -> int:
        """Compat shim: return high-water mark (not meaningful for free-list)."""
        return self.high_water_units

    @property
    def used_bytes(self) -> int:
        live = sum(a.size_units for a in self.allocations.values())
        return live * UNIT

    @property
    def free_bytes(self) -> int:
        return self.capacity - self.used_bytes

    def get(self, name: str) -> Optional[Allocation]:
        return self.allocations.get(name)


class MemoryAllocator:
    """Manages SRAM allocation across all buffers and DRAM temporaries."""

    def __init__(self):
        self.abuf = BufferAllocator(0, ABUF_SIZE)
        self.wbuf = BufferAllocator(1, WBUF_SIZE)
        self.accum = BufferAllocator(2, ACCUM_SIZE)
        # DRAM temporary region tracking
        self.dram_temp_offset = 0  # next free offset in DRAM temp region
        self.dram_temps: Dict[str, Tuple[int, int]] = {}  # name → (offset, size)

    def get_buffer(self, buf_id: int) -> BufferAllocator:
        return {0: self.abuf, 1: self.wbuf, 2: self.accum}[buf_id]

    def alloc_dram_temp(self, name: str, size_bytes: int) -> int:
        """Allocate a temporary region in DRAM. Returns byte offset."""
        offset = self.dram_temp_offset
        self.dram_temps[name] = (offset, size_bytes)
        self.dram_temp_offset += size_bytes
        return offset

    def free_dram_temp(self, name: str):
        if name in self.dram_temps:
            del self.dram_temps[name]

    @property
    def dram_temp_total(self) -> int:
        return self.dram_temp_offset
