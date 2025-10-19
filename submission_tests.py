import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functools import lru_cache
import unittest
import random

from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
import random

Engine = Literal["alu", "load", "store", "flow"]
Instruction = dict[Engine, list[tuple]]


class CoreState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@dataclass
class Core:
    id: int
    scratch: list[int]
    trace_buf: list[int]
    pc: int = 0
    state: CoreState = CoreState.RUNNING


@dataclass
class DebugInfo:
    """
    We give you some debug info but it's up to you to use it in Machine if you
    want to. You're also welcome to add more.
    """

    # Maps scratch variable addr to (name, len) pair
    scratch_map: dict[int, (str, int)]


def cdiv(a, b):
    return (a + b - 1) // b


SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

VLEN = 8
# Older versions of the take-home used multiple cores, but this version only uses 1
N_CORES = 1
SCRATCH_SIZE = 1536
BASE_ADDR_TID = 100000


class Machine:
    """
    Simulator for a custom VLIW SIMD architecture.

    VLIW (Very Large Instruction Word): Cores are composed of different
    "engines" each of which can execute multiple "slots" per cycle in parallel.
    How many slots each engine can execute per cycle is limited by SLOT_LIMITS.
    Effects of instructions don't take effect until the end of cycle. Each
    cycle, all engines execute all of their filled slots for that instruction.
    Effects like writes to memory take place after all the inputs are read.

    SIMD: There are instructions for acting on vectors of VLEN elements in a
    single slot. You can use vload and vstore to load multiple contiguous
    elements but not non-contiguous elements. Use vbroadcast to broadcast a
    scalar to a vector and then operate on vectors with valu instructions.

    The memory and scratch space are composed of 32-bit words. The solution is
    plucked out of the memory at the end of the program. You can think of the
    scratch space as serving the purpose of registers, constant memory, and a
    manually-managed cache.

    Here's an example of what an instruction might look like:

    {"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}

    In general every number in an instruction is a scratch address except for
    const and jump, and except for store and some flow instructions the first
    operand is the destination.

    This comment is not meant to be full ISA documentation though, for the rest
    you should look through the simulator code.
    """

    def __init__(
        self,
        mem_dump: list[int],
        program: list[Instruction],
        debug_info: DebugInfo,
        n_cores: int = 1,
        scratch_size: int = SCRATCH_SIZE,
        trace: bool = False,
        value_trace: dict[Any, int] = {},
    ):
        self.cores = [
            Core(id=i, scratch=[0] * scratch_size, trace_buf=[]) for i in range(n_cores)
        ]
        self.mem = copy(mem_dump)
        self.program = program
        self.debug_info = debug_info
        self.value_trace = value_trace
        self.prints = False
        self.cycle = 0
        self.enable_pause = True
        self.enable_debug = True
        if trace:
            self.setup_trace()
        else:
            self.trace = None

    def rewrite_instr(self, instr):
        """
        Rewrite an instruction to use scratch addresses instead of names
        """
        res = {}
        for name, slots in instr.items():
            res[name] = []
            for slot in slots:
                res[name].append(self.rewrite_slot(slot))
        return res

    def print_step(self, instr, core):
        # print(core.id)
        # print(core.trace_buf)
        print(self.scratch_map(core))
        print(core.pc, instr, self.rewrite_instr(instr))

    def scratch_map(self, core):
        res = {}
        for addr, (name, length) in self.debug_info.scratch_map.items():
            res[name] = core.scratch[addr : addr + length]
        return res

    def rewrite_slot(self, slot):
        return tuple(
            self.debug_info.scratch_map.get(s, (None, None))[0] or s for s in slot
        )

    def setup_trace(self):
        """
        The simulator generates traces in Chrome's Trace Event Format for
        visualization in Perfetto (or chrome://tracing if you prefer it). See
        the bottom of the file for info about how to use this.

        See the format docs in case you want to add more info to the trace:
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        """
        self.trace = open("trace.json", "w")
        self.trace.write("[")
        tid_counter = 0
        self.tids = {}
        for ci, core in enumerate(self.cores):
            self.trace.write(
                f'{{"name": "process_name", "ph": "M", "pid": {ci}, "tid": 0, "args": {{"name":"Core {ci}"}}}},\n'
            )
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid_counter += 1
                    self.trace.write(
                        f'{{"name": "thread_name", "ph": "M", "pid": {ci}, "tid": {tid_counter}, "args": {{"name":"{name}-{i}"}}}},\n'
                    )
                    self.tids[(ci, name, i)] = tid_counter

        # Add zero-length events at the start so all slots show up in Perfetto
        for ci, core in enumerate(self.cores):
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid = self.tids[(ci, name, i)]
                    self.trace.write(
                        f'{{"name": "init", "cat": "op", "ph": "X", "pid": {ci}, "tid": {tid}, "ts": 0, "dur": 0}},\n'
                    )
        for ci, core in enumerate(self.cores):
            self.trace.write(
                f'{{"name": "process_name", "ph": "M", "pid": {len(self.cores) + ci}, "tid": 0, "args": {{"name":"Core {ci} Scratch"}}}},\n'
            )
            for addr, (name, length) in self.debug_info.scratch_map.items():
                self.trace.write(
                    f'{{"name": "thread_name", "ph": "M", "pid": {len(self.cores) + ci}, "tid": {BASE_ADDR_TID + addr}, "args": {{"name":"{name}-{length}"}}}},\n'
                )

    def run(self):
        for core in self.cores:
            if core.state == CoreState.PAUSED:
                core.state = CoreState.RUNNING
        while any(c.state == CoreState.RUNNING for c in self.cores):
            for core in self.cores:
                if core.state != CoreState.RUNNING:
                    continue
                if core.pc >= len(self.program):
                    core.state = CoreState.STOPPED
                    continue
                instr = self.program[core.pc]
                if self.prints:
                    self.print_step(instr, core)
                core.pc += 1
                self.step(instr, core)
            self.cycle += 1

    def alu(self, core, op, dest, a1, a2):
        a1 = core.scratch[a1]
        a2 = core.scratch[a2]
        match op:
            case "+":
                res = a1 + a2
            case "-":
                res = a1 - a2
            case "*":
                res = a1 * a2
            case "//":
                res = a1 // a2
            case "cdiv":
                res = cdiv(a1, a2)
            case "^":
                res = a1 ^ a2
            case "&":
                res = a1 & a2
            case "|":
                res = a1 | a2
            case "<<":
                res = a1 << a2
            case ">>":
                res = a1 >> a2
            case "%":
                res = a1 % a2
            case "<":
                res = int(a1 < a2)
            case "==":
                res = int(a1 == a2)
            case _:
                raise NotImplementedError(f"Unknown alu op {op}")
        res = res % (2**32)
        self.scratch_write[dest] = res

    def valu(self, core, *slot):
        match slot:
            case ("vbroadcast", dest, src):
                for i in range(VLEN):
                    self.scratch_write[dest + i] = core.scratch[src]
            case ("multiply_add", dest, a, b, c):
                for i in range(VLEN):
                    mul = (core.scratch[a + i] * core.scratch[b + i]) % (2**32)
                    self.scratch_write[dest + i] = (mul + core.scratch[c + i]) % (2**32)
            case (op, dest, a1, a2):
                for i in range(VLEN):
                    self.alu(core, op, dest + i, a1 + i, a2 + i)
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")

    def load(self, core, *slot):
        match slot:
            case ("load", dest, addr):
                # print(dest, addr, core.scratch[addr])
                self.scratch_write[dest] = self.mem[core.scratch[addr]]
            case ("load_offset", dest, addr, offset):
                # Handy for treating vector dest and addr as a full block in the mini-compiler if you want
                self.scratch_write[dest + offset] = self.mem[
                    core.scratch[addr + offset]
                ]
            case ("vload", dest, addr):  # addr is a scalar
                addr = core.scratch[addr]
                for vi in range(VLEN):
                    self.scratch_write[dest + vi] = self.mem[addr + vi]
            case ("const", dest, val):
                self.scratch_write[dest] = (val) % (2**32)
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")

    def store(self, core, *slot):
        match slot:
            case ("store", addr, src):
                addr = core.scratch[addr]
                self.mem_write[addr] = core.scratch[src]
            case ("vstore", addr, src):  # addr is a scalar
                addr = core.scratch[addr]
                for vi in range(VLEN):
                    self.mem_write[addr + vi] = core.scratch[src + vi]
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")

    def flow(self, core, *slot):
        match slot:
            case ("select", dest, cond, a, b):
                self.scratch_write[dest] = (
                    core.scratch[a] if core.scratch[cond] != 0 else core.scratch[b]
                )
            case ("add_imm", dest, a, imm):
                self.scratch_write[dest] = (core.scratch[a] + imm) % (2**32)
            case ("vselect", dest, cond, a, b):
                for vi in range(VLEN):
                    self.scratch_write[dest + vi] = (
                        core.scratch[a + vi]
                        if core.scratch[cond + vi] != 0
                        else core.scratch[b + vi]
                    )
            case ("halt",):
                core.state = CoreState.STOPPED
            case ("pause",):
                if self.enable_pause:
                    core.state = CoreState.PAUSED
            case ("trace_write", val):
                core.trace_buf.append(core.scratch[val])
            case ("cond_jump", cond, addr):
                if core.scratch[cond] != 0:
                    core.pc = addr
            case ("cond_jump_rel", cond, offset):
                if core.scratch[cond] != 0:
                    core.pc += offset
            case ("jump", addr):
                core.pc = addr
            case ("jump_indirect", addr):
                core.pc = core.scratch[addr]
            case ("coreid", dest):
                self.scratch_write[dest] = core.id
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    def trace_post_step(self, instr, core):
        # You can add extra stuff to the trace if you want!
        for addr, (name, length) in self.debug_info.scratch_map.items():
            if any((addr + vi) in self.scratch_write for vi in range(length)):
                val = str(core.scratch[addr : addr + length])
                val = val.replace("[", "").replace("]", "")
                self.trace.write(
                    f'{{"name": "{val}", "cat": "op", "ph": "X", "pid": {len(self.cores) + core.id}, "tid": {BASE_ADDR_TID + addr}, "ts": {self.cycle}, "dur": 1 }},\n'
                )

    def trace_slot(self, core, slot, name, i):
        self.trace.write(
            f'{{"name": "{slot[0]}", "cat": "op", "ph": "X", "pid": {core.id}, "tid": {self.tids[(core.id, name, i)]}, "ts": {self.cycle}, "dur": 1, "args":{{"slot": "{str(slot)}", "named": "{str(self.rewrite_slot(slot))}" }} }},\n'
        )

    def step(self, instr: Instruction, core):
        """
        Execute all the slots in each engine for a single instruction bundle
        """
        ENGINE_FNS = {
            "alu": self.alu,
            "valu": self.valu,
            "load": self.load,
            "store": self.store,
            "flow": self.flow,
        }
        self.scratch_write = {}
        self.mem_write = {}
        for name, slots in instr.items():
            if name == "debug":
                if not self.enable_debug:
                    continue
                for slot in slots:
                    if slot[0] == "compare":
                        loc, key = slot[1], slot[2]
                        ref = self.value_trace[key]
                        res = core.scratch[loc]
                        assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
                    elif slot[0] == "vcompare":
                        loc, keys = slot[1], slot[2]
                        ref = [self.value_trace[key] for key in keys]
                        res = core.scratch[loc : loc + VLEN]
                        assert res == ref, (
                            f"{res} != {ref} for {keys} at pc={core.pc} loc={loc}"
                        )
                continue
            assert len(slots) <= SLOT_LIMITS[name]
            for i, slot in enumerate(slots):
                if self.trace is not None:
                    self.trace_slot(core, slot, name, i)
                ENGINE_FNS[name](core, *slot)
        for addr, val in self.scratch_write.items():
            core.scratch[addr] = val
        for addr, val in self.mem_write.items():
            self.mem[addr] = val

        if self.trace:
            self.trace_post_step(instr, core)

        del self.scratch_write
        del self.mem_write

    def __del__(self):
        if self.trace is not None:
            self.trace.write("]")
            self.trace.close()


@dataclass
class Tree:
    """
    An implicit perfect balanced binary tree with values on the nodes.
    """

    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        n_nodes = 2 ** (height + 1) - 1
        values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]
        return Tree(height, values)


@dataclass
class Input:
    """
    A batch of inputs, indices to nodes (starting as 0) and initial input
    values. We then iterate these for a specified number of rounds.
    """

    indices: list[int]
    values: list[int]
    rounds: int

    @staticmethod
    def generate(forest: Tree, batch_size: int, rounds: int):
        indices = [0 for _ in range(batch_size)]
        values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]
        return Input(indices, values, rounds)


HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


def myhash(a: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))

    return a


def reference_kernel(t: Tree, inp: Input):
    """
    Reference implementation of the kernel.

    A parallel tree traversal where at each node we set
    cur_inp_val = myhash(cur_inp_val ^ node_val)
    and then choose the left branch if cur_inp_val is even.
    If we reach the bottom of the tree we wrap around to the top.
    """
    for h in range(inp.rounds):
        for i in range(len(inp.indices)):
            idx = inp.indices[i]
            val = inp.values[i]
            val = myhash(val ^ t.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= len(t.values) else idx
            inp.values[i] = val
            inp.indices[i] = idx


def build_mem_image(t: Tree, inp: Input) -> list[int]:
    """
    Build a flat memory image of the problem.
    """
    header = 7
    extra_room = len(t.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    mem = [0] * (
        header + len(t.values) + len(inp.indices) + len(inp.values) + extra_room
    )
    forest_values_p = header
    inp_indices_p = forest_values_p + len(t.values)
    inp_values_p = inp_indices_p + len(inp.values)
    extra_room = inp_values_p + len(inp.values)

    mem[0] = inp.rounds
    mem[1] = len(t.values)
    mem[2] = len(inp.indices)
    mem[3] = t.height
    mem[4] = forest_values_p
    mem[5] = inp_indices_p
    mem[6] = inp_values_p
    mem[7] = extra_room

    mem[header:inp_indices_p] = t.values
    mem[inp_indices_p:inp_values_p] = inp.indices
    mem[inp_values_p:] = inp.values
    return mem


def myhash_traced(a: int, trace: dict[Any, int], round: int, batch_i: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
        trace[(round, batch_i, "hash_stage", i)] = a

    return a


def reference_kernel2(mem: list[int], trace: dict[Any, int] = {}):
    """
    Reference implementation of the kernel on a flat memory.
    """
    # This is the initial memory layout
    rounds = mem[0]
    n_nodes = mem[1]
    batch_size = mem[2]
    forest_height = mem[3]
    # Offsets into the memory which indices get added to
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    yield mem
    for h in range(rounds):
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            trace[(h, i, "idx")] = idx
            val = mem[inp_values_p + i]
            trace[(h, i, "val")] = val
            node_val = mem[forest_values_p + idx]
            trace[(h, i, "node_val")] = node_val
            val = myhash_traced(val ^ node_val, trace, h, i)
            trace[(h, i, "hashed_val")] = val
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            trace[(h, i, "next_idx")] = idx
            idx = 0 if idx >= n_nodes else idx
            trace[(h, i, "wrapped_idx")] = idx
            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx
    # You can add new yields or move this around for debugging
    # as long as it's matched by pause instructions.
    # The submission tests evaluate only on final memory.
    yield mem


from collections import defaultdict
import random
import unittest

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def write_read_sets(self, engine, slot):
        """
        Determine which scratch addresses an instruction reads and writes.
        You shouldn't really need to pay attention to this.
        """
        match engine, slot:
            case ("valu", ("vbroadcast", dest, src)):
                return [dest + i for i in range(VLEN)], [src]
            case ("valu", ("multiply_add", dest, a, b, c)):
                return [dest + i for i in range(VLEN)], [
                    x + i for i in range(VLEN) for x in (a, b, c)
                ]
            case ("alu", (_op, dest, a, b)):
                return [dest], [a, b]
            case ("valu", (_op, dest, a, b)):
                return [dest + i for i in range(VLEN)], [
                    x + i for i in range(VLEN) for x in (a, b)
                ]
            case ("load", ("vload", dest, addr)):
                return [dest + i for i in range(VLEN)], [addr]
            case ("load", ("const", dest, _)):
                return [dest], []
            case ("load", ("load_offset", dest, addr, offset)):
                return [dest + offset], [addr + offset]
            case ("load", (_op, dest, addr)):
                return [dest], [addr]
            case ("store", ("vstore", addr, src)):
                return [], [addr] + [src + i for i in range(VLEN)]
            case ("store", (_op, addr, src)):
                return [], [addr, src]
            case ("flow", ("vselect", dest, cond, a, b)):
                return [dest + i for i in range(VLEN)], [
                    x + i for i in range(VLEN) for x in (cond, a, b)
                ]
            case ("flow", ("select", dest, cond, a, b)):
                return [dest], [cond, a, b]
            case ("flow", ("add_imm", dest, a, _imm)):
                return [dest], [a]
            case ("debug", ("compare", loc, _key)):
                return [], [loc]
            case ("debug", ("vcompare", loc, _key)):
                return [], [loc + i for i in range(VLEN)]
        raise NotImplementedError(f"Unknown engine {engine} or slot {slot}")

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Pack multiple slots into instruction bundles using a packer that
        greedily packs slots as early as possible while respecting their
        dependencies. Where dependencies include the scratch address being
        written being no longer used. Writes to the same scratch address will
        have their order preserved, even where in theory a better packing
        algorithm could interleave them.

        You can get very far without understanding this further, so you don't really need to read it.
        """
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # Construct a graph of dependencies between slots
        # slot indices unblocked by each slot index
        unblocks = [[] for _ in slots]
        # last slot index to write to each address, anything that reads blocks on this
        last_writer = {}
        writer_readers = defaultdict(lambda: [])
        # number of blocks on each slot index, a slot has a block for each read with a last_writer
        blocked_count = [0 for _ in slots]
        for i, (engine, slot) in enumerate(slots):
            writes, reads = self.write_read_sets(engine, slot)
            # We block on everything reading from something we overwrite
            for addr in writes:
                for slot_i in writer_readers[addr]:
                    blocked_count[i] += 1
                    unblocks[slot_i].append(i)
            # And everything writing something we read
            for addr in reads:
                if addr in last_writer:
                    blocked_count[i] += 1
                    unblocks[last_writer[addr]].append(i)
                    writer_readers[addr].append(i)
            # And then reset the generation of our write
            for addr in writes:
                last_writer[addr] = i
                writer_readers[addr] = []

        instrs = []
        cur_instr = defaultdict(lambda: [])
        ready = set([i for i, c in enumerate(blocked_count) if c == 0])
        total_comitted = 0
        while True:
            # Pack with ready instructions
            committed = set()
            # Iterating a set here means essentially random prioritization
            for ready_i in ready:
                engine, slot = slots[ready_i]
                if len(cur_instr[engine]) < SLOT_LIMITS[engine]:
                    cur_instr[engine].append(slot)
                    committed.add(ready_i)
            if len(committed) == 0:
                break
            ready -= committed
            total_comitted += len(committed)
            # Commit the instruction
            instrs.append(dict(cur_instr))
            cur_instr = defaultdict(lambda: [])
            # Unblock based on comitted
            for i in committed:
                for j in unblocks[i]:
                    blocked_count[j] -= 1
                    if blocked_count[j] == 0:
                        ready.add(j)

        # for instr in instrs:
        #     print(instr)
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash_vector(self, val_hash_addr, vtmp1, vtmp2, round, i, hash_consts):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vconst1, vconst3 = hash_consts[hi]
            # Use pre-computed vector constants instead of broadcasting
            slots.append(("valu", (op1, vtmp1, val_hash_addr, vconst1)))
            slots.append(("valu", (op3, vtmp2, val_hash_addr, vconst3)))
            slots.append(("valu", (op2, val_hash_addr, vtmp1, vtmp2)))
            cmp_keys = [(round, i + j, "hash_stage", hi) for j in range(VLEN)]
            slots.append(("debug", ("vcompare", val_hash_addr, cmp_keys)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel implementation with loop unrolling to improve parallelism.
        The key insight is to allocate separate scratch space for multiple chunks
        so the VLIW packer can interleave their operations.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-allocate vector constants (these are shared)
        vzero = self.alloc_scratch("vzero", VLEN)
        self.add("valu", ("vbroadcast", vzero, zero_const))
        vone = self.alloc_scratch("vone", VLEN)
        self.add("valu", ("vbroadcast", vone, one_const))
        vtwo = self.alloc_scratch("vtwo", VLEN)
        self.add("valu", ("vbroadcast", vtwo, two_const))
        
        # Pre-compute hash constants as vectors to avoid repeated broadcasts
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vconst1 = self.alloc_scratch(f"vhash1_{hi}", VLEN)
            self.add("valu", ("vbroadcast", vconst1, self.scratch_const(val1)))
            vconst3 = self.alloc_scratch(f"vhash3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", vconst3, self.scratch_const(val3)))
            hash_consts.append((vconst1, vconst3))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots
        
        # OPTIMIZATION: Unroll the loop and allocate separate scratch for multiple chunks
        # This allows the VLIW packer to interleave operations from different chunks
        UNROLL_FACTOR = 32  # Process 32 VLEN chunks simultaneously
        
        # Allocate scratch space for each unrolled iteration
        scratch_vars = []
        for u in range(UNROLL_FACTOR):
            scratch_vars.append({
                'batch_i': self.alloc_scratch(f"batch_i_{u}"),
                'tmp1': self.alloc_scratch(f"tmp1_{u}"),
                'tmp2': self.alloc_scratch(f"tmp2_{u}"),
                'tmp_idx': self.alloc_scratch(f"tmp_idx_{u}", VLEN),
                'tmp_val': self.alloc_scratch(f"tmp_val_{u}", VLEN),
                'vtmp1': self.alloc_scratch(f"vtmp1_{u}", VLEN),
                'vtmp2': self.alloc_scratch(f"vtmp2_{u}", VLEN),
                'tmp_node_val': self.alloc_scratch(f"tmp_node_val_{u}", VLEN),
            })

        def vcmp(round, i, key, val):
            body.append(
                ("debug", ("vcompare", val, [(round, i + j, key) for j in range(VLEN)]))
            )

        # Process in chunks with unrolling
        for round in range(rounds):
            for chunk_start in range(0, batch_size, VLEN * UNROLL_FACTOR):
                # Generate operations for all unrolled iterations
                for u in range(UNROLL_FACTOR):
                    i = chunk_start + u * VLEN
                    if i >= batch_size:
                        break
                    
                    sv = scratch_vars[u]
                    batch_i = sv['batch_i']
                    stmp1 = sv['tmp1']
                    stmp2 = sv['tmp2']
                    tmp_idx = sv['tmp_idx']
                    tmp_val = sv['tmp_val']
                    vtmp1 = sv['vtmp1']
                    vtmp2 = sv['vtmp2']
                    tmp_node_val = sv['tmp_node_val']
                    
                    # Load batch index
                    body.append(("load", ("const", batch_i, i)))
                    
                    # idx = mem[inp_indices_p + i]
                    body.append(("alu", ("+", stmp1, self.scratch["inp_indices_p"], batch_i)))
                    body.append(("load", ("vload", tmp_idx, stmp1)))
                    vcmp(round, i, "idx", tmp_idx)
                    
                    # val = mem[inp_values_p + i]
                    body.append(("alu", ("+", stmp1, self.scratch["inp_values_p"], batch_i)))
                    body.append(("load", ("vload", tmp_val, stmp1)))
                    vcmp(round, i, "val", tmp_val)
                    
                    # node_val = mem[forest_values_p + idx]
                    body.append(("valu", ("vbroadcast", vtmp1, self.scratch["forest_values_p"])))
                    body.append(("valu", ("+", tmp_node_val, vtmp1, tmp_idx)))
                    for vi in range(VLEN):
                        body.append(("load", ("load_offset", tmp_node_val, tmp_node_val, vi)))
                    vcmp(round, i, "node_val", tmp_node_val)
                    
                    # val = myhash(val ^ node_val)
                    body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    body.extend(self.build_hash_vector(tmp_val, vtmp1, vtmp2, round, i, hash_consts))
                    vcmp(round, i, "hashed_val", tmp_val)
                    
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    body.append(("valu", ("%", vtmp1, tmp_val, vtwo)))
                    body.append(("valu", ("==", vtmp1, vtmp1, vzero)))
                    body.append(("flow", ("vselect", vtmp2, vtmp1, vone, vtwo)))
                    body.append(("valu", ("*", tmp_idx, tmp_idx, vtwo)))
                    body.append(("valu", ("+", tmp_idx, tmp_idx, vtmp2)))
                    vcmp(round, i, "next_idx", tmp_idx)
                    
                    # idx = 0 if idx >= n_nodes else idx
                    body.append(("valu", ("vbroadcast", vtmp2, self.scratch["n_nodes"])))
                    body.append(("valu", ("<", vtmp1, tmp_idx, vtmp2)))
                    body.append(("flow", ("vselect", tmp_idx, vtmp1, tmp_idx, vzero)))
                    vcmp(round, i, "wrapped_idx", tmp_idx)
                    
                    # mem[inp_indices_p + i] = idx
                    body.append(("alu", ("+", stmp2, self.scratch["inp_indices_p"], batch_i)))
                    body.append(("store", ("vstore", stmp2, tmp_idx)))
                    
                    # mem[inp_values_p + i] = val
                    body.append(("alu", ("+", stmp2, self.scratch["inp_values_p"], batch_i)))
                    body.append(("store", ("vstore", stmp2, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_correctness(self):
        for batch in range(1, 3):
            for forest_height in range(3):
                do_kernel_test(
                    forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
                )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


@lru_cache(maxsize=None)
def kernel_builder(forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb


def do_kernel_test_simple(forest_height: int, rounds: int, batch_size: int):
    print(f"Testing {forest_height=}, {rounds=}, {batch_size=}")
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = kernel_builder(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"
    print("CYCLES: ", machine.cycle)
    return machine.cycle


class CorrectnessTests(unittest.TestCase):
    def test_kernel_correctness(self):
        for i in range(8):
            do_kernel_test_simple(10, 16, 256)


BASELINE = 18532


@lru_cache(maxsize=None)
def cycles():
    try:
        res = do_kernel_test_simple(10, 16, 256)
        print("Speedup over baseline: ", BASELINE / res)
        return res
    except AssertionError as e:
        return BASELINE * 2


class SpeedTests(unittest.TestCase):
    def test_kernel_any_faster(self):
        assert cycles() < BASELINE

    def test_kernel_2x_faster(self):
        assert cycles() < BASELINE / 2

    def test_kernel_4x_faster(self):
        assert cycles() < BASELINE / 4.0

    def test_kernel_4_1x_faster(self):
        assert cycles() < BASELINE / 4.1

    def test_kernel_4_2x_faster(self):
        assert cycles() < BASELINE / 4.2

    def test_kernel_4_3x_faster(self):
        assert cycles() < BASELINE / 4.3

    def test_kernel_4_4x_faster(self):
        assert cycles() < BASELINE / 4.4

    def test_kernel_4_5x_faster(self):
        assert cycles() < BASELINE / 4.5

    def test_kernel_4_6x_faster(self):
        assert cycles() < BASELINE / 4.6

    def test_kernel_4_7x_faster(self):
        assert cycles() < BASELINE / 4.7

    def test_kernel_4_8x_faster(self):
        assert cycles() < BASELINE / 4.8

    def test_kernel_4_9x_faster(self):
        assert cycles() < BASELINE / 4.9

    def test_kernel_5x_faster(self):
        assert cycles() < BASELINE / 5.0

    def test_kernel_5_1x_faster(self):
        assert cycles() < BASELINE / 5.1

    def test_kernel_5_2x_faster(self):
        assert cycles() < BASELINE / 5.2

    def test_kernel_5_3x_faster(self):
        assert cycles() < BASELINE / 5.3

    def test_kernel_5_4x_faster(self):
        assert cycles() < BASELINE / 5.4

    def test_kernel_5_5x_faster(self):
        assert cycles() < BASELINE / 5.5

    def test_kernel_5_7x_faster(self):
        assert cycles() < BASELINE / 5.7

    def test_kernel_5_8x_faster(self):
        assert cycles() < BASELINE / 5.8

    def test_kernel_5_9x_faster(self):
        assert cycles() < BASELINE / 5.9

    def test_kernel_6x_faster(self):
        assert cycles() < BASELINE / 6.0

    def test_kernel_6_2x_faster(self):
        assert cycles() < BASELINE / 6.2

    def test_kernel_6_3x_faster(self):
        assert cycles() < BASELINE / 6.3

    def test_kernel_6_4x_faster(self):
        assert cycles() < BASELINE / 6.4

    def test_kernel_6_6x_faster(self):
        assert cycles() < BASELINE / 6.6

    def test_kernel_6_7x_faster(self):
        assert cycles() < BASELINE / 6.7

    def test_kernel_6_9x_faster(self):
        assert cycles() < BASELINE / 6.9

    def test_kernel_7x_faster(self):
        assert cycles() < BASELINE / 7.0

    def test_kernel_7_2x_faster(self):
        assert cycles() < BASELINE / 7.2

    def test_kernel_7_3x_faster(self):
        assert cycles() < BASELINE / 7.3

    def test_kernel_7_5x_faster(self):
        assert cycles() < BASELINE / 7.5

    def test_kernel_7_7x_faster(self):
        assert cycles() < BASELINE / 7.7

    def test_kernel_7_8x_faster(self):
        assert cycles() < BASELINE / 7.8

    def test_kernel_8x_faster(self):
        assert cycles() < BASELINE / 8.0

    def test_kernel_8_2x_faster(self):
        assert cycles() < BASELINE / 8.2

    def test_kernel_8_4x_faster(self):
        assert cycles() < BASELINE / 8.4

    def test_kernel_8_5x_faster(self):
        assert cycles() < BASELINE / 8.5

    def test_kernel_8_7x_faster(self):
        assert cycles() < BASELINE / 8.7

    def test_kernel_8_9x_faster(self):
        assert cycles() < BASELINE / 8.9

    def test_kernel_9_1x_faster(self):
        assert cycles() < BASELINE / 9.1

    def test_kernel_9_3x_faster(self):
        assert cycles() < BASELINE / 9.3

    def test_kernel_9_5x_faster(self):
        assert cycles() < BASELINE / 9.5

    def test_kernel_9_7x_faster(self):
        assert cycles() < BASELINE / 9.7

    def test_kernel_9_9x_faster(self):
        assert cycles() < BASELINE / 9.9

    def test_kernel_10_2x_faster(self):
        assert cycles() < BASELINE / 10.2

    def test_kernel_10_4x_faster(self):
        assert cycles() < BASELINE / 10.4

    def test_kernel_10_6x_faster(self):
        assert cycles() < BASELINE / 10.6

    def test_kernel_10_8x_faster(self):
        assert cycles() < BASELINE / 10.8

    def test_kernel_11_1x_faster(self):
        assert cycles() < BASELINE / 11.1

    def test_kernel_11_3x_faster(self):
        assert cycles() < BASELINE / 11.3

    def test_kernel_11_6x_faster(self):
        assert cycles() < BASELINE / 11.6

    def test_kernel_11_8x_faster(self):
        assert cycles() < BASELINE / 11.8

    def test_kernel_12_1x_faster(self):
        assert cycles() < BASELINE / 12.1

    def test_kernel_12_3x_faster(self):
        assert cycles() < BASELINE / 12.3

    def test_kernel_12_6x_faster(self):
        assert cycles() < BASELINE / 12.6

    def test_kernel_12_9x_faster(self):
        assert cycles() < BASELINE / 12.9

    def test_kernel_13_2x_faster(self):
        assert cycles() < BASELINE / 13.2

    def test_kernel_13_5x_faster(self):
        assert cycles() < BASELINE / 13.5

    def test_kernel_13_8x_faster(self):
        assert cycles() < BASELINE / 13.8

    def test_kernel_14_1x_faster(self):
        assert cycles() < BASELINE / 14.1

    def test_kernel_14_4x_faster(self):
        assert cycles() < BASELINE / 14.4

    def test_kernel_14_7x_faster(self):
        assert cycles() < BASELINE / 14.7

    def test_kernel_15x_faster(self):
        assert cycles() < BASELINE / 15.0


if __name__ == "__main__":
    unittest.main()
