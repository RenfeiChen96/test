import os, sys, inspect
from functools import lru_cache
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
import random
import unittest

Engine = Literal["alu", "load", "store", "flow"]
Instruction = dict[Engine, list[tuple]]


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
N_CORES = 1
SCRATCH_SIZE = 1536
BASE_ADDR_TID = 100000


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
    scratch_map: dict[int, (str, int)]


class Machine:
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

    def rewrite_slot(self, slot):
        return tuple(
            self.debug_info.scratch_map.get(s, (None, None))[0] or s for s in slot
        )

    def setup_trace(self):
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

    def trace_slot(self, core, slot, name, i):
        self.trace.write(
            f'{{"name": "{slot[0]}", "cat": "op", "ph": "X", "pid": {core.id}, "tid": {self.tids[(core.id, name, i)]}, "ts": {self.cycle}, "dur": 1, "args":{{"slot": "{str(slot)}", "named": "{str(self.rewrite_slot(slot))}" }} }},\n'
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
                self.scratch_write[dest] = self.mem[core.scratch[addr]]
            case ("load_offset", dest, addr, offset):
                self.scratch_write[dest + offset] = self.mem[
                    core.scratch[addr + offset]
                ]
            case ("vload", dest, addr):
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
            case ("vstore", addr, src):
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
        for addr, (name, length) in self.debug_info.scratch_map.items():
            if any((addr + vi) in self.scratch_write for vi in range(length)):
                val = str(core.scratch[addr : addr + length])
                val = val.replace("[", "").replace("]", "")
                self.trace.write(
                    f'{{"name": "{val}", "cat": "op", "ph": "X", "pid": {len(self.cores) + core.id}, "tid": {BASE_ADDR_TID + addr}, "ts": {self.cycle}, "dur": 1 }},\n'
                )

    def step(self, instr: Instruction, core):
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
        if getattr(self, "trace", None) is not None:
            self.trace.write("]")
            self.trace.close()


@dataclass
class Tree:
    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        n_nodes = 2 ** (height + 1) - 1
        values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]
        return Tree(height, values)


@dataclass
class Input:
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


def myhash_traced(a: int, trace: dict[Any, int], round: int, batch_i: int) -> int:
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
    rounds = mem[0]
    n_nodes = mem[1]
    batch_size = mem[2]
    forest_height = mem[3]
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
    yield mem


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
        match engine, slot:
            case ("valu", ("vbroadcast", dest, src)):
                return [dest + i for i in range(VLEN)], [src]
            case ("valu", ("multiply_add", dest, a, b, c)):
                return [dest + i for i in range(VLEN)], [x + i for i in range(VLEN) for x in (a, b, c)]
            case ("alu", (_op, dest, a, b)):
                return [dest], [a, b]
            case ("valu", (_op, dest, a, b)):
                return [dest + i for i in range(VLEN)], [x + i for i in range(VLEN) for x in (a, b)]
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
                return [dest + i for i in range(VLEN)], [x + i for i in range(VLEN) for x in (cond, a, b)]
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
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        unblocks = [[] for _ in slots]
        last_writer = {}
        writer_readers = defaultdict(lambda: [])
        blocked_count = [0 for _ in slots]
        for i, (engine, slot) in enumerate(slots):
            writes, reads = self.write_read_sets(engine, slot)
            for addr in writes:
                for slot_i in writer_readers[addr]:
                    blocked_count[i] += 1
                    unblocks[slot_i].append(i)
            for addr in reads:
                if addr in last_writer:
                    blocked_count[i] += 1
                    unblocks[last_writer[addr]].append(i)
                    writer_readers[addr].append(i)
            for addr in writes:
                last_writer[addr] = i
                writer_readers[addr] = []

        instrs = []
        cur_instr = defaultdict(lambda: [])
        ready = set([i for i, c in enumerate(blocked_count) if c == 0])
        while True:
            committed = set()
            for ready_i in ready:
                engine, slot = slots[ready_i]
                if len(cur_instr[engine]) < SLOT_LIMITS[engine]:
                    cur_instr[engine].append(slot)
                    committed.add(ready_i)
            if len(committed) == 0:
                break
            ready -= committed
            instrs.append(dict(cur_instr))
            cur_instr = defaultdict(lambda: [])
            for i in committed:
                for j in unblocks[i]:
                    blocked_count[j] -= 1
                    if blocked_count[j] == 0:
                        ready.add(j)
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # Scalars for temporary address math
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Header fields loaded from memory header
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

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector constants (once)
        vzero = self.alloc_scratch("vzero", VLEN)
        self.add("valu", ("vbroadcast", vzero, zero_const))
        vone = self.alloc_scratch("vone", VLEN)
        self.add("valu", ("vbroadcast", vone, one_const))
        vtwo = self.alloc_scratch("vtwo", VLEN)
        self.add("valu", ("vbroadcast", vtwo, two_const))

        # n_nodes and forest base as vectors (once)
        vn_nodes = self.alloc_scratch("vn_nodes", VLEN)
        self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))
        vforest_base = self.alloc_scratch("vforest_base", VLEN)
        self.add("valu", ("vbroadcast", vforest_base, self.scratch["forest_values_p"]))

        # Pre-broadcast hash constants as vectors (once)
        vhash1 = []
        vhash3 = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v1 = self.alloc_scratch(f"vhash1_{hi}", VLEN)
            v3 = self.alloc_scratch(f"vhash3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v1, self.scratch_const(val1)))
            self.add("valu", ("vbroadcast", v3, self.scratch_const(val3)))
            vhash1.append(v1)
            vhash3.append(v3)

        # Match first yield in reference_kernel2
        self.add("flow", ("pause",))

        body: list[tuple[Engine, tuple]] = []

        tiles = batch_size // VLEN
        # Per-tile scratch
        tile_data = []
        for t in range(tiles):
            base_i = t * VLEN
            batch_it = self.alloc_scratch(f"batch_i_{t}")
            self.add("load", ("const", batch_it, base_i))

            addr_idx = self.alloc_scratch(f"addr_idx_{t}")
            addr_val = self.alloc_scratch(f"addr_val_{t}")

            v_idx = self.alloc_scratch(f"tmp_idx_{t}", VLEN)
            v_val = self.alloc_scratch(f"tmp_val_{t}", VLEN)
            v_addr = self.alloc_scratch(f"node_addr_{t}", VLEN)
            v_node = self.alloc_scratch(f"tmp_node_val_{t}", VLEN)
            vtmp1 = self.alloc_scratch(f"vtmp1_{t}", VLEN)
            vtmp2 = self.alloc_scratch(f"vtmp2_{t}", VLEN)
            tile_data.append((batch_it, addr_idx, addr_val, v_idx, v_val, v_addr, v_node, vtmp1, vtmp2))

        for round_i in range(rounds):
            for t in range(tiles):
                batch_it, addr_idx, addr_val, v_idx, v_val, v_addr, v_node, vtmp1, vtmp2 = tile_data[t]
                # Addresses and loads
                body.append(("alu", ("+", addr_idx, self.scratch["inp_indices_p"], batch_it)))
                body.append(("load", ("vload", v_idx, addr_idx)))
                body.append(("alu", ("+", addr_val, self.scratch["inp_values_p"], batch_it)))
                body.append(("load", ("vload", v_val, addr_val)))

                # Compute gather addresses and gather
                body.append(("valu", ("+", v_addr, vforest_base, v_idx)))
                for vi in range(VLEN):
                    body.append(("load", ("load_offset", v_node, v_addr, vi)))

                # Mix node value into hash input
                body.append(("valu", ("^", v_val, v_val, v_node)))

                # Hash rounds with pre-broadcast constants
                for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                    body.append(("valu", (op1, vtmp1, v_val, vhash1[hi])))
                    body.append(("valu", (op3, vtmp2, v_val, vhash3[hi])))
                    body.append(("valu", (op2, v_val, vtmp1, vtmp2)))

                # Next index: idx = (idx<<1) + (1 + (val & 1))
                body.append(("valu", ("&", vtmp1, v_val, vone)))       # parity
                body.append(("valu", ("+", vtmp2, vone, vtmp1)))       # addend 1/2
                body.append(("valu", ("<<", v_idx, v_idx, vone)))      # idx<<1
                body.append(("valu", ("+", v_idx, v_idx, vtmp2)))      # add addend

                # Wrap: keep idx if idx < n_nodes else 0
                body.append(("valu", ("<", vtmp1, v_idx, vn_nodes)))
                body.append(("flow", ("vselect", v_idx, vtmp1, v_idx, vzero)))

                # Stores
                body.append(("store", ("vstore", addr_idx, v_idx)))
                body.append(("store", ("vstore", addr_val, v_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Match second yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


def build_mem_image(t: Tree, inp: Input) -> list[int]:
    header = 7
    extra_room = len(t.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    mem = [0] * (header + len(t.values) + len(inp.indices) + len(inp.values) + extra_room)
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


@lru_cache(maxsize=None)
def kernel_builder(forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"Testing {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = kernel_builder(forest.height, len(forest.values), len(inp.indices), rounds)

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
    print("CYCLES: ", machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)


# This baseline corresponds to a naive (unscheduled) implementation
BASELINE = 18532


@lru_cache(maxsize=None)
def cycles():
    try:
        res = do_kernel_test(10, 16, 256)
        print("Speedup over baseline: ", BASELINE / res)
        return res
    except AssertionError:
        return BASELINE * 2


class SpeedTests(unittest.TestCase):
    def test_kernel_any_faster(self):
        assert cycles() < BASELINE

    def test_kernel_2x_faster(self):
        assert cycles() < BASELINE / 2


if __name__ == "__main__":
    unittest.main()