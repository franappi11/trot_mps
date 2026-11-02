"""Where the time of one MPS-CPMC block goes: a sampling jax.profiler trace of a production block, mapped onto the
compiled program (its optimized HLO) and from there onto the source functions and lines.

A production block of the doped 8x8 at chi_T 2048 is one XLA call of ~40 min and ~10^7 kernels, too many to trace
whole (the trace viewer keeps 10^6 events). ex_6x6_doped_diag.py --profile (so also ex_8x8_doped.py --profile) traces
one block of a run in windows instead: jax.profiler traces of --profile-window seconds every --profile-every seconds,
started and stopped by a thread (WindowTracer) while the main thread waits on the device. The traced block is the
program the production run compiled: with TROT_BLOCKS_PER_CALL=1 and the same flags it comes from the persistent
compilation cache, so N_EQL=1 N_BLOCKS=1 costs two blocks and no compile. Block 1 loads the program and warms up,
block --profile-block (default 2) is traced (BlockProfiler, run_qmc's chunk hook). Before it the hook writes the
compiled block's optimized HLO: Compiled.as_text of the jitted run_blocks that trot.driver.make_run_blocks built for
run_qmc (captured while it builds it), i.e. the executable the run used, without a reload. --profile-at places the
windows at given seconds into the block instead (fewer profiler sessions: see WindowTracer). The profiler slows the
traced windows somewhat (CUPTI on every kernel): compare the block's dt[s/bl] with an untraced one.

analyze() then reads <dir>/windows.json, run_blocks.hlo.txt and the windows' w<k>/plugins/profile/*/*.trace.json.gz
and reports, writing report.txt and summary.json next to them:

  * the windows: where they sit in the block, their kernels, the GPU busy fraction (union of the kernel intervals over
    the window) and the outermost loop they ran in;
  * the block by loop nest: every kernel belongs to an HLO instruction (inside a CUDA graph by its name, the sanitised
    fusion name, a cuBLAS kernel there by data flow, Hlo.resolve; outside by its hlo_op, e.g. a cuSOLVER call), and
    every instruction to a nest of while loops (the half-step scan, the measurement's chunk map, the per-walker szsz
    map, ...; identical site steps are one deduplicated computation, called per site). Per loop: kernel time, idle
    time before its kernels (launch gaps, while-loop predicates read by the host), kernels, all per block: each window
    stands for the time until the next one starts;
  * per outermost loop the top source functions with their callers (the stack frames JAX records in the HLO metadata,
    printed with the module; without them source_file / source_line and the enclosing def, or the op_name), and
    overall the top source lines and HLO instructions (opcode, fusion kind or custom-call target, shape);
  * kernel classes (solver, gemm, fusion, copy) per outermost loop.

The same report from the saved files, e.g. after a change to this analysis (stdlib only, no JAX or GPU needed):

    sbatch --time=00:30:00 --cpus-per-task=2 --mem=16G -o slurm/block_profile-%j.out \\
        --export=ALL,TARGET=block_profile.py run_mps_sweep.sh <out>/profile

A run that ended inside the traced block (7358122 segfaulted stopping its 4th window) leaves the windows' traces
only: the HLO dump and windows.json are written before the windows and after each now, but for such a run
--every / --window / --block rebuild windows.json from the window folders (nominal times) and --hlo-from-xplane
takes the HLO out of the first window's xplane.pb (needs jaxlib: run_mps_sweep.sh has it).
"""

import argparse
import ast
import contextlib
import dataclasses
import faulthandler
import gzip
import json
import linecache
import math
import os
import re
import threading
import time
import traceback
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

HLO_FILE = "run_blocks.hlo.txt"
WINDOWS_FILE = "windows.json"
# kernel classes by kernel name (bench_mps_cpmc.summarize_trace's)
KERNEL_CLASSES = (("solver", r"cusolver|syevd|syevj|jacobi|geqrf|orgqr|potrf|getrf|gesvd|trsm|cholesky|larf"),
                  ("gemm", r"gemm|cutlass|cublas|xmma|matmul|dot"), ("copy", r"memcpy|memset|copy"),
                  ("fusion", r"fusion|loop|reduce|select|broadcast|slice|scatter|gather|concat|transpose|iota"))


def kernel_class(name):
    return next((label for label, pattern in KERNEL_CLASSES if re.search(pattern, name, re.IGNORECASE)), "other")


# ---------------------------------------------------------------------------------------------------------------------
# tracing a running block
# ---------------------------------------------------------------------------------------------------------------------

def check(block, total):
    """SystemExit unless block `block` of a run of `total` blocks can be traced by a BlockProfiler."""
    from trot import driver as trot_driver

    if trot_driver.BLOCKS_PER_CALL != 1:
        raise SystemExit("--profile needs TROT_BLOCKS_PER_CALL=1: one block per compiled call with the hook between "
                         "them, and the production run's compiled program from the persistent cache")
    if not 2 <= block <= total:
        raise SystemExit(f"--profile-block {block} must be in [2, {total}] (N_EQL + N_BLOCKS): block 1 loads the "
                         "compiled program and warms up")


class WindowTracer:
    """jax.profiler traces of `window` seconds between start() and stop(), each into its own directory <path>/w<k>,
    from a thread, so the windows cut through one long XLA call: at the seconds `at` after start() if given (window
    k at at[k]), else every `every` seconds (slot k at k * every; a window that misses its slot because the previous
    export took long takes the next free one). on_window(windows) is called after every window.

    Every window is a profiler session of its own. Job 7358122 segfaulted when the 4th session of a block (12 s every
    150 s) stopped, while the 3 before it had worked: keep the sessions few (`at`)."""

    def __init__(self, path, every, window, at=None, on_window=None):
        self.path, self.every, self.window = Path(path), float(every), float(window)
        self.at = sorted(float(t) for t in at) if at else None
        self.on_window = on_window
        self.windows, self.error, self.t0 = [], None, None
        self._done = threading.Event()
        self._thread = None

    def start(self):
        faulthandler.enable(all_threads=True)  # a crash inside the profiler prints every thread's Python stack
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="window-tracer", daemon=True)
        self._thread.start()

    def _slot(self, k):
        """Start of window k in seconds after start(), None after the last one."""
        if self.at is None:
            return k * self.every
        return self.at[k] if k < len(self.at) else None

    def stop(self):
        """Ends the windows (a running one stops now) and returns them: slot k, dir, start and stop in seconds from
        start(), and the seconds its export took."""
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        return self.windows

    @staticmethod
    def _options(first):
        """No Python tracer (the main thread only waits on the device); the HLO protos (large, the same in every
        window, and slow to export: the device keeps being traced meanwhile) in the first window's xplane only, and
        only if the HLO text was not written."""
        import jax

        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.enable_hlo_proto = first
        return options

    def _run(self):
        import jax

        k = 0
        try:
            while self._slot(k) is not None and not self._done.wait(
                    max(0.0, self.t0 + self._slot(k) - time.perf_counter())):
                directory = self.path / f"w{k:03d}"
                first = not self.windows and not (self.path / HLO_FILE).exists()  # no HLO text: keep the protos
                jax.profiler.start_trace(str(directory), profiler_options=self._options(first))
                start = time.perf_counter() - self.t0
                self._done.wait(self.window)
                stop = time.perf_counter() - self.t0
                jax.profiler.stop_trace()
                self.windows.append(dict(k=k, dir=directory.name, start=start, stop=stop,
                                         export=time.perf_counter() - self.t0 - stop))
                if self.on_window is not None:
                    self.on_window(self.windows)
                k += 1
                if self.at is None:
                    k = max(k, math.ceil((time.perf_counter() - self.t0) / self.every))
        except Exception as error:  # the run goes on without the remaining windows
            self.error = f"{type(error).__name__}: {error}"
            print(f"[profile] window tracing stopped: {self.error}", flush=True)
            with contextlib.suppress(Exception):
                jax.profiler.stop_trace()


class BlockProfiler:
    """trot.driver.run_qmc chunk hook that, before block `block` (counted over equilibration and sampling), writes the
    compiled block's optimized HLO, then traces the block with a WindowTracer into `path`, recording the windows in
    windows.json after every one of them (a crash keeps what was traced) and the block's time at its end. Needs
    TROT_BLOCKS_PER_CALL=1 (check()): every run_blocks call is one block and the hook runs between them.

        profiler = BlockProfiler(run, path)
        with profiler.capture():
            result = driver.run_prepared(run, ..., chunk_hook=profiler)
        profiler.report()
    """

    def __init__(self, run, path, *, block=2, every=150.0, window=12.0, at=None):
        self.run, self.path, self.block = run, Path(path), int(block)
        self.path.mkdir(parents=True, exist_ok=True)
        self.tracer = WindowTracer(self.path, every, window, at=at, on_window=lambda windows: self._write())
        self.built = []  # the jitted run_blocks of trot.driver.make_run_blocks, recorded by capture()
        self.started = None
        self.record = dict(block=self.block, phase=None, block_seconds=None, every=self.tracer.every,
                           window=self.tracer.window, at=self.tracer.at, windows=[], error=None,
                           job=os.environ.get("SLURM_JOB_ID"))

    def _write(self, **values):
        """windows.json: the record so far (block_seconds None until the block has ended)."""
        self.record.update(values, windows=list(self.tracer.windows), error=self.tracer.error)
        (self.path / WINDOWS_FILE).write_text(json.dumps(self.record, indent=1) + "\n")

    @contextlib.contextmanager
    def capture(self):
        """Records the jitted run_blocks that trot.driver.make_run_blocks builds for run_qmc, which keeps no handle on
        it: its lower().compile() after the block returns the executable the run used, already loaded."""
        from trot import driver as trot_driver

        make = trot_driver.make_run_blocks

        def recording(**kwargs):
            self.built.append(make(**kwargs))
            return self.built[-1]

        trot_driver.make_run_blocks = recording
        try:
            yield self
        finally:
            trot_driver.make_run_blocks = make

    def __call__(self, phase, blocks_done, state, meas_ctx, prop_ctx):
        if blocks_done == self.block - 1:
            self._write(**self.dump_hlo(state, meas_ctx, prop_ctx))
            when = (f"at {', '.join(f'{t:g}' for t in self.tracer.at)} s" if self.tracer.at else
                    f"every {self.tracer.every:g} s")
            print(f"[profile] tracing block {self.block} in windows of {self.tracer.window:g} s {when} into "
                  f"{self.path}", flush=True)
            self.started = time.perf_counter()
            self.tracer.start()
        elif blocks_done == self.block and self.started is not None:
            seconds = time.perf_counter() - self.started
            windows = self.tracer.stop()
            print(f"[profile] block {self.block} took {seconds:.1f} s with {len(windows)} windows", flush=True)
            self._write(phase=phase, block_seconds=seconds)
        return None

    def dump_hlo(self, state, meas_ctx, prop_ctx):
        """Writes the optimized HLO of the run's compiled block to <path>/run_blocks.hlo.txt (lowered with the
        arguments of the run's calls, so it is the jitted function's own, loaded executable); returns XLA's memory
        analysis of it (bytes) for windows.json."""
        if not self.built:
            print("[profile] no run_blocks was captured: no HLO", flush=True)
            return {}
        run, start = self.run, time.perf_counter()
        try:
            compiled = self.built[-1].lower(state, ham_data=run.ham_data, trial_data=run.trial, meas_ctx=meas_ctx,
                                            prop_ctx=prop_ctx, n_blocks=1).compile()
            (self.path / HLO_FILE).write_text(compiled.as_text())
        except Exception as error:
            print(f"[profile] no HLO ({type(error).__name__}: {error})", flush=True)
            return {}
        print(f"[profile] compiled block's HLO in {self.path / HLO_FILE} ({time.perf_counter() - start:.1f} s)",
              flush=True)
        try:
            memory = compiled.memory_analysis()
        except Exception:
            memory = None
        names = ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                 "generated_code_size_in_bytes")
        return dict(memory={name: int(getattr(memory, name)) for name in names if hasattr(memory, name)})

    def report(self):
        """analyze() of the traced block, if it was traced; a failure only prints (the run's results are saved)."""
        if self.started is None:
            print(f"[profile] block {self.block} was not traced (the run has fewer blocks?)", flush=True)
            return
        try:
            analyze(self.path)
        except Exception:
            traceback.print_exc()
            print(f"[profile] analysis failed; rerun it on the saved files: python block_profile.py {self.path}",
                  flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# the compiled program
# ---------------------------------------------------------------------------------------------------------------------

_HEADER = re.compile(r"(?:ENTRY\s+)?%?([^\s(]+)")
_INSTRUCTION = re.compile(r"\s+(?:ROOT\s+)?%?([^\s=]+)\s*=\s*(.*)")
_OPCODE = re.compile(r"\s([a-z][a-z0-9\-]*)\(")
_CALLED = re.compile(r"\b(?:calls|to_apply|body|condition|true_computation|false_computation|select|scatter)="
                     r"%?([\w.\-]+)")
_CALLED_SETS = re.compile(r"\b(?:branch_computations|called_computations)=\{([^}]*)\}")
_TRIP_COUNT = re.compile(r'"known_trip_count":\{"n":"(\d+)"')
_KIND = re.compile(r"\bkind=(k\w+)")
_TARGET = re.compile(r'custom_call_target="([^"]*)"')
_OP_NAME = re.compile(r'op_name="((?:[^"\\]|\\.)*)"')
_SOURCE_FILE = re.compile(r'source_file="([^"]*)"')
_SOURCE_LINE = re.compile(r"\bsource_line=(\d+)")
_FRAME_ID = re.compile(r"\bstack_frame_id=(\d+)")
# the module's stack frame index, printed with the computations: FileNames, FunctionNames ('<id> "<name>"'),
# FileLocations ('<id> {file_name_id=.. function_name_id=.. line=..}') and StackFrames ('<id> {file_location_id=..
# parent_frame_id=..}'); an instruction's metadata names its innermost frame (stack_frame_id)
_SECTIONS = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
_NAME_ENTRY = re.compile(r'(\d+)\s+"((?:[^"\\]|\\.)*)"')
_LOCATION_ENTRY = re.compile(r"(\d+)\s+\{file_name_id=(\d+)\s+function_name_id=(\d+)\s+line=(\d+)")
_FRAME_ENTRY = re.compile(r"(\d+)\s+\{file_location_id=(\d+)\s+parent_frame_id=(\d+)")
_OPERAND = re.compile(r"%([\w.\-]+)")
_FORWARDING = ("bitcast", "get-tuple-element", "copy")  # an operand's data comes from the instruction's first operand
_LIBRARY_GEMM = re.compile(r"cublas|gemm", re.IGNORECASE)  # custom-call targets of library GEMMs


def _sanitize(name):
    """XLA's kernel name of a fusion instruction: '.' and '-' (anything else than [A-Za-z0-9_]) become '_'."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


@dataclasses.dataclass
class Instruction:
    name: str
    computation: str
    opcode: str
    shape: str
    kind: str  # fusion kind or custom-call target
    op_name: str
    file: str  # source_file and source_line, if the metadata has them
    line: int
    frame: int  # stack_frame_id (0: none)
    trip_count: int | None  # while loops
    called: tuple
    operands: tuple


@dataclasses.dataclass
class Source:
    line: str  # file:line of the innermost frame (the op_name without a source)
    function: str  # file:function of the innermost frame
    callers: tuple  # the next frames' functions, innermost first
    file: str = ""
    lineno: int = 0


class Hlo:
    """The instructions of an optimized HLO text (Compiled.as_text) with the computations they call, so that every
    instruction has its nest of while loops, and with the module's stack frames, so that it has its source; kernels
    are matched to instructions by key_of() and resolve()."""

    def __init__(self, text=""):
        self.instructions, self.caller = {}, {}
        self._consumers = None
        self.file_names, self.function_names, self.locations, self.frames = {}, {}, {}, {}
        computation = section = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or line.startswith("HloModule"):
                continue
            if stripped in _SECTIONS:
                section = stripped
                continue
            if section is not None and stripped[0].isdigit():
                self._entry(section, stripped)
                continue
            if not line[0].isspace():
                if stripped.endswith("{"):
                    computation, section = _HEADER.match(line).group(1), None
                continue
            match = _INSTRUCTION.match(line)
            if match is None or computation is None:
                continue
            name, rhs = match.groups()
            padded = " " + rhs
            op = _OPCODE.search(padded)
            called = _CALLED.findall(rhs)
            for group in _CALLED_SETS.findall(rhs):
                called += [c.strip().lstrip("%") for c in group.split(",") if c.strip()]
            kind = _KIND.search(rhs) or _TARGET.search(rhs)
            op_name, file, line_no, frame, trips = (_OP_NAME.search(rhs), _SOURCE_FILE.search(rhs),
                                                    _SOURCE_LINE.search(rhs), _FRAME_ID.search(rhs),
                                                    _TRIP_COUNT.search(rhs))
            # operands are printed without shapes (no parentheses inside the list)
            operands = tuple(_OPERAND.findall(padded[op.end():padded.find(")", op.end())])) if op else ()
            self.instructions[name] = Instruction(
                name=name, computation=computation, opcode=op.group(1) if op else "?",
                shape=padded[:op.start()].strip() if op else "", kind=kind.group(1) if kind else "",
                op_name=op_name.group(1) if op_name else "", file=file.group(1) if file else "",
                line=int(line_no.group(1)) if line_no else 0, frame=int(frame.group(1)) if frame else 0,
                trip_count=int(trips.group(1)) if trips else None, called=tuple(called), operands=operands)
            for callee in called:
                self.caller.setdefault(callee, name)
        # kernels carry the sanitised names of fusions and other top-level instructions, never of fused ones
        fused = {c for inst in self.instructions.values() if inst.opcode == "fusion" for c in inst.called}
        self.by_kernel = {_sanitize(name): name for name, inst in self.instructions.items()
                          if inst.computation not in fused}
        self._keys, self._loops, self._sources = {}, {}, {}

    def _entry(self, section, text):
        if section in ("FileNames", "FunctionNames"):
            match = _NAME_ENTRY.match(text)
            if match:
                names = self.file_names if section == "FileNames" else self.function_names
                names[int(match.group(1))] = match.group(2)
        elif section == "FileLocations":
            match = _LOCATION_ENTRY.match(text)
            if match:
                self.locations[int(match.group(1))] = tuple(int(match.group(i)) for i in (2, 3, 4))
        else:
            # the printed parent_frame_id is one more than the parent's printed id (0: no parent; read as the parent
            # itself, frames ended in the function they started in): dense_engine.py:353 < 367 mpo_contract <
            # meas/mps.py:570 fused_measurement.energy < 577 batch, as in the source (job 7363725's HLO)
            match = _FRAME_ENTRY.match(text)
            if match:
                self.frames[int(match.group(1))] = (int(match.group(2)), int(match.group(3)) - 1)

    def loops(self, name):
        """The while instructions around instruction `name`, outermost first."""
        if name not in self._loops:
            path, seen = [], set()
            computation = self.instructions[name].computation if name in self.instructions else None
            while computation in self.caller and computation not in seen:
                seen.add(computation)
                caller = self.instructions[self.caller[computation]]
                if caller.opcode == "while":
                    path.append(caller.name)
                computation = caller.computation
            self._loops[name] = tuple(reversed(path))
        return self._loops[name]

    def outer(self, anchor):
        """The outermost while loop around an anchor that iterates (trip count not 1: run_blocks' scan of one block
        is such a loop around everything), '' outside them all."""
        loops = [name for name in (self.loops(anchor) if anchor else ()) if self.instructions[name].trip_count != 1]
        return loops[0] if loops else ""

    def key_of(self, name, args):
        """(key, anchor) of a kernel: the instruction it ran for and the instruction whose loops it sits in. The
        instruction is the fusion of its (sanitised) name, else its hlo_op (a library call outside a CUDA graph:
        cuSOLVER, cuBLAS). A library kernel or copy inside a CUDA graph has neither (hlo_op names the graph, a
        command buffer made from the thunks, which the HLO text does not have): its key is '<hlo_op>:<kernel class>'
        without an anchor, which resolve() replaces for the GEMMs."""
        op = args.get("hlo_op", "")
        cached = self._keys.get((name, op))
        if cached is not None:
            return cached
        if not self.instructions:
            key = (op if op and not op.startswith("command_buffer") else name, None)
        else:
            key = None
            for candidate in (_sanitize(name), re.sub(r"_\d+$", "", _sanitize(name))):
                if candidate in self.by_kernel:
                    key = (self.by_kernel[candidate],) * 2
                    break
            if key is None and op in self.instructions and self.instructions[op].opcode != "call":
                key = (op, op)
            if key is None:
                key = (f"{op or '?'}:{kernel_class(name)}", None)
        self._keys[(name, op)] = key
        return key

    def _root(self, name):
        """The instruction whose data an operand is: through bitcasts, copies and tuple elements."""
        for _ in range(32):
            inst = self.instructions.get(name)
            if inst is None or inst.opcode not in _FORWARDING or not inst.operands:
                break
            name = inst.operands[0]
        return name

    def _gemm_inputs(self, name):
        """The library GEMM calls whose outputs instruction `name` reads."""
        inst = self.instructions.get(name)
        roots = (self._root(o) for o in (inst.operands if inst is not None else ()))
        return [r for r in roots if r in self.instructions and self.instructions[r].opcode == "custom-call"
                and _LIBRARY_GEMM.search(self.instructions[r].kind)]

    def gemm_readers(self, name):
        """The library GEMM calls that read the output of instruction `name`."""
        if self._consumers is None:
            self._consumers = defaultdict(list)
            for call, inst in self.instructions.items():
                if inst.opcode == "custom-call" and _LIBRARY_GEMM.search(inst.kind):
                    for operand in inst.operands:
                        self._consumers[self._root(operand)].append(call)
        return self._consumers.get(name, ())

    def resolve(self, events, keys):
        """keys (key_of's, one per event of device_events) with the GEMM kernels inside CUDA graphs resolved to their
        cuBLAS calls by data flow, launch by launch (the kernels of one graph launch share hlo_op and correlation_id):
        such a kernel is the call that reads the output of one of the launch's last kernels and has not run in it yet,
        else the call whose output the launch's next resolved kernel reads. A gemm kernel that neither finds keeps
        its '<hlo_op>:gemm' key. (The HLO text is not in schedule order, so the order of the calls cannot tell.)"""
        keys = list(keys)
        start = 0
        while start < len(events):
            launch = (events[start][3].get("hlo_op", ""), events[start][3].get("correlation_id"))
            stop = start + 1
            while stop < len(events) and (events[stop][3].get("hlo_op", ""),
                                          events[stop][3].get("correlation_id")) == launch:
                stop += 1
            if launch[0].startswith("command_buffer"):
                self._resolve_launch(keys, range(start, stop))
            start = stop
        return keys

    def _resolve_launch(self, keys, rows):
        recent, used, pending = [], set(), []
        for row in rows:
            key, anchor = keys[row]
            if anchor is None and key.endswith(":gemm"):
                call = next((c for k in reversed(recent[-6:]) for c in self.gemm_readers(k) if c not in used), None)
                if call is None:
                    pending.append(row)
                    continue
                keys[row] = (call, call)
                key = call
            if key not in self.instructions:
                continue
            for call in self._gemm_inputs(key):  # the pending gemms in the order this kernel reads them
                if pending and call not in used:
                    keys[pending.pop(0)] = (call, call)
                    used.add(call)
            recent.append(key)
            used.add(key)

    def stack(self, inst):
        """(file, function, line) of an instruction's source frames, innermost first: its stack frame chain, else
        source_file and source_line (one frame, its function from the file)."""
        frames, frame, seen = [], inst.frame, set()
        while frame in self.frames and frame not in seen:
            seen.add(frame)
            location, frame = self.frames[frame]
            if location in self.locations:
                file_id, function_id, line = self.locations[location]
                frames.append((self.file_names.get(file_id, "?"), self.function_names.get(function_id, "?"), line))
        if not frames and inst.file:
            frames.append((inst.file, function_at(inst.file, inst.line), inst.line))
        return frames

    def source(self, key, anchor):
        """The Source of an instruction (of the anchor for a key that is none); without frames the op_name, the
        jit/vmap/while path JAX records, stands for both the line and the function."""
        if (key, anchor) not in self._sources:
            inst = self.instructions.get(key) or self.instructions.get(anchor)
            frames = self.stack(inst) if inst is not None else []
            if frames:
                file, function, line = frames[0]
                source = Source(f"{_short(file)}:{line}", f"{_short(file)}:{_function(function)}",
                                tuple(_function(f) for _, f, _ in frames[1:]), file, line)
            elif inst is not None and inst.op_name:
                path = re.sub(r"^jit\(run_blocks\)/", "", inst.op_name)
                source = Source(path, path.rpartition("/")[0] or path, ())
            else:
                source = Source(key, key, ())
            self._sources[(key, anchor)] = source
        return self._sources[(key, anchor)]

    def describe(self, key):
        """opcode[fusion kind | custom-call target] and shape of an instruction, for the tables."""
        inst = self.instructions.get(key)
        if inst is None:
            return "", ""
        return inst.opcode + (f"[{inst.kind}]" if inst.kind else ""), inst.shape

    def loop_label(self, name):
        """A while loop as 'x<trip count> file:line function (instruction)'."""
        if not name:
            return "(outside the loops)"
        inst, source = self.instructions[name], self.source(name, name)
        return f"x{inst.trip_count or '?'} {source.line} {source.function.rpartition(':')[2]} ({name})"

    def iterations(self, path):
        """The iterations of the innermost loop of `path` per block: the product of the trip counts (None if any is
        unknown)."""
        product = 1
        for name in path:
            trips = self.instructions[name].trip_count
            if trips is None:
                return None
            product *= trips
        return product


def _short(path):
    """A source path from the trot package down (gmps/dense_engine.py), else its file name."""
    path = str(path)
    i = path.rfind("/trot/")
    return path[i + 6:] if i >= 0 else Path(path).name


def _function(name):
    """A function name without Python's '<locals>' parts."""
    return name.replace(".<locals>.", ".")


@lru_cache(maxsize=None)
def _functions(path):
    """(first line, last line, qualified name) of every def in a source file (a lambda counts as its def)."""
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, ValueError):
        return ()
    spans = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    spans.append((child.lineno, child.end_lineno, name))
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return tuple(spans)


def function_at(path, line):
    """The innermost def around a line of a source file ('<module>' outside every def). The file must be as it was at
    the compile (the run's sources unchanged since)."""
    best = None
    for first, last, name in _functions(path):
        if first <= line <= last and (best is None or first >= best[0]):
            best = (first, name)
    return best[1] if best else "<module>"


# ---------------------------------------------------------------------------------------------------------------------
# the traces
# ---------------------------------------------------------------------------------------------------------------------

def device_events(trace_file):
    """(start us, duration us, name, args) of the GPU's kernels and copies in a trace.json.gz, ordered by start: the
    events of its stream lines (all of its lines if none is a stream), as bench_mps_cpmc.summarize_trace."""
    with gzip.open(trace_file, "rt") as stream:
        data = json.load(stream)
    events = data["traceEvents"] if isinstance(data, dict) else data
    process = {e["pid"]: e.get("args", {}).get("name", "") for e in events
               if e.get("ph") == "M" and e.get("name") == "process_name"}
    thread = {(e["pid"], e.get("tid")): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "thread_name"}
    gpus = {pid for pid, name in process.items() if "GPU" in name.upper() and "HOST" not in name.upper()}
    device = [e for e in events if e.get("ph") == "X" and e.get("pid") in gpus and "dur" in e]
    streams = [e for e in device if "stream" in thread.get((e["pid"], e.get("tid")), "").lower()] or device
    return sorted(((float(e["ts"]), float(e["dur"]), e.get("name", "?"), e.get("args", {})) for e in streams),
                  key=lambda e: e[0])


def _add(target, values, scale=1.0):
    for i, value in enumerate(values):
        target[i] += value * scale


def analyze(path, top=25, block_seconds=None):
    """The report of a profile directory (module docstring): printed, and written to report.txt and summary.json.
    block_seconds: the block's length, if windows.json has none (the run ended inside the block); else the end of the
    last window, so the per-block numbers then cover the traced part of the block only."""
    path = Path(path)
    record = json.loads((path / WINDOWS_FILE).read_text())
    hlo_file = path / HLO_FILE
    hlo = Hlo(hlo_file.read_text() if hlo_file.exists() else "")
    windows = sorted(record["windows"], key=lambda w: w["start"])
    if not windows:
        raise SystemExit(f"{path / WINDOWS_FILE} lists no windows")
    block = float(block_seconds or record.get("block_seconds") or windows[-1]["stop"])
    out = []
    say = out.append
    when = f"at {record['at']} s" if record.get("at") else f"every {record['every']:g} s"
    say(f"== block {record['block']} ({record.get('phase')}) of job {record.get('job')}: {block:.1f} s"
        + ("" if record.get("block_seconds") else " (the run ended inside the block: per block = up to this time)")
        + f", {len(windows)} windows of {record['window']:g} s {when}"
        + (f"; tracing stopped early: {record['error']}" if record.get("error") else ""))
    if record.get("memory"):
        say("XLA memory analysis of the block (GB): " + ", ".join(
            f"{name.replace('_size_in_bytes', '')} {value / 2 ** 30:.2f}" for name, value in record["memory"].items()))
    if hlo.instructions:
        say(f"HLO: {len(hlo.instructions)} instructions, {len(hlo.frames)} stack frames, from {hlo_file}")
    else:
        say(f"no HLO ({hlo_file} missing): kernels by hlo_op or name only, no loops or sources")

    # per (key, anchor): kernel seconds, idle seconds before its kernels, kernels; per block (scaled windows)
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    classes = defaultdict(float)  # (outermost loop, kernel class) -> kernel seconds per block
    rows = []
    say("\nwindows (kernels and GPU busy within the window; stands: the seconds of the block it stands for; loop: "
        "where most of its kernel time went):")
    say(f"  {'window':<6} {'start s':>8} {'span s':>7} {'stands':>7} {'kernels':>9} {'kernel s':>9} {'busy':>6}  loop")
    for i, window in enumerate(windows):
        files = sorted((path / window["dir"]).rglob("*.trace.json.gz"))
        if not files:
            say(f"  {window['dir']:<6} no trace")
            continue
        events = device_events(files[-1])
        # the trace's own extent if longer: the device tracer can record on while the session stops (7358122's first
        # window, exporting its 353 MB HLO proto, traced ~24 s of a nominal 12)
        extent = (max(ts + dur for ts, dur, _, _ in events) - events[0][0]) / 1e6 if events else 0.0
        span = max(window["stop"] - window["start"], extent)
        stands = max((windows[i + 1]["start"] if i + 1 < len(windows) else block) - window["start"], 0.0)
        scale = stands / span if span > 0 else 0.0
        stats = defaultdict(lambda: [0.0, 0.0, 0.0])
        names = defaultdict(float)
        busy, end = 0.0, None
        keys = hlo.resolve(events, [hlo.key_of(name, args) for _, _, name, args in events])
        for (ts, dur, name, args), key in zip(events, keys):
            idle = 0.0 if end is None else max(0.0, ts - end)
            busy += dur if end is None else max(0.0, ts + dur - max(ts, end))
            end = ts + dur if end is None else max(end, ts + dur)
            _add(stats[key], (dur / 1e6, idle / 1e6, 1.0))
            names[(key[1], name)] += dur / 1e6
        by_loop = defaultdict(float)
        for key, values in stats.items():
            by_loop[hlo.outer(key[1])] += values[0]
            _add(totals[key], values, scale)
        for (anchor, name), seconds in names.items():
            classes[(hlo.outer(anchor), kernel_class(name))] += seconds * scale
        main = max(by_loop, key=by_loop.get) if by_loop else ""
        kernel = sum(v[0] for v in stats.values())
        rows.append(dict(window=window["dir"], start=window["start"], span=span, stands=stands, kernels=len(events),
                         kernel_s=kernel, busy=busy / 1e6 / span if span > 0 else None, loop=main))
        say(f"  {window['dir']:<6} {window['start']:8.1f} {span:7.2f} {stands:7.1f} {len(events):9d} {kernel:9.3f} "
            f"{100 * busy / 1e6 / span if span > 0 else 0:5.1f}%  {hlo.loop_label(main) if hlo.instructions else ''}")

    kernel_total = sum(v[0] for v in totals.values())
    idle_total = sum(v[1] for v in totals.values())
    say(f"\nper block, from the windows: kernels {kernel_total:.1f} s + GPU idle between them {idle_total:.1f} s = "
        f"{kernel_total + idle_total:.1f} s of the block's {block:.1f} s; {sum(v[2] for v in totals.values()):.3g} "
        "kernels")

    def share(seconds):
        return 100 * seconds / max(kernel_total + idle_total, 1e-12)

    tree = defaultdict(lambda: [0.0, 0.0, 0.0])
    for key, values in totals.items():
        loops = hlo.loops(key[1]) if key[1] else ()
        for depth in range(len(loops) + 1):
            _add(tree[loops[:depth]], values)
    tree_rows = []
    if hlo.instructions:
        say("\nby loop nest (inclusive; self: not in a deeper loop; ms/iter: (kernel + idle) / iterations per block):")
        say(f"  {'kernel s':>9} {'idle s':>8} {'share':>6} {'self s':>8} {'kernels':>9} {'iters':>8} "
            f"{'ms/iter':>9}  loop")

        def show(prefix):
            children = sorted((p for p in tree if len(p) == len(prefix) + 1 and p[:len(prefix)] == prefix),
                              key=lambda p: -sum(tree[p][:2]))
            kernel, idle, count = tree[prefix]
            inner = sum(sum(tree[p][:2]) for p in children)
            iterations = hlo.iterations(prefix)
            per = f"{1e3 * (kernel + idle) / iterations:9.2f}" if iterations else f"{'':>9}"
            label = hlo.loop_label(prefix[-1]) if prefix else "the block"
            say(f"  {kernel:9.1f} {idle:8.1f} {share(kernel + idle):5.1f}% {kernel + idle - inner:8.1f} {count:9.3g} "
                f"{iterations or '':>8} {per}  {'  ' * len(prefix)}{label}")
            tree_rows.append(dict(loops=list(prefix), label=label, kernel_s=kernel, idle_s=idle, kernels=count,
                                  iterations=iterations))
            for child in children:
                if share(sum(tree[child][:2])) >= 0.5:
                    show(child)

        show(())

    def table(title, groups, describe):
        say(f"\n{title}")
        say(f"  {'kernel s':>9} {'idle s':>8} {'share':>6} {'kernels':>9} {'us/kernel':>10}  what")
        ranked = sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]
        for name, (kernel, idle, count) in ranked:
            say(f"  {kernel:9.2f} {idle:8.2f} {share(kernel + idle):5.1f}% {count:9.3g} "
                f"{1e6 * kernel / max(count, 1e-12):10.1f}  {describe(name)}")
        return [dict(what=str(name), kernel_s=v[0], idle_s=v[1], kernels=v[2]) for name, v in ranked]

    by_outer = defaultdict(dict)
    for key, values in totals.items():
        by_outer[hlo.outer(key[1])][key] = values
    functions = {}
    for loop in sorted(by_outer, key=lambda name: -sum(v[0] + v[1] for v in by_outer[name].values())):
        groups = defaultdict(lambda: [0.0, 0.0, 0.0])
        for key, values in by_outer[loop].items():
            source = hlo.source(*key)
            _add(groups[source.function + (f"  < {source.callers[0]}" if source.callers else "")], values)
        if share(sum(v[0] + v[1] for v in groups.values())) < 0.5:
            continue
        label = hlo.loop_label(loop) if hlo.instructions else "all kernels"
        functions[label] = table(f"top source functions (< their caller) in {label}:", groups, str)

    lines, kinds, callers = defaultdict(lambda: [0.0, 0.0, 0.0]), defaultdict(set), {}
    for key, values in totals.items():
        source = hlo.source(*key)
        _add(lines[source.line], values)
        kinds[source.line].add(hlo.describe(key[0])[0] or key[0].rsplit(":", 1)[-1])
        callers.setdefault(source.line, source)

    def line_text(where):
        source = callers[where]
        text = linecache.getline(source.file, source.lineno).strip() if source.file else ""
        chain = " < ".join((source.function.rpartition(":")[2],) + source.callers[:3])
        return f"{where}  [{', '.join(sorted(kinds[where]))[:40]}]  {chain}\n{'':>48}{text[:100]}"

    top_lines = table("top source lines (the HLO ops there, the innermost frames < callers, the line):", lines,
                      line_text)
    instructions = defaultdict(lambda: [0.0, 0.0, 0.0])
    for key, values in totals.items():
        _add(instructions[key[0]], values)
    anchors = {key[0]: key[1] for key in totals}
    top_instructions = table(
        "top HLO instructions (opcode[fusion kind | custom-call target], shape, source line):", instructions,
        lambda name: "  ".join(x for x in (name, *hlo.describe(name)) if x)[:150] + "  "
        + hlo.source(name, anchors[name]).line)

    say("\nkernel classes per outermost loop (kernel s per block):")
    outers = {loop for loop, _ in classes}
    for loop in sorted(outers, key=lambda name: -sum(v for (o, _), v in classes.items() if o == name)):
        parts = sorted(((c, v) for (o, c), v in classes.items() if o == loop), key=lambda cv: -cv[1])
        say(f"  {hlo.loop_label(loop) if hlo.instructions else 'all kernels'}: "
            + ", ".join(f"{c} {v:.1f}" for c, v in parts))

    text = "\n".join(out)
    print(text, flush=True)
    (path / "report.txt").write_text(text + "\n")
    summary = dict(record=record, windows=rows, kernel_s=kernel_total, idle_s=idle_total, loops=tree_rows,
                   functions=functions, lines=top_lines, instructions=top_instructions,
                   classes=[dict(loop=loop, kind=kind, kernel_s=v) for (loop, kind), v in classes.items()])
    (path / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


# ---------------------------------------------------------------------------------------------------------------------
# a run that ended inside the traced block (no windows.json, no HLO dump)
# ---------------------------------------------------------------------------------------------------------------------

def synthesize_record(path, every, window, block):
    """windows.json of a run that ended before writing one: windows w<k> at k * every s for `window` s each (their
    nominal slots), block `block` of an unknown job."""
    dirs = sorted(p.name for p in Path(path).glob("w[0-9][0-9][0-9]") if any(p.rglob("*.trace.json.gz")))
    windows = [dict(k=int(d[1:]), dir=d, start=int(d[1:]) * every, stop=int(d[1:]) * every + window) for d in dirs]
    record = dict(block=block, phase=None, block_seconds=None, every=every, window=window, at=None, windows=windows,
                  error="windows.json reconstructed from the window folders (nominal times)", job=None)
    (Path(path) / WINDOWS_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return record


def _varint(buf, i):
    shift = result = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7


def _fields(buf, start=0, end=None):
    """(field number, value) of the protobuf message buf[start:end]: an int for varint and fixed fields, a (start,
    end) span for length-delimited ones (wire type 2)."""
    end = len(buf) if end is None else end
    i = start
    while i < end:
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = int.from_bytes(buf[i:i + 8], "little"), i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = (i, i + size), i + size
        elif wire == 5:
            value, i = int.from_bytes(buf[i:i + 4], "little"), i + 4
        else:
            raise ValueError(f"protobuf wire type {wire} at byte {i}")
        yield number, value


def hlo_from_xplane(xplane, prefix="jit_run_blocks"):
    """The optimized HLO text of the largest module named prefix* in an xplane.pb (a window traced with
    enable_hlo_proto: BlockProfiler's first one): the 'Hlo Proto' stat (an HloProto, its hlo_module field 1) of the
    event metadata of the /host:metadata plane, printed by jaxlib's HloModule. Needs jaxlib (CPU is enough).

    tsl's xplane.proto: XSpace.planes 1; XPlane.name 2, .event_metadata 4 and .stat_metadata 5 (maps: key 1, value
    2); XEventMetadata.name 2, .stats 5; XStatMetadata.id 1, .name 2; XStat.metadata_id 1, .bytes_value 6."""
    buf = Path(xplane).read_bytes()
    found = []
    for number, plane in _fields(buf):
        if number != 1:
            continue
        fields = list(_fields(buf, *plane))
        name = next((buf[v[0]:v[1]].decode() for n, v in fields if n == 2), "")
        if name != "/host:metadata":
            continue
        hlo_stat = None
        for n, entry in fields:
            if n == 5:
                value = next((v for k, v in _fields(buf, *entry) if k == 2), None)
                stat = dict(_fields(buf, *value)) if value else {}
                if 2 in stat and buf[stat[2][0]:stat[2][1]] == b"Hlo Proto":
                    hlo_stat = stat.get(1, 0)
        for n, entry in fields:
            if n != 4:
                continue
            value = next((v for k, v in _fields(buf, *entry) if k == 2), None)
            if value is None:
                continue
            event_name, protos = "", []
            for k, v in _fields(buf, *value):
                if k == 2:
                    event_name = buf[v[0]:v[1]].decode(errors="replace")
                elif k == 5:
                    stat = dict(_fields(buf, *v))
                    if stat.get(1, 0) == hlo_stat and 6 in stat:
                        protos.append(stat[6])
            if event_name.startswith(prefix):
                found += [(event_name, span) for span in protos]
    if not found:
        raise SystemExit(f"no HLO proto of a {prefix}* module in {xplane} (traced without enable_hlo_proto?)")
    event_name, (start, end) = max(found, key=lambda item: item[1][1] - item[1][0])
    module = next(v for k, v in _fields(buf, start, end) if k == 1)  # HloProto.hlo_module
    from jaxlib import _hlo, _jax

    for lib in (_hlo, _jax):
        cls = getattr(lib, "HloModule", None)
        if cls is not None and hasattr(cls, "from_serialized_hlo_module_proto"):
            print(f"HLO of {event_name} ({(end - start) / 2 ** 20:.0f} MB proto) from {xplane}", flush=True)
            return cls.from_serialized_hlo_module_proto(buf[module[0]:module[1]]).to_string()
    raise SystemExit("this jaxlib has no HloModule.from_serialized_hlo_module_proto")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="the profile directory (<out>/profile of a --profile run)")
    parser.add_argument("--top", type=int, default=25, help="rows of the top tables")
    parser.add_argument("--block-seconds", type=float, default=None,
                        help="the block's length when windows.json has none (the run ended inside the block)")
    parser.add_argument("--hlo-from-xplane", action="store_true",
                        help=f"without {HLO_FILE}: take the HLO from the window xplane.pb that holds it (needs jaxlib)")
    parser.add_argument("--every", type=float, default=150.0,
                        help="without windows.json: the windows' spacing of the run (its --profile-every)")
    parser.add_argument("--window", type=float, default=12.0,
                        help="without windows.json: the windows' length of the run (its --profile-window)")
    parser.add_argument("--block", type=int, default=2, help="without windows.json: the traced block's number")
    args = parser.parse_args(argv)
    if not (args.path / WINDOWS_FILE).exists():
        synthesize_record(args.path, args.every, args.window, args.block)
        print(f"wrote {args.path / WINDOWS_FILE} from the window folders (nominal times)", flush=True)
    if args.hlo_from_xplane and not (args.path / HLO_FILE).exists():
        for xplane in sorted(args.path.glob("w*/plugins/profile/*/*.xplane.pb"), key=lambda p: -p.stat().st_size):
            try:
                (args.path / HLO_FILE).write_text(hlo_from_xplane(xplane))
                break
            except SystemExit as error:
                print(error, flush=True)
    analyze(args.path, top=args.top, block_seconds=args.block_seconds)


if __name__ == "__main__":
    main()
