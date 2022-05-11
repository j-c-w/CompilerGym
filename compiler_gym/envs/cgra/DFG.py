import json
from pathlib import Path
from compiler_gym.service.proto import (
Benchmark
)
from typing import Optional
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
import compiler_gym.third_party.llvm as llvm

class Edge(object):
    def __init__(self, type):
        self.type = type

class Node(object):
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation

class DFG(object):
    def __init__(self, working_directory: Path, benchmark: Benchmark, from_json: Optional[Path] = None):
        # Copied from here: https://github.com/facebookresearch/CompilerGym/blob/development/examples/loop_optimizations_service/service_py/loops_opt_service.py
        # self.inst2vec = _INST2VEC_ENCODER

        self.clang = str(llvm.clang_path())
        self.llc = str(llvm.llc_path())
        self.llvm_diff = str(llvm.llvm_diff_path())
        self.opt = str(llvm.opt_path())

        self.working_directory = working_directory
        self.llvm_path = str(self.working_directory / "benchmark.ll")
        self.src_path = str(self.working_directory / "benchmark.c")
        self.dfg_path = str(self.working_directory / "benchmark.dfg.json")

        if from_json is None:
            # Only re-create the JSON file if we aren't providing an existing one.
            # The existing ones are mostly a debugging functionality.
            with open(self.working_directory / "benchmark.c", "wb") as f:
                f.write(benchmark.program.contents)

            # We use CGRA-Mapper to produce a DFG in JSON.
            run_command(
                ["cgra-mapper", self.src_path, self.dfg_path]
            )

            # Now, load in the DFG.
            self.load_dfg_from_json(self.dfg_path)
        else:
            self.load_dfg_from_json(from_json)

    def load_dfg_from_json(self, path):
        import json
        with open(path, 'r') as p:
            f = json.load(p)

            self.nodes = []
            self.edges = []
            self.adj = {}

            # build the nodes first.
            for node in f['nodes']:
                self.nodes.append(Node(node['name'], node['operation']))

            for edge in f['edges']:
                self.edges.append(Edge(edge['type']))

            # Build the adj matrix:
            for edge in f['edges']:
                self.adj[edge['from']] = edge['to']