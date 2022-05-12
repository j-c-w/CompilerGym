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

    def __str__(self):
        return "Node with name " + self.name + " and op " + self.operation

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

            self.nodes = {}
            self.edges = []
            self.adj = {}
            self.entry_points = f['entry_points']

            # build the nodes first.
            for node in f['nodes']:
                self.nodes[node['name']] = (Node(node['name'], node['operation']))
                self.adj[node['name']] = []

            for edge in f['edges']:
                self.edges.append(Edge(edge['type']))

            # Build the adj matrix:
            for edge in f['edges']:
                fnode = edge['from']
                tnode = edge['to']

                self.adj[fnode].append(tnode)
    
    # Bit slow this one --- the adjacency matrix is backwards for it :'(
    def get_preds(self, node):
        preds = []
        for n in self.adj:
            if node.name in self.adj[n]:
                preds.append(self.nodes[n])

        return preds

    # TODO -- fix this, because for a graph with multiple entry nodes,
    # this doesn't actually give the right answer :)
    # (should do in most cases)
    def bfs(self):
        to_explore = self.entry_points[:]
        seen = set()

        while len(to_explore) > 0:
            head = to_explore[0]
            to_explore = to_explore[1:]
            if head in seen:
                continue
            seen.add(head)
            yield self.nodes[head]

            # Get the following nodes.
            following_nodes = self.adj[head]
            to_explore += following_nodes