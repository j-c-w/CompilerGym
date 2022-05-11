import logging

from typing import Optional, Tuple
from pathlib import Path
from compiler_gym.service import CompilationSession
from compiler_gym.util.commands import run_command
from compiler_gym.service.proto import (
ActionSpace,
Benchmark,
DoubleRange,
Event,
Int64Box,
Int64Range,
Int64Tensor,
NamedDiscreteSpace,
ObservationSpace,
Space,
StringSpace
)
import compiler_gym.third_party.llvm as llvm
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
from DFG import DFG

from compiler_gym.service.proto.compiler_gym_service_pb2 import Int64SequenceSpace
#from compiler_gym.service.runtime import create_and_run_compiler_gym_service

Operations = [
    # TODO --- should we support more operations as heterogeneous?
    # IMO most of the other things that are scheduled are
    # pretty vacuous, although we could explore supporting those.
    "basic_operation", # Some operations, like phi, etc. can be treated
    # as basic operations that any node could preform.
    "add",
    "mul",
    "sub",
    "div",
    "and",
    "or",
    "xor",
    "fmul",
    "fsub",
    "fadd",
    "fdiv",
    "rsh",
    "lsh",
    "load",
    "store",
    "noop"
]

def operation_latency(op):
    # TODO --- model latency --- or at least expost this
    # to a configuration.
    return 1



class CGRA(object):
    # Assume a rectangular matrix with neighbour
    # connections.  We can handle the rest later.
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim

        grid = []
        # TODO -- load in the heterogeneous CGRA file.

    def __str__(self):
        return "CGRA: (" + (str(self.x_dim)) + ", " + (str(self.y_dim)) + ")"

    def is_supported(node_index, op):
        # TODO -- support heterogeneity
        return True

    def get_coords(self, i):
        x = i // self.y_dim
        y = i % self.y_dim

        return x, y

    def cells_as_list(self):
        l = []
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                l.append(str(i * self.y_dim + j))

        return l

    def distance(self, index_1, index_2):
        # Does this need to be SPP eventually?
        x_1, y_1 = self.get_coords(index_1)
        x_2, y_2 = self.get_coords(index_2)

        return abs(x_1 - x_2) + abs(y_1 - y_2)

    def delay(self, index_1, index_2):
        # Assume one hop per cell right now.
        return self.distance(index_1, index_2)

class Schedule(object):
    def __init__(self, cgra):
        self.cgra = cgra

        self.operations = []
        for x in range(cgra.x_dim):
            arr = []
            for y in range(cgra.y_dim):
                arr.append(None)
            self.operations.append(arr)

    def set_operation(self, index, node):
        x, y = self.cgra.get_coords(index)

        if self.operations[x][y] is None:
            self.operations[x][y] = node
            return True
        else:
            # Not set
            return False

    def get_location(self, node):
        # TODO -- make a hash table or something more efficient if required.
        for x in range(self.cgra.x_dim):
            for y in range(self.cgra.y_dim):
                if self.operations[x][y].name == node.name:
                    return x, y
        return None, None

    def compute_communication_distance(self, n1, n2):
        # TODO -- thre is a lot that we should account
        # for.  Like not overloading particular routing
        # resources.
        n1_x, n1_y = self.get_location(n1)
        n2_x, n2_y = self.get_location(n2)

        # TODO --- This needs to be adjusted to account for non-grid
        # CGRAs.
        return abs(n2_x - n1_x) + abs(n2_y - n1_y)

    def get_II(self, dfg):
        # Compute the II of the current schedule.

        # What cycle does this node get executed on?
        cycles_start = {}
        # What cycle does the result of this node become
        # available on?
        cycles_end = {}

        # Keep track of when resources can be re-used.
        freed = {} # When we're done
        used = {} # When we start

        # Step 1 is to iterate over all the nodes
        # in a BFS manner.
        for node in dfg.bfs():
            # For each node, compute the latency,
            # and the delay to get the arguments to
            # reach it.
            preds = dfg.getPreds(node)

            earliest_time = 0
            for pred in preds:
                loc = self.get_location(node)
                pred_cycle = cycles_end[pred.name]

                # Compute the time to this node:
                # TODO -- should we also account for not being
                # able to use the NOC for multiple things at once?
                distance = self.compute_communication_distance(pred, node)

                # Compute when this predecessor reaches this node:
                arrival_time = distance + pred_cycle
                earliest_time = max(earliest_time, arrival_time)

            # Make sure that the PE is actually free.
            if loc in freed:
                earliest_time = max(earliest_time, max(freed[loc]))

            # This node should run at the earliest time available.
            cycles_start[node] = earliest_time
            cycles_end[node] = earliest_time + operation_latency(node.operation)

            # Keep track of when this PE can be used again.
            if loc in freed:
                freed[loc].append(cycles_end[node])
                used[loc].append(cycles_start[node])
            else:
                freed[loc] = [cycles_end[node]]
                used[loc] = [cycles_start[node]]


        # Now that we've done that, we need to go through all the nodes and
        # work out the II.
        # When was this computation slot last used? (i.e. when could
        # we overlap the next iteration?)
        last_freed = {}
        first_used = {}
        min_II = 0
        for loc in freed:
            # Now, we could achieve better performance
            # by overlapping these in a more fine-grained
            # manner --- but that seems like a lot of effort
            # for probably not much gain?
            # there ar probably loops where the gain
            # is not-so-marginal.
            last_freed[loc] = max(freed[loc])
            first_used[loc] = min(used[loc])

            difference = last_freed[loc] = first_used[loc]
            min_II = max(min_II, difference)

        # TODO --- we should probably return some kind of object
        # that would enable final compilation also.
        return min_II

compilation_session_cgra = CGRA(5, 5)

action_space = [ActionSpace(name="Schedule",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=compilation_session_cgra.cells_as_list()
                )
                # int64_box=Int64Box(
                #     low=Int64Tensor(shape=[2], value=[0, 0]),
                #     high=Int64Tensor(shape=[2], value=[compilation_session_cgra.x_dim, compilation_session_cgra.y_dim])
                # )
                )
            )
        ]

MAX_WINDOW_SIZE = 100

observation_space = [
            # ObservationSpace(
            #     name="dfg",
            #     space=Space(
            #         string_value=StringSpace(length_range=(Int64Range(min=0)))
            #     ),
            #     deterministic=True,
            #     platform_dependent=False,
            #     default_observation=Event(string_value="")
            # ),
            ObservationSpace(name="ir",
                space=Space(
                    # TODO -- I think this should be a window of operations
                    # around the current one.
                    int64_sequence=Int64SequenceSpace(length_range=Int64Range(min=0, max=MAX_WINDOW_SIZE), scalar_range=Int64Range(min=0, max=len(Operations)))
                )
            ),
            ObservationSpace(name="CurrentInstruction",
                space=Space(
                    int64_value=Int64Range(min=0, max=len(Operations)),
                # TODO -- also need to figure out how to make this
                # a graph?
                ),
                deterministic=True,
                platform_dependent=False
            ),
            ObservationSpace(name="CurrentInstructionIndex",
                space=Space(
                    int64_value=Int64Range(min=0, max=MAX_WINDOW_SIZE)
                ))
            # ObservationSpace(
            #     name="Schedule",
            #     space=Space(
            #         int64_box=Int64Box(
            #             low=Int64Tensor(shape=[2], value=[0, 0]),
            #             high=Int64Tensor(shape=[2], value=[cgra.x_dim, cgra.y_dim])
            #         )
            #     )
            # )
        ]

class CGRACompilationSession(CompilationSession):
    def __init__(self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark):
        super().__init__(working_directory, action_space, benchmark)
        self.schedule = Schedule(self.cgra)
        logging.info("Starting a compilation session for CGRA" + str(self.cgra))

        # Load the DFG (from a test_dfg.json file):
        self.dfg = DFG(working_directory, benchmark, from_json='test/test_dfg.json')

        self.current_operation_index = 0
        # TODO -- load this properly.
        self.dfg_to_ops_list()

    def dfg_to_ops_list(self):
        # Embed the DFG into an operations list that we go through ---
        # it contains two things: the name of the node, and the index
        # that corresponds to within the Operations list.
        self.ops = []
        for op in self.dfg.nodes:
            # Do we need to do a topo-sort here?
            if op.operation in Operations:
                ind = Operations.index(op.operation)
            else:
                print("Did not find operation " + op.operation + " in the set of Operations")
                ind = 0

            self.ops.append(ind)

    cgra = compilation_session_cgra
    schedule = Schedule(cgra)

    action_spaces = action_space

    observation_spaces = observation_space
    # TODO --- a new observation space corresponding to previous actions

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        # print("Action has fields {}".format(str(action.__dict__)))
        print("Action is {}".format(str(action)))

        response = action.int64_value

        # Update the CGRA to schedule the current operation at this space:
        # Take 0 to correspond to a no-op.
        had_effect = False
        if response > 0:
            # Schedule is set up to take the operation at the response index
            # index - 1.
            op_set = self.schedule.set_operation(response - 1, self.ops[self.current_operation_index])

            if op_set:
                had_effect = True
                self.current_operation_index += 1
        # TODO --- Update the state.

        done = False
        if self.current_operation_index >= len(self.ops):
            done = True
        return done, None, had_effect

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing an observation over the space")

        if observation_space.name == "ir":
            # TODO --- This should be a DFG?
            return Event(int64_tensor=Int64Tensor(shape=[len(self.ops)], value=self.ops))
        elif observation_space.name == "Schedule":
            # TODO -- needs to return the schedule for the past
            # CGRA history also?
            box_value = self.schedule.current_iteration
            return Event(int64_box_value=box_value)
        elif observation_space.name == "CurrentInstruction":
            # Return the properties of the current instruction.
            if self.current_operation_index >= len(self.ops):
                # I don't get why this is ahpepning --- just make
                # sure the agent doesn't yse this.  I think it
                # might happen on the last iteration.
                return Event(int64_value=-1)
            else:
                return Event(int64_value=self.ops[self.current_operation_index])
        elif observation_space.name == "CurrentInstructionIndex":
            # Return a way to localize the instruction within the graph.
            return Event(int64_value=self.current_operation_index)