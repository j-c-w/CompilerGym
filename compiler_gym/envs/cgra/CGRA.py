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
import DFG

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

# This is just a wrapper around an actual object that
# is a schedule --- see the Schedule object for something
# that can be interacted with.
class InternalSchedule(object):
    def __init__(self, cgra):
        self.cgra = cgra
        self.operations = self.initialize_schedule()

    def locations(self):
        for x in range(self.cgra.x_dim):
            for y in range(self.cgra.y_dim):
                yield x, y

    def add_timestep(self):
        ops = []
        for x in range(self.cgra.x_dim):
            arr = []
            for y in range(self.cgra.y_dim):
                arr.append(None)
            ops.append(arr)
        return ops

    def initialize_schedule(self):
        ops = []

        ops.append(self.add_timestep())
        return ops

    # Return true if the CGRA slots are free between
    # start_tiem and end_time in location (x, y)
    def slots_are_free(self, x, y, start_time, end_time):
        for t in range(start_time, end_time):
            # Add more timesteps to the schedule as required.
            while t >= len(self.operations):
                self.operations.append(self.add_timestep())

            if self.operations[t][x][y] is not None:
                return False
        return True

    # Return the earliest time after earliest time that we can
    # fit an op of length 'length' in location x, y
    def get_free_time(self, earliest_time, length, x, y):
        while not self.slots_are_free(x, y, earliest_time, earliest_time + length):
            earliest_time += 1
        return earliest_time

    def set_operation(self, time, x_loc, y_loc, node, latency):
        while time + latency >= len(self.operations):
            self.operations.append(self.add_timestep())

        if self.slots_are_free(x_loc, y_loc, time, time + latency):
            for t in range(time, time + latency):
                self.operations[t][x_loc][y_loc] = node
            return True
        else:
            # Not set
            return False

    def get_location(self, node: DFG.Node):
        # TODO -- make a hash table or something more efficient if required.
        for t in range(len(self.operations)):
            for x in range(self.cgra.x_dim):
                for y in range(self.cgra.y_dim):
                    if self.operations[t][x][y] is None:
                        continue
                    if self.operations[t][x][y].name == node.name:
                        return t, x, y
        return None, None, None

    def free_times(self, x, y):
        occupied_before = False
        for t in range(len(self.operations)):
            if self.operations[t][x][y] is not None:
                occupied_before = True
            else:
                if occupied_before:
                    # This was occupired at the last timestep t,
                    # so it's become freed at this point.
                    occupied_before = False
                    yield t

    def has_use(self, x, y):
        for t in range(len(self.operations)):
            if self.operations[t][x][y] is not None:
                return True
        return False

    def alloc_times(self, x, y):
        free_before = True
        for t in range(len(self.operations)):
            if self.operations[t][x][y] is not None:
                # Was previously free.
                if free_before:
                    # Now was not free before.
                    free_before = False
                    yield t
            else:
                free_before = True

class Schedule(object):
    def __init__(self, cgra):
        self.cgra = cgra

        self.operations = InternalSchedule(cgra)

    def set_operation(self, time, index, node, latency):
        x, y = self.cgra.get_coords(index)
        return self.operations.set_operation(time, x, y, node, latency)

    def get_location(self, node):
        return self.operations.get_location(node)

    def compute_communication_distance(self, n1, n2):
        # TODO -- thre is a lot that we should account
        # for.  Like not overloading particular routing
        # resources.
        print("Looking at nodes", n1, n2)
        n1_t, n1_x, n1_y = self.get_location(n1)
        n2_t, n2_x, n2_y = self.get_location(n2)

        # TODO --- This needs to be adjusted to account for non-grid
        # CGRAs.
        return abs(n2_x - n1_x) + abs(n2_y - n1_y)

    def get_II(self, dfg):
        # Compute the II of the current schedule.

        # We don't require the placement part to be actually correct ---
        # do the actual schedule what we generate can differ
        # from the schedule we have internally.
        actual_schedule = InternalSchedule(self.cgra)

        # What cycle does this node get executed on?
        cycles_start = {}
        # What cycle does the result of this node become
        # available on?
        cycles_end = {}

        # Keep track of when resources can be re-used.
        freed = {} # When we're done
        used = {} # When we start

        # We keep track of whether scheduling is finished
        # elsewhere --- this is just a sanity-check.
        finished = True

        # Step 1 is to iterate over all the nodes
        # in a BFS manner.
        for node in dfg.bfs():
            # For each node, compute the latency,
            # and the delay to get the arguments to
            # reach it.
            preds = dfg.get_preds(node)

            # Get the time that this operation has
            # been scheduled for.
            scheduled_time, loc_x, loc_y = self.get_location(node)
            earliest_time = scheduled_time

            if scheduled_time is None:
                finished = False
                # This is not a complete operation
                continue

            for pred in preds:
                if pred.name not in cycles_end:
                    finished = False
                    continue

                pred_cycle = cycles_end[pred.name]
                print ("Have pred that finishes at cycle", pred_cycle)

                # Compute the time to this node:
                # TODO -- should we also account for not being
                # able to use the NOC for multiple things at once?
                distance = self.compute_communication_distance(pred, node)

                # Compute when this predecessor reaches this node:
                arrival_time = distance + pred_cycle
                earliest_time = max(earliest_time, arrival_time)

                # TODO --- compute a penalty based on the gap between
                # operations to account for buffering.

            # Check that the PE is actually free at this time --- if it
            # isn't, push the operation back.
            latency = operation_latency(node.operation)
            free_time = actual_schedule.get_free_time(earliest_time, latency, loc_x, loc_y)
            actual_schedule.set_operation(free_time, loc_x, loc_y, node, latency)
            if free_time != earliest_time:
                # We should probably punish the agent for this.
                # Doesn't have any correctness issues as long as we
                # assume infinite buffering (which we shouldn't do, and
                # will eventually fix).
                print("Place failed to place node in a sensible place: it is already in use!")

            # TODO --- do we need to punish this more? (i.e. integrate
            # buffering requirements?)

            # This node should run at the earliest time available.
            cycles_start[node.name] = free_time
            cycles_end[node.name] = free_time + operation_latency(node.operation)

            print ("Node ", node.name, "has earliest time", earliest_time)

        # Now that we've done that, we need to go through all the nodes and
        # work out the II.
        # When was this computation slot last used? (i.e. when could
        # we overlap the next iteration?)
        min_II = 0
        for x_loc, y_loc in actual_schedule.locations():
            # Now, we could achieve better performance
            # by overlapping these in a more fine-grained
            # manner --- but that seems like a lot of effort
            # for probably not much gain?
            # there ar probably loops where the gain
            # is not-so-marginal.
            if actual_schedule.has_use(x_loc, y_loc):
                # Can only do this for PEs that actually have uses!
                last_free = max(actual_schedule.free_times(x_loc, y_loc))
                first_alloc = min(actual_schedule.alloc_times(x_loc, y_loc))

                difference = last_free - first_alloc
                print ("Diff at loc", x_loc, y_loc, "is", difference)
                min_II = max(min_II, difference)

        # TODO --- we should probably return some kind of object
        # that would enable final compilation also.
        return min_II

compilation_session_cgra = CGRA(2, 2)

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
                )),
            ObservationSpace(name="II",
                space=Space(
                    int64_value=Int64Range(min=0)
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
        self.dfg = DFG.DFG(working_directory, benchmark, from_json='test/test_dfg.json')

        self.current_operation_index = 0
        self.time = 0 # Starting schedulign time --- we could do
        # this another way also, by asking the agent to come up with a raw
        # time rather than stepping through.
        # TODO -- load this properly.
        self.dfg_to_ops_list()

    def dfg_to_ops_list(self):
        # Embed the DFG into an operations list that we go through ---
        # it contains two things: the name of the node, and the index
        # that corresponds to within the Operations list.
        self.ops = []
        self.node_order = []
        for n in self.dfg.nodes:
            op = self.dfg.nodes[n]
            # Do we need to do a topo-sort here?
            if op.operation in Operations:
                ind = Operations.index(op.operation)
            else:
                print("Did not find operation " + op.operation + " in the set of Operations")
                ind = 0

            self.ops.append(ind)
            self.node_order.append(self.dfg.nodes[n])

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
            node = self.node_order[self.current_operation_index]
            latency = operation_latency(node.operation)
            op_set = self.schedule.set_operation(self.time, response - 1, node, latency)

            if op_set:
                had_effect = True
                self.current_operation_index += 1
        elif response == 0:
            self.time += 1

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
        elif observation_space.name == "II":
            return Event(int64_value=self.schedule.get_II(self.dfg))