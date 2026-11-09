"""Memory shared by the MPI ranks of a node.

The distributed PCG keeps its vectors on the pixels of each rank, but P reads
the whole search direction and P^T (like the coverage and the blocks of the
preconditioner) sums into the whole map: one full map per rank would be
(ranks per node) copies on every node, 1.2 GB each at nside 2048 with I/Q/U.
Here the ranks of a node share one copy, in an MPI-3 shared-memory window.

Needs only mpi4py, imported when a communicator is given. The pixels are
assigned to the ranks node by node (see `Node.pixels`), so each node owns a
contiguous range of the map, and only the node leaders exchange data between
nodes: an allgather of the node ranges for P, a reduce-scatter for the sums.
"""

import atexit
import weakref

import numpy as np

_NODES = {}  # communicator handle -> Node
_TEST_NODE = None  # tests: rank -> fake node within the real node, e.g. two "nodes" on one machine


def node_of(comm):
    """The Node of `comm` (collective the first time for each communicator).
    Cached: a communicator split for every GLS would leak, and the shared
    work maps are reused by the GLS of the same communicator."""
    node = _NODES.get(comm.handle)
    if node is None or not node.matches(comm):
        # a freed communicator whose handle was reused: its windows are not
        # freed (collective on ranks that may no longer be all here)
        if not _NODES:
            atexit.register(_free_at_exit)  # after mpi4py's, so it runs before MPI_Finalize
        node = _NODES[comm.handle] = Node(comm)
    return node


class Layout:
    """The pixels of a node: rows [lo, hi) of the full map, and on the leaders
    the number of rows of every node (`counts`, in node order)."""

    def __init__(self, lo, hi, counts):
        self.lo, self.hi, self.counts = lo, hi, counts


class Node:
    """The ranks of `comm` grouped by node.

    `pixels` is `comm` with the ranks of each node consecutive (the same
    order as `comm` when the ranks are placed on the nodes in blocks), used to
    split the pixels among the ranks: a node owns the pixels of its ranks, a
    contiguous range. `node` holds the ranks of one node, `leaders` its rank
    0 on every node (MPI.COMM_NULL elsewhere), in node order.

    The shared arrays are in windows with a passive-target epoch open on all
    the ranks of the node (MPI's unified memory model: plain loads and
    stores, made visible to the other ranks by `barrier`)."""

    def __init__(self, comm):
        from mpi4py import MPI

        self._group = comm.Get_group()
        self.node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
        if _TEST_NODE is not None:
            self.node = self.node.Split(_TEST_NODE(comm.rank), key=comm.rank)
        leader = self.node.bcast(comm.rank)  # the lowest rank of comm on the node
        key = comm.allgather((leader, self.node.rank))
        slot = sorted(key).index(key[comm.rank])
        self.pixels = comm.Split(0, slot)
        self.first = self.node.bcast(slot)  # the slot of the node's rank 0: its slots are first, first + 1, ...
        self.leader = self.node.rank == 0
        self.leaders = comm.Split(0 if self.leader else MPI.UNDEFINED, slot)
        self._wins = {}  # name -> (window, number of float64)
        self._outs = []  # (window, weak reference to its array): the outputs, see `output`

    def matches(self, comm):
        """Still the communicator this Node was made for (handles are reused)."""
        from mpi4py import MPI

        group = comm.Get_group()
        same = MPI.Group.Compare(self._group, group) == MPI.IDENT
        group.Free()
        return same

    def layout(self, bounds):
        """The Layout of this node for the rows `bounds` (size + 1,) of the slots
        of `pixels`. Collective on the leaders."""
        lo, hi = int(bounds[self.first]), int(bounds[self.first + self.node.size])
        counts = np.asarray(self.leaders.allgather(hi - lo), np.int64) if self.leader else None
        return Layout(lo, hi, counts)

    def work(self, name, n):
        """A float64 array of n values shared by the ranks of the node (collective
        on the node). One per `name`, reused and enlarged when needed: fetch it
        again before each use (an enlargement frees the old one), after a
        `barrier` if other ranks may still be using it."""
        from mpi4py import MPI

        win, size = self._wins.get(name, (None, 0))
        if n > size:
            if win is not None:
                win.Unlock_all()
                win.Free()
            win = MPI.Win.Allocate_shared(8 * n if self.leader else 0, 8, comm=self.node)
            win.Lock_all(MPI.MODE_NOCHECK)
            self._wins[name] = win, n
        buf, _ = win.Shared_query(0)
        return np.ndarray(n, np.float64, buffer=buf)

    def output(self, shape, dtype):
        """A new array shared by the ranks of the node, for a result handed to
        the caller (collective on the node). Not reused: its window is freed by
        a later `output` once no rank of the node references the array (or a
        view of it), so the ranks must not write it after it is handed out
        (see `freeze`). First frees the windows of the outputs released."""
        from mpi4py import MPI

        self._collect()
        dtype = np.dtype(dtype)
        n = int(np.prod(shape, dtype=np.int64))
        win = MPI.Win.Allocate_shared(max(n * dtype.itemsize, 1) if self.leader else 0, dtype.itemsize, comm=self.node)
        win.Lock_all(MPI.MODE_NOCHECK)
        buf, _ = win.Shared_query(0)
        # views of `out` keep `out` as their base (numpy stops at an array
        # whose base is not an array): the weak reference sees them all
        out = np.ndarray(shape, dtype, buffer=buf)
        self._outs.append((win, weakref.ref(out)))
        return out

    def _collect(self):
        """Free the windows of the outputs no rank of the node references (collective on the node)."""
        from mpi4py import MPI

        if not self._outs:
            return
        alive = np.array([ref() is not None for _, ref in self._outs])
        self.node.Allreduce(MPI.IN_PLACE, alive, op=MPI.LOR)
        for (win, _), keep in zip(self._outs, alive):
            if not keep:
                win.Unlock_all()
                win.Free()
        self._outs = [o for o, keep in zip(self._outs, alive) if keep]

    def freeze(self, out):
        """Hand out an `output` written by the ranks of the node: wait for them
        all, then make it read-only (it is the same memory on every rank)."""
        self.barrier()
        out.flags.writeable = False
        return out

    def part(self, n):
        """This rank's share [a, b) of n items split among the ranks of the node."""
        r, s = self.node.rank, self.node.size
        return r * n // s, (r + 1) * n // s

    def gather_rows(self, full, layout):
        """Fill the columns of the other nodes in the shared `full` (k, nobs),
        whose columns [lo, hi) of this node its ranks have written: an
        allgather of each row on the leaders. Collective."""
        from mpi4py import MPI

        self.barrier()
        if self.leader and self.leaders.size > 1:
            counts = layout.counts.tolist()
            displs = np.concatenate([[0], np.cumsum(layout.counts)[:-1]]).tolist()
            for row in full.reshape(-1, full.shape[-1]):
                self.leaders.Allgatherv(MPI.IN_PLACE, [row, (counts, displs)])  # datatype from the dtype
        self.barrier()

    def barrier(self):
        """Wait for the ranks of the node, the stores of each one visible to all."""
        wins = [w for w, _ in self._wins.values()] + [w for w, _ in self._outs]
        for win in wins:
            win.Sync()
        self.node.Barrier()
        for win in wins:
            win.Sync()

    def allgather(self, full, layout):
        """Fill the rows of the other nodes in the shared `full` (nobs, k), whose
        rows of this node its ranks have written. Collective."""
        from mpi4py import MPI

        self.barrier()
        if self.leader and self.leaders.size > 1:
            counts = layout.counts * full.shape[1]
            displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
            self.leaders.Allgatherv(MPI.IN_PLACE, [full, counts.tolist(), displs.tolist(), MPI.DOUBLE])
        self.barrier()

    def reduce(self, full, layout, by_column=False):
        """Sum the shared `full` (nobs, k) of the nodes: afterwards the rows
        [lo, hi) of this node are the rows [0, hi - lo) of `full` (the other
        rows are garbage). Collective. by_column caps the temporaries of MPI at
        one column (on the leaders), at the price of copies."""
        from mpi4py import MPI

        self.barrier()
        n = layout.hi - layout.lo
        if self.leader and self.leaders.size > 1:
            if by_column:
                col = np.empty(full.shape[0])
                for j in range(full.shape[1]):
                    col[:] = full[:, j]
                    self.leaders.Reduce_scatter(MPI.IN_PLACE, [col, MPI.DOUBLE], recvcounts=layout.counts.tolist())
                    full[:n, j] = col[:n]
            else:
                self.leaders.Reduce_scatter(MPI.IN_PLACE, [full, MPI.DOUBLE],
                                            recvcounts=(layout.counts * full.shape[1]).tolist())
        self.barrier()

    def free(self):
        """Free the shared windows (collective on the node)."""
        for win, _ in list(self._wins.values()) + self._outs:
            win.Unlock_all()
            win.Free()
        self._wins, self._outs = {}, []


def _free_at_exit():
    """MPI windows must be freed before MPI_Finalize (mpi4py calls it at exit):
    some MPI implementations (Intel MPI) hang otherwise."""
    from mpi4py import MPI

    if not MPI.Is_finalized():
        for node in _NODES.values():
            node.free()
    _NODES.clear()
