"""
test_ray_cluster.py

Run this ON the head node (after `ray start --head` on the head and
`ray start --address=...` on the worker) to confirm the cluster is
genuinely running across both Linux machines, not just executing
everything locally on whichever one you happen to run this from.

    python3 test_ray_cluster.py
"""

import os
import socket
import time
from collections import Counter

import ray


def main():

    # address="auto" finds the Ray instance already running on this
    # machine (the one started by `ray start --head`), rather than
    # spinning up a new local-only instance.
    ray.init(address="auto")

    # ----------------------------------------------------------
    # Step 1: does Ray itself think there are 2 nodes?
    # ----------------------------------------------------------

    print("=== Cluster resources ===")
    print(ray.cluster_resources())

    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n["Alive"]]

    print(f"\n{len(alive_nodes)} alive node(s):")
    for n in alive_nodes:
        cpus = n["Resources"].get("CPU", 0)
        print(f"  {n['NodeManagerHostname']} ({n['NodeManagerAddress']}) - {cpus} CPUs")

    if len(alive_nodes) < 2:
        print(
            "\nFAIL: only 1 node visible to Ray. The second machine "
            "hasn't joined - check that you ran "
            "`ray start --address='<head-ip>:6379'` on it, and that "
            "`ray status` (run on either machine) also shows 2 nodes."
        )
        return

    print(f"\nOK: Ray sees {len(alive_nodes)} nodes.")

    # ----------------------------------------------------------
    # Step 2: does work actually GET DISTRIBUTED to both, not just
    # scheduled-but-somehow-still-run-on-one? This is the real test -
    # step 1 alone only proves the nodes are connected, not that
    # tasks actually execute on the second machine.
    # ----------------------------------------------------------

    @ray.remote
    def get_worker_info(i):
        # Small delay so 20 tasks can't all finish on one worker
        # before the scheduler has a chance to spread the rest to
        # the other node - without this, fast no-op tasks can
        # sometimes all land on whichever worker was already warm.
        time.sleep(0.5)
        return {
            "task": i,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }

    print("\n=== Submitting 20 tasks to see which machine each runs on ===")

    futures = [get_worker_info.remote(i) for i in range(20)]
    results = ray.get(futures)

    hostname_counts = Counter(r["hostname"] for r in results)

    print("\nTasks per machine:")
    for hostname, count in hostname_counts.items():
        print(f"  {hostname}: {count} tasks")

    if len(hostname_counts) >= 2:
        print(
            f"\nPASS: tasks actually ran on {len(hostname_counts)} "
            f"different machines - the cluster is genuinely "
            f"distributing work, not just connected."
        )
    else:
        print(
            "\nFAIL: every task ran on a single machine "
            f"({next(iter(hostname_counts))}) despite 2 nodes being "
            "visible. The cluster is connected but not actually "
            "spreading work - check for resource limits (e.g. "
            "num_cpus set very low on one node) or try increasing "
            "the number of tasks submitted above."
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
