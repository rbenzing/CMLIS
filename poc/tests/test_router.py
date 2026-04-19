from cmlis import router, topology
from cmlis.memctl import BindingPlan


def test_classify_boundaries():
    assert router.classify(128) is router.WorkloadClass.SHORT
    assert router.classify(2048) is router.WorkloadClass.MEDIUM
    assert router.classify(8192) is router.WorkloadClass.LONG


def test_short_prompt_caps_experts():
    d = router.decide(512, cores_per_node=8, mixture_of_experts=True)
    assert d.workload is router.WorkloadClass.SHORT
    assert d.active_experts == 2
    assert d.kv_cache_chunks == 1


def test_long_prompt_chunks_kv():
    d = router.decide(9000, cores_per_node=16)
    assert d.workload is router.WorkloadClass.LONG
    assert d.kv_cache_chunks >= 2
    assert d.active_experts == 0
    assert d.notes


def test_threads_leaves_headroom():
    d = router.decide(1024, cores_per_node=8)
    assert d.threads == 7


def test_route_index_selects_numa_node():
    d = router.decide(2048, cores_per_node=8, numa_nodes=2, route_index=3)
    assert d.prefer_numa_node == 1


def test_validate_binding_reports_mismatch():
    topo = topology.Topology(
        system="Linux",
        sockets=1,
        physical_cores=4,
        logical_cores=4,
        total_memory_mb=1024,
        numa_nodes=[
            topology.NumaNode(node_id=0, cpus=[0, 1], memory_mb=512),
            topology.NumaNode(node_id=1, cpus=[2, 3], memory_mb=512),
        ],
        numa_available=True,
    )
    decision = router.decide(2048, cores_per_node=2, numa_nodes=2, route_index=1)
    binding = BindingPlan(numa_node=0, cpus=[0, 1], enforced=True, prefix=["numactl"], notes=[])
    issues = router.validate_binding(decision, binding, topo)
    assert issues
