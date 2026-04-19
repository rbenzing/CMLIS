from cmlis import topology


def test_discover_returns_populated_struct():
    t = topology.discover()
    assert t.logical_cores >= 1
    assert t.total_memory_mb > 0
    assert len(t.numa_nodes) >= 1
    assert all(isinstance(n.node_id, int) for n in t.numa_nodes)


def test_summary_non_empty():
    t = topology.discover()
    s = t.summary()
    assert "Sockets" in s
    assert "NUMA nodes" in s
