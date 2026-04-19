from cmlis import router


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


def test_threads_leaves_headroom():
    d = router.decide(1024, cores_per_node=8)
    assert d.threads == 7
