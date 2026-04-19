import platform

from cmlis import memctl


def test_non_linux_returns_noop():
    if platform.system() == "Linux":
        return
    plan = memctl.build_binding(0, [0, 1, 2, 3])
    assert plan.enforced is False
    assert plan.prefix == []
    assert any("unavailable" in n for n in plan.notes)


def test_linux_prefix_shape():
    if platform.system() != "Linux":
        return
    plan = memctl.build_binding(0, [0, 1])
    # Either numactl, taskset, or both should appear if installed
    text = " ".join(plan.prefix)
    if plan.enforced:
        assert "numactl" in text or "taskset" in text
