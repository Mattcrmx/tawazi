import os
from functools import partial
from typing import Any, Tuple

import pytest
from networkx import NetworkXUnfeasible
from tawazi import DAG, dag, xn
from tawazi._dag.digraph import DiGraphEx
from tawazi._dag.helpers import execute
from tawazi.node import ExecNode, UsageExecNode


def shortcut_execute(dag: DAG[Any, Any], graph: DiGraphEx) -> Any:
    return execute(
        results=dag.results,
        exec_nodes=dag.exec_nodes,
        max_concurrency=dag.max_concurrency,
        graph=graph,
    )


def a(c: str) -> str:
    return "a"


def b(a: str) -> str:
    return a + "b"


def c(b: str) -> str:
    return b + "c"


def func(a: str, b: str) -> str:
    return a + b


def test_same_constant_name_in_two_exec_nodes() -> None:
    @xn
    def a(cst: int) -> int:
        return cst

    @xn
    def b(a: int, cst: str) -> str:
        return str(a) + cst

    @dag
    def my_dag() -> None:
        var_a = a(1234)
        b(var_a, "poulpe")

    exec_nodes, results = shortcut_execute(my_dag, my_dag.graph_ids.make_subgraph())
    assert len(exec_nodes) == 4
    assert results[a.id] == 1234
    assert results[b.id] == "1234poulpe"


def test_dag_with_weird_nodes() -> None:
    @xn
    def toto(a: str, b: str) -> str:
        return a + "_wow_" + b

    @dag
    def my_dag() -> Tuple[Any, ...]:
        var_a = xn(partial(func, a="x"))(b="y")
        var_b = xn(partial(func, b="y"))(a="x")
        var_c = xn(lambda x: x)("e")
        var_d = toto(var_a, var_b)

        return var_a, var_b, var_c, var_d

    res_a, res_b, res_c, res_d = my_dag()

    assert res_a == "xy"
    assert res_b == "xy"
    assert res_c == "e"
    assert res_d == "xy_wow_xy"


def test_setup_debug_nodes() -> None:
    with pytest.raises(ValueError):

        @xn(debug=True, setup=True)
        def a() -> None:
            ...


def test_circular_deps() -> None:
    en_a = ExecNode(id_="a", exec_function=a, args=[], is_sequential=True)
    en_b = ExecNode(
        id_="b", exec_function=b, args=[UsageExecNode(en_a.id)], priority=2, is_sequential=False
    )
    en_c = ExecNode(
        id_="c", exec_function=c, args=[UsageExecNode(en_a.id)], priority=1, is_sequential=False
    )
    en_a.args = [UsageExecNode(en_c.id)]

    with pytest.raises(NetworkXUnfeasible):
        DAG({xn.id: xn for xn in [en_a, en_b, en_c]}, [], [], 2)


def test_non_pickalable_args() -> None:
    @xn
    def _a(in1: Any = 1) -> Any:
        if isinstance(in1, int):
            return in1 + 1
        return in1

    @xn
    def _b(in1: int, in2: int) -> int:
        return in1 + in2

    @dag
    def pipe() -> Tuple[Any, int]:
        return _a(os), _b(_a(10), _a())

    assert pipe() == (os, 13)
