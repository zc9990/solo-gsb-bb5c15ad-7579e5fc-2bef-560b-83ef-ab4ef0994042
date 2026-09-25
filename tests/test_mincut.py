"""最少断边分析的自测（无测试框架依赖）：python -m tests.test_mincut

使用独立临时 SQLite 库验证只读性，不触碰 ./data；
核心逻辑只依赖 analyzer/cyclonedx/mincut，无需安装 fastapi 即可运行。
"""

from __future__ import annotations

import copy
import itertools
import random
import tempfile
from pathlib import Path

from app import analyzer, cyclonedx, database, mincut

# 隔离的临时数据库（仅用于只读性验证）
_TMP_DIR = tempfile.mkdtemp(prefix="mincut-test-")
database.DB_PATH = Path(_TMP_DIR) / "test.db"
database.init_db()


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError("%s %s" % (name, detail))
    print("ok -", name)


def _demo_manifest():
    """构造含并行路径、汇合路径与有向环的图。

    祖先 app@1（GPL-3.0）到各 Proprietary 目标：

    * t@1：两条边不相交路径 app-a-x-t、app-b-y-t          -> 最少断 2 边；
    * conv@1：app-p-m-conv、app-q-m-conv 共享 m-conv      -> 最少断 1 边；
    * ring@1：app-u-ring 与 app-u-v-ring 共享 app-u，u/v 互连成环
                                                            -> 最少断 1 边；
    * direct@1：app-direct 直达                            -> 最少断 1 边。
    """
    return {
        "name": "mincut-demo",
        "packages": [
            {"name": "app", "version": "1", "license": "GPL-3.0"},
            {"name": "a", "version": "1", "license": "MIT"},
            {"name": "b", "version": "1", "license": "MIT"},
            {"name": "x", "version": "1", "license": "MIT"},
            {"name": "y", "version": "1", "license": "MIT"},
            {"name": "t", "version": "1", "license": "Proprietary"},
            {"name": "p", "version": "1", "license": "MIT"},
            {"name": "q", "version": "1", "license": "MIT"},
            {"name": "m", "version": "1", "license": "MIT"},
            {"name": "conv", "version": "1", "license": "Proprietary"},
            {"name": "u", "version": "1", "license": "MIT"},
            {"name": "v", "version": "1", "license": "MIT"},
            {"name": "ring", "version": "1", "license": "Proprietary"},
            {"name": "direct", "version": "1", "license": "Proprietary"},
        ],
        "edges": [
            # 两条边不相交路径（并行）
            {"source": "app@1", "target": "a@1"},
            {"source": "a@1", "target": "x@1"},
            {"source": "x@1", "target": "t@1"},
            {"source": "app@1", "target": "b@1"},
            {"source": "b@1", "target": "y@1"},
            {"source": "y@1", "target": "t@1"},
            # 汇合路径（共享 m -> conv）
            {"source": "app@1", "target": "p@1"},
            {"source": "p@1", "target": "m@1"},
            {"source": "app@1", "target": "q@1"},
            {"source": "q@1", "target": "m@1"},
            {"source": "m@1", "target": "conv@1"},
            # 环 u <-> v，且两条到 ring 的路径共享 app -> u
            {"source": "app@1", "target": "u@1"},
            {"source": "u@1", "target": "v@1"},
            {"source": "v@1", "target": "u@1"},
            {"source": "u@1", "target": "ring@1"},
            {"source": "v@1", "target": "ring@1"},
            # 直达边
            {"source": "app@1", "target": "direct@1"},
        ],
    }


def _demo_result():
    return analyzer.analyze_manifest(_demo_manifest())


def _conflict_by_pair(result, ancestor: str, node: str):
    hit = [
        c for c in result.get("conflicts", [])
        if c["ancestor"] == ancestor and c["node"] == node
    ]
    _check("conflict pair exists: %s -> %s" % (ancestor, node), len(hit) == 1, str(
        [(c["ancestor"], c["node"], c["severity"]) for c in result.get("conflicts", [])]
    ))
    return hit[0]


def _cut(rep):
    return [(e["source"], e["target"]) for e in rep["cut_edges"]]


def _reachable(edges, source: str, target: str, removed=None) -> bool:
    removed = removed or set()
    adjacency = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
    if source == target:
        return True
    seen = {source}
    stack = [source]
    while stack:
        cur = stack.pop()
        for nxt in adjacency.get(cur, []):
            if (cur, nxt) in removed or nxt in seen:
                continue
            if nxt == target:
                return True
            seen.add(nxt)
            stack.append(nxt)
    return False


def test_direct_edge() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "direct@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("direct cut count", rep["cut_count"] == 1, str(rep["cut_edges"]))
    _check("direct cut edge", _cut(rep) == [("app@1", "direct@1")])
    edge = rep["cut_edges"][0]
    _check("cut edge on original shortest path", edge["on_original_path"] is True)
    _check("cut edge flow path index", edge["flow_path"] == 1)
    _check("original shortest path echoed",
           rep["original_shortest_path"] == ["app@1", "direct@1"])
    _check("one edge-disjoint path", rep["edge_disjoint_path_count"] == 1)
    _check("reachable before", rep["reachable_before"] is True)
    _check("unreachable after", rep["reachable_after"] is False)
    _check("verified", rep["verified"] is True, str(rep["checks"]))


def test_parallel_paths_require_two_edges() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "t@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("parallel cut count is 2 (not shortest-path-only)",
           rep["cut_count"] == 2, str(rep["cut_edges"]))
    _check("cut edges deterministic",
           _cut(rep) == [("app@1", "a@1"), ("app@1", "b@1")], str(_cut(rep)))
    _check("two edge-disjoint paths", rep["edge_disjoint_path_count"] == 2,
           str(rep["edge_disjoint_paths"]))
    paths = sorted(tuple(p) for p in rep["edge_disjoint_paths"])
    _check("path set",
           paths == [("app@1", "a@1", "x@1", "t@1"),
                     ("app@1", "b@1", "y@1", "t@1")],
           str(paths))
    edges = [(e["source"], e["target"]) for e in result["edges"]]
    cut_set = set(_cut(rep))
    _check("deleting cut edges disconnects", not _reachable(edges, "app@1", "t@1", cut_set))
    # 关键：删除任意单条边都无法断开（证明不是只处理当前最短路径）
    alone = [
        (u, v) for (u, v) in edges
        if not _reachable(edges, "app@1", "t@1", {(u, v)})
    ]
    _check("no single edge suffices", alone == [], str(alone))
    # 原最短路径是 3 跳（app-a-x-t），而最少断边数 2，且不能只断最短路径上的一条
    _check("shortest path distance kept", c["distance"] == 3 and rep["original_distance"] == 3,
           str(c["distance"]))
    on_path = [e for e in rep["cut_edges"] if e["on_original_path"]]
    _check("only one cut edge lies on the BFS shortest path", len(on_path) == 1,
           str(rep["cut_edges"]))
    flow_paths_seen = {e["flow_path"] for e in rep["cut_edges"]}
    _check("cut edges cover both disjoint paths", flow_paths_seen == {1, 2})
    _check("verified", rep["verified"] is True, str(rep["checks"]))


def test_merging_paths_cut_shared_edge() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "conv@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("merging cut count is 1", rep["cut_count"] == 1, str(rep["cut_edges"]))
    _check("shared bottleneck edge chosen",
           _cut(rep) == [("m@1", "conv@1")], str(_cut(rep)))
    edges = [(e["source"], e["target"]) for e in result["edges"]]
    _check("deleting bottleneck disconnects",
           not _reachable(edges, "app@1", "conv@1", {("m@1", "conv@1")}))
    _check("one disjoint path reported", rep["edge_disjoint_path_count"] == 1)
    _check("verified", rep["verified"] is True, str(rep["checks"]))


def test_cycle_does_not_inflate_cut() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "ring@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("cycle graph cut count is 1", rep["cut_count"] == 1, str(rep["cut_edges"]))
    _check("shared edge app->u chosen",
           _cut(rep) == [("app@1", "u@1")], str(_cut(rep)))
    edges = [(e["source"], e["target"]) for e in result["edges"]]
    _check("unreachable after cut despite u<->v cycle",
           not _reachable(edges, "app@1", "ring@1", {("app@1", "u@1")}))
    # 环内删任何一条边（u->v / v->u）都不能断开 app -> ring
    _check("breaking the cycle alone does not help",
           _reachable(edges, "app@1", "ring@1", {("u@1", "v@1")}) and
           _reachable(edges, "app@1", "ring@1", {("v@1", "u@1")}))
    _check("verified with cycle", rep["verified"] is True, str(rep["checks"]))


def test_warning_entries_analyzable() -> None:
    # MIT 祖先 a -> Proprietary 目标 t 是“需人工确认”告警，同样可做断边分析
    result = _demo_result()
    c = _conflict_by_pair(result, "a@1", "t@1")
    _check("a->t is warning severity", c["severity"] == "warning", c["severity"])
    rep = mincut.analyze_mincut(result, c["id"])
    _check("warning cut count", rep["cut_count"] == 1)
    # a-x-t 是唯一路径，a->x 与 x->t 都是合法最小割（确定性选其一）
    _check("warning cut edge lies on unique path",
           _cut(rep) in ([("a@1", "x@1")], [("x@1", "t@1")]), str(_cut(rep)))
    _check("conflict copy carries severity",
           rep["conflict"]["severity"] == "warning" and rep["conflict_id"] == c["id"])


def test_deterministic_across_runs_and_orderings() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "t@1")
    rep1 = mincut.analyze_mincut(result, c["id"])
    rep2 = mincut.analyze_mincut(result, c["id"])
    _check("repeated runs identical cut", _cut(rep1) == _cut(rep2))
    _check("repeated runs identical paths",
           rep1["edge_disjoint_paths"] == rep2["edge_disjoint_paths"])

    # 打乱审计中边的存储顺序，割集与路径仍一致（输出按节点 id 字典序确定）
    shuffled = copy.deepcopy(result)
    random.Random(7).shuffle(shuffled["edges"])
    rep3 = mincut.analyze_mincut(shuffled, c["id"])
    _check("edge-order independent cut", _cut(rep3) == _cut(rep1), str(_cut(rep3)))
    _check("edge-order independent paths",
           rep3["edge_disjoint_paths"] == rep1["edge_disjoint_paths"],
           str(rep3["edge_disjoint_paths"]))


def test_conflict_not_found() -> None:
    result = _demo_result()
    try:
        mincut.analyze_mincut(result, "C9999")
    except mincut.ConflictNotFound as exc:
        _check("error names missing entry", "C9999" in str(exc), str(exc))
        return
    raise AssertionError("expected ConflictNotFound")

    # 直接调用 find_conflict 返回 None
    _check("find_conflict missing", mincut.find_conflict(result, "nope") is None)


def test_report_metadata() -> None:
    result = _demo_result()
    c = _conflict_by_pair(result, "app@1", "t@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("kind", rep["kind"] == "minimum-edge-cut")
    _check("read only flag", rep["read_only"] is True)
    _check("algorithm", rep["algorithm"] == "dinic-max-flow")
    _check("direction stated", "使用方 -> 被依赖方" in rep["direction"])
    _check("source brief", rep["source"]["id"] == "app@1" and rep["source"]["license"] == "GPL-3.0")
    _check("target brief", rep["target"]["id"] == "t@1" and rep["target"]["missing"] is False)
    _check("graph size", rep["graph"] == {"nodes": 14, "edges": 17}, str(rep["graph"]))


# --- 对随机小图做暴力枚举校验（割的最小性） -----------------------------------


def _bruteforce_mincut_size(edges, source, target):
    """按边子集大小递增枚举，返回最小断边数（与具体实现无关的参照）。"""
    if not _reachable(edges, source, target):
        return 0
    for k in range(1, len(edges) + 1):
        for combo in itertools.combinations(edges, k):
            if not _reachable(edges, source, target, set(combo)):
                return k
    return len(edges)


def test_random_graphs_against_bruteforce() -> None:
    rng = random.Random(20260924)
    checked = 0
    for _ in range(240):
        n = rng.randint(2, 6)
        nodes = ["n%d" % i for i in range(n)]
        edge_set = set()
        for _ in range(rng.randint(0, n * (n - 1))):
            u, v = rng.sample(nodes, 2)  # 无自环（分析阶段已剔除自依赖）
            edge_set.add((u, v))
        edges = sorted(edge_set)
        source, target = nodes[0], nodes[-1]
        result = {
            "audit_id": 1,
            "manifest_name": "rand",
            "summary": {"packages": n, "edges": len(edges)},
            "nodes": [{"id": x} for x in nodes],
            "edges": [{"source": u, "target": v, "from": "edges"} for u, v in edges],
            "conflicts": [
                {"id": "C001", "ancestor": source, "node": target,
                 "path": [source, target], "distance": 1,
                 "severity": "error", "reason": ""}
            ],
        }
        rep = mincut.analyze_mincut(result, "C001")
        expect = _bruteforce_mincut_size(edges, source, target)
        _check("random cut size equals brute force", rep["cut_count"] == expect,
               "edges=%r cut=%d brute=%d" % (edges, rep["cut_count"], expect))
        # 所有返回的割边确实让 target 不可达
        if expect:
            _check(
                "random cut disconnects",
                not _reachable(edges, source, target, set(_cut(rep))),
                str(_cut(rep)),
            )
            _check("random report verified", rep["verified"] is True, str(rep["checks"]))
        checked += 1
    print("ok - random graphs checked:", checked)


def test_cyclonedx_parallel_cut() -> None:
    bom = {
        "bomFormat": "CycloneDX",
        "components": [
            {"bom-ref": "app", "name": "app", "version": "1",
             "licenses": [{"license": {"id": "GPL-3.0"}}]},
            {"bom-ref": "a", "name": "a", "version": "1",
             "licenses": [{"license": {"id": "MIT"}}]},
            {"bom-ref": "b", "name": "b", "version": "1",
             "licenses": [{"license": {"id": "MIT"}}]},
            {"bom-ref": "t", "name": "t", "version": "1",
             "licenses": [{"license": {"id": "Proprietary"}}]},
        ],
        "dependencies": [
            {"ref": "app", "dependsOn": ["a", "b"]},
            {"ref": "a", "dependsOn": ["t"]},
            {"ref": "b", "dependsOn": ["t"]},
        ],
    }
    result = cyclonedx.analyze_bom(bom)
    c = _conflict_by_pair(result, "app@1", "t@1")
    rep = mincut.analyze_mincut(result, c["id"])
    _check("cyclonedx parallel cut count", rep["cut_count"] == 2, str(rep["cut_edges"]))
    _check("cyclonedx audit meta carries format",
           rep["audit"].get("source_format") == "cyclonedx")
    _check("cyclonedx verified", rep["verified"] is True)


def test_analysis_is_read_only() -> None:
    manifest = _demo_manifest()
    result = analyzer.analyze_manifest(manifest)
    audit_id = database.save_audit(result, manifest)

    stored_before = database.get_audit(audit_id)
    audits_before = database.list_audits(limit=500)
    reviews_before = database.list_reviews(limit=500)
    snapshot = copy.deepcopy(stored_before)

    for cid in ["C001", "C002", "C003"]:
        if mincut.find_conflict(stored_before, cid):
            rep = mincut.analyze_mincut(stored_before, cid)
            _check("report read_only for %s" % cid, rep["read_only"] is True)

    _check("input result not mutated", stored_before == snapshot)
    _check("no new audit created", database.list_audits(limit=500) == audits_before)
    _check("stored audit unchanged", database.get_audit(audit_id) == snapshot)
    _check("reviews untouched", database.list_reviews(limit=500) == reviews_before)

    # 打一条复核标记后再分析：标记不受影响
    from app import reviews as reviews_mod
    conflict = stored_before["conflicts"][0]
    parts = reviews_mod.fingerprint_parts(stored_before, conflict)
    database.save_review(
        fingerprint=reviews_mod.conflict_fingerprint(*parts),
        ancestor=parts[0], node=parts[1],
        ancestor_license=parts[2], node_license=parts[3],
        note="法务确认",
    )
    reviews_with_mark = database.list_reviews(limit=500)
    _check("one review exists", len(reviews_with_mark) == 1)
    mincut.analyze_mincut(database.get_audit(audit_id), conflict["id"])
    _check("review survives analysis untouched",
           database.list_reviews(limit=500) == reviews_with_mark)
    _check("stored audit still identical", database.get_audit(audit_id) == snapshot)


def test_api_layer() -> None:
    """可选的 HTTP 层测试：缺少 fastapi/httpx 时跳过（核心逻辑不依赖它们）。"""
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover - 环境缺依赖时跳过
        print("ok - api layer test skipped (fastapi/httpx unavailable)")
        return

    api_dir = tempfile.mkdtemp(prefix="mincut-api-")
    from fastapi import FastAPI  # noqa: F401
    from app import main as main_mod
    from app import database as db_mod

    db_mod.DB_PATH = Path(api_dir) / "api.db"
    db_mod.init_db()
    client = TestClient(main_mod.app)

    manifest = _demo_manifest()
    resp = client.post("/api/audit", json={"manifest": manifest})
    _check("api audit created", resp.status_code == 200, resp.text)
    aid = resp.json()["audit_id"]
    conflict = _conflict_by_pair(resp.json(), "app@1", "t@1")

    rep = client.get("/api/audits/%s/mincut/%s" % (aid, conflict["id"]))
    _check("api mincut 200", rep.status_code == 200, rep.text)
    data = rep.json()
    _check("api cut count 2", data["cut_count"] == 2, str(data["cut_edges"]))
    _check("api unreachable after cut", data["reachable_after"] is False)

    missing_audit = client.get("/api/audits/999999/mincut/C001")
    _check("api missing audit 404", missing_audit.status_code == 404, missing_audit.text)
    _check("api missing audit message", "审计记录不存在" in missing_audit.json()["detail"])

    missing_conflict = client.get("/api/audits/%s/mincut/C9999" % aid)
    _check("api missing conflict 404",
           missing_conflict.status_code == 404, missing_conflict.text)
    _check("api missing conflict message",
           "不存在冲突条目" in missing_conflict.json()["detail"])

    # 只读：审计列表不增加，复核标记不变化
    items = client.get("/api/audits?limit=500").json()["items"]
    _check("api no extra audit rows", len(items) == 1, str(len(items)))
    _check("api reviews empty", client.get("/api/reviews").json()["items"] == [])


def main() -> None:
    test_direct_edge()
    test_parallel_paths_require_two_edges()
    test_merging_paths_cut_shared_edge()
    test_cycle_does_not_inflate_cut()
    test_warning_entries_analyzable()
    test_deterministic_across_runs_and_orderings()
    test_conflict_not_found()
    test_report_metadata()
    test_random_graphs_against_bruteforce()
    test_cyclonedx_parallel_cut()
    test_analysis_is_read_only()
    test_api_layer()
    print("\nall mincut tests passed.")


if __name__ == "__main__":
    main()
