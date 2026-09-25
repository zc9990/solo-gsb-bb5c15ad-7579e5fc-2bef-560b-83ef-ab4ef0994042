"""最少断边分析（只读）。

对一条已保存审计中的冲突/告警，沿现有 **使用方 -> 被依赖方** 依赖图，
计算要让该条冲突的祖先包（source）无法再到达被依赖包（target），
所需删除的**最少依赖边**以及一组确定性的边集合（s-t 最小边割）。

理论依据为有向图版 Menger 定理：source 到 target 的两两边不相交
（edge-disjoint）有向路径的最大条数，恰好等于使 target 从 source
不可达所需删除的最少边数。实现上对每条依赖边赋容量 1，用
Dinic 最大流求最大流值，再从 source 在残量网络中的可达集合 R
提取割集——所有满足 ``u ∈ R、v ∉ R`` 的前向边构成一个最小割
（它们在最大流中全部饱和，数量恰好等于最大流值）。

该方法对图结构没有限制：

* **并行路径**——多条互不相干的 source→target 路径必须各断一条，
  边不相交路径数即为最少断边数，不能只处理当前最短路径；
* **汇合路径**——多条路径共享边/汇合于同一节点时，断共享边可一条
  顶多条，最大流割集会自动选出这种“瓶颈边”；
* **有向环**——Dinic 的 BFS 分层只走残量正容量边，环既不影响
  最大流值也不影响割集；流分解时如遇环会先摘除环再继续。

只允许删边、不允许删节点；分析为纯内存只读计算：不创建审计、
不改动任何复核标记。原生 manifest 与 CycloneDX 审计结果结构一致，
均可直接分析。

确定性：邻接表按 ``(目标节点, 边序号)`` 字典序排序，Dinic 的增广
与残量 BFS 均按此固定顺序进行；割边按 ``(source, target)`` 字典序
返回；边不相交路径按 flow 分解时的字典序返回。同一份审计与同一条
冲突，任意时刻计算结果完全一致。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

KIND = "minimum-edge-cut"

DIRECTION = "依赖边方向：使用方 -> 被依赖方；沿现有依赖图求祖先包→被依赖包的最小边割。"

ALGORITHM = "dinic-max-flow"  # 每条依赖边容量 1；依据 Menger 定理等价于最少断边数


class ConflictNotFound(LookupError):
    """审计存在但指定的冲突/告警条目不存在（API 层转为 404）。"""


# --- 图构建 -----------------------------------------------------------------


def _edge_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按审计存储顺序取去重后的依赖边（分析阶段已去重，这里再兜底一次）。"""
    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for edge in result.get("edges", []) or []:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "source": source,
                "target": target,
                "from": edge.get("from"),
            }
        )
    return records


def _node_map(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(n.get("id")): n for n in result.get("nodes", []) or []}


def _audit_meta(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.get("summary") or {}
    return {
        "audit_id": result.get("audit_id"),
        "manifest_name": result.get("manifest_name"),
        "status": result.get("status"),
        "created_at": result.get("created_at"),
        "source_format": result.get("source_format"),
        "packages": summary.get("packages"),
        "edges": summary.get("edges"),
        "conflicts": summary.get("conflicts"),
        "license_warnings": summary.get("license_warnings"),
    }


def _node_brief(node: Optional[Dict[str, Any]], node_id: str) -> Dict[str, Any]:
    if node is None:
        return {"id": node_id, "name": None, "version": None,
                "license_display": None, "license_category": None, "missing": True}
    return {
        "id": node.get("id"),
        "name": node.get("name"),
        "version": node.get("version"),
        "version_unspecified": node.get("version_unspecified", False),
        "license_display": node.get("license_display"),
        "license": node.get("license"),
        "license_category": node.get("license_category"),
        "missing": False,
    }


# --- Dinic 最大流（单位容量边） ----------------------------------------------
#
# 每条前向弧对应一条依赖边（容量 1），反向弧为残量弧（初始容量 0）。
# 弧用可变列表表示：[目标节点, 反向弧下标, 残量容量, 对应边序号(仅前向弧)]。


def _build_network(
    edges: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[list]], Dict[Tuple[str, str], int]]:
    """构建 Dinic 邻接网络（按 (目标, 边序号) 字典序排序，保证确定性）。"""
    graph: Dict[str, List[list]] = {}
    edge_index: Dict[Tuple[str, str], int] = {}
    pairs: List[Tuple[list, list]] = []
    for idx, edge in enumerate(edges):
        u, v = edge["source"], edge["target"]
        graph.setdefault(u, [])
        graph.setdefault(v, [])
        edge_index[(u, v)] = idx
        # 反向弧的反向下标先占位，排序后再统一绑定，避免排序导致错位
        forward = [v, -1, 1, idx]
        backward = [u, -1, 0, None]
        pairs.append((forward, backward))
        graph[u].append(forward)
        graph[v].append(backward)

    for u in graph:
        graph[u].sort(key=lambda arc: (arc[0], -1 if arc[3] is None else arc[3]))

    # 排序完成后按对象身份绑定互反下标
    position: Dict[str, Dict[int, int]] = {
        u: {id(arc): i for i, arc in enumerate(arcs)} for u, arcs in graph.items()
    }
    for forward, backward in pairs:
        u, v = backward[0], forward[0]
        forward[1] = position[v][id(backward)]
        backward[1] = position[u][id(forward)]
    return graph, edge_index


def _dinic_max_flow(
    graph: Dict[str, List[list]],
    source: str,
    sink: str,
) -> int:
    """Dinic 求 source→sink 最大流（迭代实现，避免长链图递归过深）。

    邻接顺序固定，增广路径的搜索按同一顺序推进，结果确定。
    每条前向弧容量为 1，每次增广恰好推送 1 单位流。
    """
    if source not in graph or sink not in graph or source == sink:
        return 0

    def bfs_level() -> Dict[str, int]:
        levels: Dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])
        while queue:
            current = queue.popleft()
            for arc in graph[current]:
                if arc[2] > 0 and arc[0] not in levels:
                    levels[arc[0]] = levels[current] + 1
                    queue.append(arc[0])
        return levels

    flow = 0
    while True:
        levels = bfs_level()
        if sink not in levels:
            break
        ptr: Dict[str, int] = {u: 0 for u in graph}
        # 迭代寻找一条分层增广路径；单位容量下找到即推送 1 单位
        while True:
            path_nodes: List[str] = [source]
            path_arcs: List[list] = []
            while path_nodes and path_nodes[-1] != sink:
                current = path_nodes[-1]
                arcs = graph[current]
                advanced = False
                while ptr[current] < len(arcs):
                    arc = arcs[ptr[current]]
                    if arc[2] > 0 and levels.get(arc[0]) == levels[current] + 1:
                        path_arcs.append(arc)
                        path_nodes.append(arc[0])
                        advanced = True
                        break
                    ptr[current] += 1
                if not advanced:
                    # 当前层图中 current 是死端：标记后回溯，并推进父节点弧指针
                    levels[current] = -1
                    path_nodes.pop()
                    if path_arcs:
                        path_arcs.pop()
                    if path_nodes:
                        ptr[path_nodes[-1]] += 1
            if not path_nodes or path_nodes[-1] != sink:
                break  # 该分层下不存在更多增广路径
            for arc in path_arcs:
                arc[2] -= 1
                graph[arc[0]][arc[1]][2] += 1
            flow += 1
    return flow


def _residual_reachable(graph: Dict[str, List[list]], source: str) -> Set[str]:
    """最大流后沿残量正容量弧从 source 做 BFS，返回源侧集合 R。"""
    if source not in graph:
        return {source} if source else set()
    seen = {source}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        for arc in graph[current]:
            if arc[2] > 0 and arc[0] not in seen:
                seen.add(arc[0])
                queue.append(arc[0])
    return seen


def _plain_reachable(adjacency: Dict[str, List[str]], source: str, target: str) -> bool:
    """原图（或删边后的图）中 source 是否可沿边到达 target（BFS，含环处理）。"""
    if source == target:
        return True
    seen = {source}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        for nxt in adjacency.get(current, []):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _flow_edge_paths(
    graph: Dict[str, List[list]],
    source: str,
    sink: str,
    flow_value: int,
) -> List[List[str]]:
    """从饱和前向弧（承载 1 单位流）分解出 source→sink 的边不相交路径。

    每个节点的出流弧按目标字典序取用，路径集合确定。流守恒保证
    分解出的路径数等于最大流值；若沿当前路径回到栈中节点（有向环），
    先摘除环再继续沿新出弧前进。
    """
    outgoing: Dict[str, List[str]] = {}
    for u, arcs in graph.items():
        used = [arc[0] for arc in arcs if arc[3] is not None and arc[2] == 0]
        if used:
            outgoing[u] = sorted(used)  # 容量 1：饱和即承载流

    paths: List[List[str]] = []
    for _ in range(flow_value):
        if not outgoing.get(source):
            break  # 兜底：理论上流守恒保证不会发生
        path = [source]
        index_on_path: Dict[str, int] = {source: 0}
        while path[-1] != sink:
            current = path[-1]
            bucket = outgoing.get(current)
            if not bucket:
                break  # 兜底
            nxt = bucket.pop(0)
            if not bucket:
                outgoing.pop(current, None)
            if nxt in index_on_path:
                # 走到当前路径上的节点 -> 经过了一个有向环；丢弃环尾，
                # 被弹出的环上其余弧已无悬挂流量（流守恒），继续前进。
                loop_start = index_on_path[nxt]
                for dropped in path[loop_start + 1:]:
                    index_on_path.pop(dropped, None)
                del path[loop_start + 1:]
                continue
            index_on_path[nxt] = len(path)
            path.append(nxt)
        if path[-1] == sink:
            paths.append(path)
    return paths


# --- 分析入口 ---------------------------------------------------------------


def find_conflict(result: Dict[str, Any], conflict_id: str) -> Optional[Dict[str, Any]]:
    """按 id（如 C001）在审计结果中定位冲突/告警条目。"""
    cid = str(conflict_id or "").strip()
    for conflict in result.get("conflicts", []) or []:
        if str(conflict.get("id")) == cid:
            return conflict
    return None


def analyze_mincut(result: Dict[str, Any], conflict_id: str) -> Dict[str, Any]:
    """对一份审计结果中的单条冲突做最少断边分析，返回报告（只读，不改写输入）。

    审计中不存在该冲突 id 时抛出 :class:`ConflictNotFound`（API 层 404）。
    """
    conflict = find_conflict(result, conflict_id)
    if conflict is None:
        total = len(result.get("conflicts", []) or [])
        raise ConflictNotFound(
            "审计 #%s 中不存在冲突条目：%s。请从该审计的冲突/告警列表中选择（共 %d 条）。"
            % (result.get("audit_id"), conflict_id, total)
        )

    source = str(conflict["ancestor"])
    sink = str(conflict["node"])
    edges = _edge_records(result)
    nodes = _node_map(result)

    # 原图邻接（确定性顺序）：用于可达性校验
    adjacency: Dict[str, List[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], []).append(edge["target"])
    for u in adjacency:
        adjacency[u].sort()

    reachable_before = _plain_reachable(adjacency, source, sink)
    graph, edge_index = _build_network(edges)
    flow_value = _dinic_max_flow(graph, source, sink)
    source_side = _residual_reachable(graph, source) if source in graph else {source}

    # 最小割：源侧 R -> 汇侧（非 R）的全部前向边（最大流后必然全部饱和）
    cut_keys: List[Tuple[str, str]] = sorted(
        (u, v)
        for (u, v), idx in edge_index.items()
        if u in source_side and v not in source_side
    )
    cut_edges_record: List[Dict[str, Any]] = []
    for u, v in cut_keys:
        idx = edge_index[(u, v)]
        cut_edges_record.append(dict(edges[idx]))

    # 删边后重算可达性（不允许只信最大流结论，做一次独立 BFS 校验）
    cut_set = set(cut_keys)
    adjacency_after: Dict[str, List[str]] = {
        u: [v for v in vs if (u, v) not in cut_set]
        for u, vs in adjacency.items()
    }
    reachable_after = _plain_reachable(adjacency_after, source, sink)

    # 流分解：边不相交路径（正常情形全部位于 source→sink；含环时先摘环）
    flow_paths = _flow_edge_paths(graph, source, sink, flow_value) if reachable_before else []
    path_index_by_edge: Dict[Tuple[str, str], int] = {}
    used_flow_edges: Set[Tuple[str, str]] = set()
    paths_are_disjoint = True
    paths_are_valid = True
    for i, path in enumerate(flow_paths, start=1):
        if path[0] != source or path[-1] != sink or len(path) < 2:
            paths_are_valid = False
        for u, v in zip(path, path[1:]):
            if (u, v) in used_flow_edges:
                paths_are_disjoint = False
            used_flow_edges.add((u, v))
            path_index_by_edge.setdefault((u, v), i)

    original_path = [str(p) for p in conflict.get("path", [])]
    original_path_edges = {
        (a, b) for a, b in zip(original_path, original_path[1:])
    }

    cut_payload: List[Dict[str, Any]] = []
    for edge in cut_edges_record:
        key = (edge["source"], edge["target"])
        cut_payload.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "from": edge["from"],
                "on_original_path": key in original_path_edges,
                "flow_path": path_index_by_edge.get(key),
            }
        )

    checks = {
        "cut_size_equals_max_flow": len(cut_payload) == flow_value,
        "paths_are_edge_disjoint": (
            paths_are_disjoint
            and paths_are_valid
            and len(flow_paths) == flow_value
        ),
        "unreachable_after_cut": not reachable_after,
    }
    verified = all(checks.values()) and reachable_before

    return {
        "kind": KIND,
        "read_only": True,
        "direction": DIRECTION,
        "algorithm": ALGORITHM,
        "algorithm_note": "每条依赖边容量 1 的最大流；依据 Menger 定理，最大流值=边不相交路径数=最少断边数。",
        "audit": _audit_meta(result),
        "conflict_id": conflict.get("id"),
        "conflict": dict(conflict),
        "source": _node_brief(nodes.get(source), source),
        "target": _node_brief(nodes.get(sink), sink),
        "reachable_before": reachable_before,
        "original_shortest_path": original_path,
        "original_distance": conflict.get("distance"),
        "cut_count": len(cut_payload),
        "cut_edges": cut_payload,
        "edge_disjoint_path_count": len(flow_paths),
        "edge_disjoint_paths": flow_paths,
        "reachable_after": reachable_after,
        "verified": verified,
        "checks": checks,
        "graph": {
            "nodes": len(result.get("nodes", []) or []),
            "edges": len(edges),
        },
    }
