"""
Prepared by Mohammad Walid Charrwi
Code for circuit feature extraction and ML embedding
"""
#!/usr/bin/env python3
"""
egraph_circuit_features_final.py
=================================
Structure-aware quantum-circuit feature extraction with a lightweight e-graph.

The script is designed for OpenQASM 2.0 circuits and supports:
  * qreg / creg declarations
  * one-line and multi-line user-defined gates
  * recursive macro expansion (e.g., ECR/RZX)
  * circuit e-graph construction using SSA-like qubit states
  * dependency DAG and qubit interaction graph construction
  * safe local equivalence-rewrite detection/saturation
  * hardware-oriented two-qubit statistics
  * parameter statistics
  * gate-local, qubit-local, and graph-level features
  * GNN-ready gate-node/dependency output
  * compact fixed-length ML feature vectors

Safe local rewrites implemented:
  1. RZ(a); RZ(b) on the same qubit -> RZ(a+b)
  2. X; X on the same qubit -> identity
  3. H; H on the same qubit -> identity
  4. SX; SX on the same qubit -> X
  5. SX; X; SX on the same qubit -> identity
  6. X; SX; X -> SX

The rewrites are intentionally conservative. No multi-qubit cancellation is
assumed unless an exact local pattern is explicitly encoded.

Examples:
    python egraph_circuit_features_final.py heisenberg_8.qasm \
        --json heisenberg_8_features_final.json

    python egraph_circuit_features_final.py heisenberg_8.qasm \
        --expanded \
        --json heisenberg_8_features_expanded_final.json

    python egraph_circuit_features_final.py heisenberg_8.qasm \
        --expanded \
        --nodes-json heisenberg_8_nodes_final.json \
        --rewrite-json heisenberg_8_rewrites.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# QASM parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDef:
    name: str
    param_names: Tuple[str, ...]
    formal_qubits: Tuple[str, ...]
    body: Tuple[str, ...]


@dataclass(frozen=True)
class Instruction:
    gate: str
    params: Tuple[str, ...]
    qubits: Tuple[int, ...]
    line_no: int
    source_gate: Optional[str] = None
    source_index: Optional[int] = None
    is_measure: bool = False


QREG_RE = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;$")
CREG_RE = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;$")


def strip_inline_comment(line: str) -> str:
    return line.split('//', 1)[0].strip()


def split_top_level_csv(text: str) -> List[str]:
    out: List[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_body_statements(body: str) -> List[str]:
    """Split a custom-gate body on semicolons."""
    return [x.strip() + ';' for x in body.split(';') if x.strip()]


def parse_gate_call(text: str) -> Tuple[str, List[str], List[str]]:
    text = text.strip().rstrip(';').strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\((.*?)\))?\s+(.+)$", text)
    if not m:
        raise ValueError(f"Cannot parse gate instruction: {text}")
    name = m.group(1)
    param_text = m.group(2)
    arg_text = m.group(3).strip()
    params = split_top_level_csv(param_text) if param_text else []
    args = split_top_level_csv(arg_text)
    return name, params, args


def parse_qubit_operand(arg: str, qreg_sizes: Dict[str, int], line_no: int) -> int:
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$", arg.strip())
    if not m:
        raise ValueError(f"Unsupported qubit operand '{arg}' on line {line_no}")
    qreg, qidx = m.group(1), int(m.group(2))
    if qreg not in qreg_sizes:
        raise ValueError(f"Unknown qreg {qreg} on line {line_no}")
    if qidx < 0 or qidx >= qreg_sizes[qreg]:
        raise ValueError(f"Qubit index {qidx} out of range for {qreg} on line {line_no}")
    return qidx


def parse_qasm(path: Path) -> Tuple[int, Dict[str, GateDef], List[Instruction]]:
    lines = path.read_text(encoding='utf-8').splitlines()
    qreg_sizes: Dict[str, int] = {}
    gate_defs: Dict[str, GateDef] = {}
    instructions: List[Instruction] = []

    i = 0
    while i < len(lines):
        raw = strip_inline_comment(lines[i])
        if not raw:
            i += 1
            continue

        qm = QREG_RE.match(raw)
        if qm:
            qreg_sizes[qm.group(1)] = int(qm.group(2))
            i += 1
            continue

        if CREG_RE.match(raw):
            i += 1
            continue

        # One-line custom gate definitions.
        gm_inline = re.match(
            r"^gate\s+([A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\(([^)]*)\))?\s+([^\{]+)\{(.*)\}\s*$", raw
        )
        if gm_inline:
            name = gm_inline.group(1)
            param_text = gm_inline.group(2) or ''
            formals = tuple(x.strip() for x in split_top_level_csv(gm_inline.group(3)) if x.strip())
            params = tuple(x.strip() for x in split_top_level_csv(param_text) if x.strip())
            body = tuple(split_body_statements(gm_inline.group(4)))
            gate_defs[name] = GateDef(name, params, formals, body)
            i += 1
            continue

        # Multi-line custom gate definition.
        gm = re.match(
            r"^gate\s+([A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\(([^)]*)\))?\s+([^\{]+)\{\s*$", raw
        )
        if gm:
            name = gm.group(1)
            param_text = gm.group(2) or ''
            formals = tuple(x.strip() for x in split_top_level_csv(gm.group(3)) if x.strip())
            params = tuple(x.strip() for x in split_top_level_csv(param_text) if x.strip())
            body_lines: List[str] = []
            i += 1
            while i < len(lines):
                body_line = strip_inline_comment(lines[i])
                if body_line == '}':
                    break
                if body_line:
                    body_lines.append(body_line)
                i += 1
            gate_defs[name] = GateDef(name, params, formals, tuple(body_lines))
            i += 1
            continue

        if raw.startswith('OPENQASM') or raw.startswith('include') or raw.startswith('barrier'):
            i += 1
            continue

        mm = re.match(
            r"^measure\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\s*->\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\s*;$", raw
        )
        if mm:
            qreg = mm.group(1)
            qidx = int(mm.group(2))
            if qreg not in qreg_sizes:
                raise ValueError(f"Unknown qreg {qreg} on line {i + 1}")
            instructions.append(Instruction('measure', (), (qidx,), i + 1, is_measure=True))
            i += 1
            continue

        if raw.endswith(';'):
            name, params, args = parse_gate_call(raw)
            qubits = tuple(parse_qubit_operand(arg, qreg_sizes, i + 1) for arg in args)
            instructions.append(Instruction(name, tuple(params), qubits, i + 1))

        i += 1

    if not qreg_sizes:
        raise ValueError("No qreg declaration found.")
    if len(qreg_sizes) > 1:
        raise ValueError(f"Multiple qregs found {qreg_sizes}; only one qreg is supported.")

    return next(iter(qreg_sizes.values())), gate_defs, instructions


# ---------------------------------------------------------------------------
# Parameter parsing and macro expansion
# ---------------------------------------------------------------------------


def safe_eval_angle(expr: str) -> Optional[float]:
    """Evaluate a tiny safe arithmetic grammar containing numeric values and pi."""
    s = expr.strip().replace('PI', 'pi')
    if not re.fullmatch(r"[0-9eE+\-*/().\s]*|pi", s):
        if 'pi' not in s or re.search(r"[^0-9eE+\-*/().\s*pi]", s):
            return None
    tokens = re.findall(r"pi|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[()+\-*/]", s)
    if ''.join(tokens).replace(' ', '') != re.sub(r'\s+', '', s):
        return None
    pos = 0

    def expr() -> float:
        nonlocal pos
        v = term()
        while pos < len(tokens) and tokens[pos] in ('+', '-'):
            op = tokens[pos]
            pos += 1
            r = term()
            v = v + r if op == '+' else v - r
        return v

    def term() -> float:
        nonlocal pos
        v = factor()
        while pos < len(tokens) and tokens[pos] in ('*', '/'):
            op = tokens[pos]
            pos += 1
            r = factor()
            if op == '*':
                v *= r
            else:
                if abs(r) < 1e-15:
                    raise ZeroDivisionError
                v /= r
        return v

    def factor() -> float:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError
        tok = tokens[pos]
        if tok == '+':
            pos += 1
            return factor()
        if tok == '-':
            pos += 1
            return -factor()
        if tok == '(':
            pos += 1
            v = expr()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError
            pos += 1
            return v
        pos += 1
        return math.pi if tok == 'pi' else float(tok)

    try:
        value = expr()
        return value if pos == len(tokens) else None
    except (ValueError, ZeroDivisionError):
        return None


def normalize_angle(x: float) -> float:
    """Normalize an angle to (-pi, pi], reducing numerical noise."""
    y = ((x + math.pi) % (2 * math.pi)) - math.pi
    if y <= -math.pi + 1e-12:
        y = math.pi
    if abs(y) < 1e-12:
        y = 0.0
    return y


def format_angle(x: float) -> str:
    return f"{normalize_angle(x):.15g}"


def substitute_params(expr: str, mapping: Dict[str, str]) -> str:
    out = expr
    for key, value in mapping.items():
        out = re.sub(rf"\b{re.escape(key)}\b", f"({value})", out)
    return out


def expand_instructions(
    instructions: Sequence[Instruction],
    gate_defs: Dict[str, GateDef],
    max_depth: int = 32,
) -> List[Instruction]:
    expanded: List[Instruction] = []

    def expand_one(inst: Instruction, depth: int, source_index: int) -> None:
        if depth > max_depth:
            raise RecursionError(f"Gate expansion exceeded max depth at {inst.gate}")
        if inst.gate not in gate_defs:
            expanded.append(
                Instruction(
                    inst.gate,
                    inst.params,
                    inst.qubits,
                    inst.line_no,
                    source_gate=inst.source_gate,
                    source_index=source_index,
                    is_measure=inst.is_measure,
                )
            )
            return

        gd = gate_defs[inst.gate]
        if len(inst.qubits) != len(gd.formal_qubits):
            raise ValueError(f"Gate {inst.gate} expects {len(gd.formal_qubits)} qubits, got {len(inst.qubits)}")
        if len(inst.params) != len(gd.param_names):
            raise ValueError(f"Gate {inst.gate} expects {len(gd.param_names)} parameters, got {len(inst.params)}")

        qmap = dict(zip(gd.formal_qubits, inst.qubits))
        pmap = dict(zip(gd.param_names, inst.params))
        for body_line in gd.body:
            name, params, args = parse_gate_call(body_line)
            body_params = tuple(substitute_params(p, pmap) for p in params)
            body_qubits: List[int] = []
            for arg in args:
                if arg not in qmap:
                    raise ValueError(f"Unknown formal qubit '{arg}' in gate {inst.gate}")
                body_qubits.append(qmap[arg])
            child = Instruction(name, body_params, tuple(body_qubits), inst.line_no, source_gate=inst.gate)
            expand_one(child, depth + 1, source_index)

    for source_index, inst in enumerate(instructions):
        expand_one(inst, 0, source_index)
    return expanded


# ---------------------------------------------------------------------------
# Lightweight circuit e-graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ENode:
    op: str
    children: Tuple[int, ...]
    attrs: Tuple[Tuple[str, str], ...] = ()


@dataclass
class EClass:
    eid: int
    nodes: List[int] = field(default_factory=list)


class UnionFind:
    def __init__(self) -> None:
        self.parent: List[int] = []
        self.rank: List[int] = []

    def new(self) -> int:
        eid = len(self.parent)
        self.parent.append(eid)
        self.rank.append(0)
        return eid

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class EGraph:
    """Minimal hash-consed e-graph suitable for circuit data extraction."""

    def __init__(self) -> None:
        self.uf = UnionFind()
        self.classes: Dict[int, EClass] = {}
        self.enodes: List[ENode] = []
        self.enode_owner: Dict[ENode, int] = {}
        self.merge_count = 0

    def new_eclass(self) -> int:
        eid = self.uf.new()
        self.classes[eid] = EClass(eid)
        return eid

    def canonical(self, eid: int) -> int:
        return self.uf.find(eid)

    def add_enode(self, enode: ENode, force_unique: bool = False) -> int:
        children = tuple(self.canonical(c) for c in enode.children)
        normalized = ENode(enode.op, children, enode.attrs)
        if not force_unique and normalized in self.enode_owner:
            return self.canonical(self.enode_owner[normalized])
        eid = self.new_eclass()
        nid = len(self.enodes)
        self.enodes.append(normalized)
        self.classes[eid].nodes.append(nid)
        self.enode_owner[normalized] = eid
        return eid

    def add_unique(self, op: str, children: Sequence[int], **attrs: object) -> int:
        attrs_tuple = tuple(sorted((k, str(v)) for k, v in attrs.items()))
        return self.add_enode(ENode(op, tuple(children), attrs_tuple), force_unique=True)

    def merge(self, a: int, b: int) -> int:
        ra, rb = self.canonical(a), self.canonical(b)
        if ra == rb:
            return ra
        root = self.uf.union(ra, rb)
        other = rb if root == ra else ra
        self.classes[root].nodes.extend(self.classes[other].nodes)
        self.merge_count += 1
        return root

    @property
    def num_eclasses(self) -> int:
        return len({self.canonical(i) for i in self.classes})

    @property
    def num_enodes(self) -> int:
        return len(self.enodes)

    def root_ids(self) -> List[int]:
        return [i for i in self.classes if self.canonical(i) == i]


@dataclass
class GateRecord:
    gid: int
    gate: str
    params: Tuple[str, ...]
    qubits: Tuple[int, ...]
    line_no: int
    input_eclasses: Tuple[int, ...]
    output_eclasses: Tuple[int, ...]
    source_gate: Optional[str]
    source_index: Optional[int]
    is_measure: bool = False


@dataclass
class EGraphCircuit:
    egraph: EGraph
    gates: List[GateRecord]
    final_qubit_eclasses: Dict[int, int]


def build_egraph(num_qubits: int, instructions: Sequence[Instruction]) -> EGraphCircuit:
    eg = EGraph()
    current = {q: eg.add_unique('wire_init', (), qubit=q) for q in range(num_qubits)}
    records: List[GateRecord] = []

    for gid, inst in enumerate(instructions):
        inputs = tuple(current[q] for q in inst.qubits)
        gate_ec = eg.add_unique(
            f"gate:{inst.gate}",
            inputs,
            gid=gid,
            params='|'.join(inst.params),
            qubits='|'.join(map(str, inst.qubits)),
            line=inst.line_no,
        )
        outputs: List[int] = []
        for slot, q in enumerate(inst.qubits):
            out_ec = eg.add_unique(
                'gate_out',
                (gate_ec,),
                gid=gid,
                slot=slot,
                qubit=q,
            )
            current[q] = out_ec
            outputs.append(out_ec)

        records.append(GateRecord(
            gid=gid,
            gate=inst.gate,
            params=inst.params,
            qubits=inst.qubits,
            line_no=inst.line_no,
            input_eclasses=inputs,
            output_eclasses=tuple(outputs),
            source_gate=inst.source_gate,
            source_index=inst.source_index,
            is_measure=inst.is_measure,
        ))

    return EGraphCircuit(eg, records, current)


# ---------------------------------------------------------------------------
# Safe local e-graph rewrites
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewriteEvent:
    rule: str
    gid: int
    qubits: Tuple[int, ...]
    before: Tuple[str, ...]
    after: Tuple[str, ...]
    estimated_gate_reduction: int
    parameter_reduction: float = 0.0


def same_single_qubit(a: GateRecord, b: GateRecord) -> bool:
    return len(a.qubits) == len(b.qubits) == 1 and a.qubits == b.qubits


def local_rewrite_analysis(ec: EGraphCircuit) -> Dict[str, object]:
    """
    Find exact local equivalence patterns and create an auxiliary rewrite e-graph.

    The original circuit e-graph is never altered. This prevents rewrite
    statistics from changing the source representation while still exposing
    equivalence opportunities that can guide circuit optimization.
    """
    gates = ec.gates
    events: List[RewriteEvent] = []
    counters = Counter()
    param_reduction = 0.0

    # Direct adjacency is required. We intentionally do not commute gates.
    i = 0
    while i < len(gates) - 1:
        a, b = gates[i], gates[i + 1]
        if same_single_qubit(a, b):
            if a.gate == 'rz' and b.gate == 'rz' and a.params and b.params:
                va, vb = safe_eval_angle(a.params[0]), safe_eval_angle(b.params[0])
                if va is not None and vb is not None:
                    combined = normalize_angle(va + vb)
                    reduction = 1
                    events.append(RewriteEvent(
                        'rz_fusion', b.gid, a.qubits,
                        ('rz', 'rz'), ('rz',), reduction,
                        parameter_reduction=abs(va) + abs(vb) - abs(combined),
                    ))
                    counters['rz_fusion'] += 1
                    param_reduction += abs(va) + abs(vb) - abs(combined)
            elif a.gate in {'x', 'h'} and b.gate == a.gate:
                rule = 'xx_cancellation' if a.gate == 'x' else 'hh_cancellation'
                events.append(RewriteEvent(
                    rule, b.gid, a.qubits,
                    (a.gate, b.gate), ('identity',), 2
                ))
                counters[rule] += 1
            elif a.gate == 'sx' and b.gate == 'sx':
                events.append(RewriteEvent(
                    'sx_sx_to_x', b.gid, a.qubits,
                    ('sx', 'sx'), ('x',), 1
                ))
                counters['sx_sx_to_x'] += 1
            i += 1
        i += 1

    # Three-gate patterns.
    for i in range(len(gates) - 2):
        a, b, c = gates[i], gates[i + 1], gates[i + 2]
        if not (same_single_qubit(a, b) and same_single_qubit(b, c)):
            continue
        if a.gate == 'sx' and b.gate == 'x' and c.gate == 'sx':
            events.append(RewriteEvent(
                'sx_x_sx_cancellation', c.gid, a.qubits,
                ('sx', 'x', 'sx'), ('identity',), 3
            ))
            counters['sx_x_sx_cancellation'] += 1
        elif a.gate == 'x' and b.gate == 'sx' and c.gate == 'x':
            # X = SX^2, and all X/SX rotations are about the same X axis:
            # X * SX * X = SX^5 = SX (up to global phase).
            events.append(RewriteEvent(
                'x_sx_x_to_sx', c.gid, a.qubits,
                ('x', 'sx', 'x'), ('sx',), 2
            ))
            counters['x_sx_x_to_sx'] += 1

    # Basic gate-opportunity density.
    non_measure = sum(1 for g in gates if not g.is_measure)
    total_reduction = sum(e.estimated_gate_reduction for e in events)
    opportunity_density = total_reduction / max(1, non_measure)

    return {
        'rewrite_rule_counts': dict(sorted(counters.items())),
        'rewrite_event_count': len(events),
        'rewrite_estimated_gate_reduction': int(total_reduction),
        'rewrite_opportunity_density': float(opportunity_density),
        'rz_fusion_count': int(counters.get('rz_fusion', 0)),
        'x_x_cancellation_count': int(counters.get('xx_cancellation', 0)),
        'h_h_cancellation_count': int(counters.get('hh_cancellation', 0)),
        'sx_sx_to_x_count': int(counters.get('sx_sx_to_x', 0)),
        'sx_x_sx_cancellation_count': int(counters.get('sx_x_sx_cancellation', 0)),
        'x_sx_x_to_sx_count': int(counters.get('x_sx_x_to_sx', 0)),
        'rewrite_parameter_abs_reduction': float(max(0.0, param_reduction)),
        'rewrite_events': [
            {
                'rule': e.rule,
                'gid': e.gid,
                'qubits': list(e.qubits),
                'before': list(e.before),
                'after': list(e.after),
                'estimated_gate_reduction': e.estimated_gate_reduction,
                'parameter_reduction': e.parameter_reduction,
            }
            for e in events
        ],
    }


# ---------------------------------------------------------------------------
# Circuit graphs and statistics
# ---------------------------------------------------------------------------


def asap_depth(num_qubits: int, gates: Sequence[GateRecord]) -> Tuple[int, List[int]]:
    qdepth = [0] * num_qubits
    depths: List[int] = []
    for g in gates:
        d = 1 + max((qdepth[q] for q in g.qubits), default=0)
        depths.append(d)
        for q in set(g.qubits):
            qdepth[q] = d
    return max(qdepth, default=0), depths


def build_interaction_graph(gates: Sequence[GateRecord]) -> nx.Graph:
    G = nx.Graph()
    for g in gates:
        G.add_nodes_from(g.qubits)
        uq = sorted(set(g.qubits))
        for i in range(len(uq)):
            for j in range(i + 1, len(uq)):
                G.add_edge(uq[i], uq[j])
    return G


def build_dependency_graph(gates: Sequence[GateRecord]) -> nx.DiGraph:
    D = nx.DiGraph()
    D.add_nodes_from(g.gid for g in gates)
    last: Dict[int, int] = {}
    for g in gates:
        preds = {last[q] for q in g.qubits if q in last}
        for p in preds:
            D.add_edge(p, g.gid)
        for q in set(g.qubits):
            last[q] = g.gid
    return D


def graph_stats(G: nx.Graph) -> Dict[str, float]:
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = [d for _, d in G.degree()]
    comps = nx.number_connected_components(G) if n else 0
    return {
        'interaction_nodes': n,
        'interaction_edges_count': m,
        'interaction_density': float(nx.density(G)) if n > 1 else 0.0,
        'interaction_components': comps,
        'interaction_degree_mean': float(sum(deg) / n) if n else 0.0,
        'interaction_degree_max': float(max(deg)) if deg else 0.0,
        'interaction_degree_std': float(
            math.sqrt(sum((x - (sum(deg) / n if n else 0.0)) ** 2 for x in deg) / n)
        ) if n else 0.0,
        'interaction_avg_shortest_path': float(nx.average_shortest_path_length(G)) if n and comps == 1 else 0.0,
    }


def dependency_stats(D: nx.DiGraph) -> Dict[str, float]:
    n, m = D.number_of_nodes(), D.number_of_edges()
    if n == 0:
        return {
            'dependency_nodes': 0, 'dependency_edge_count': 0,
            'dependency_density': 0.0, 'dependency_max_in_degree': 0,
            'dependency_max_out_degree': 0, 'dependency_longest_path': 0,
        }
    longest = 0
    if nx.is_directed_acyclic_graph(D):
        dist = {v: 0 for v in D.nodes}
        for u in nx.topological_sort(D):
            for v in D.successors(u):
                dist[v] = max(dist[v], dist[u] + 1)
        longest = max(dist.values(), default=0)
    indeg = [d for _, d in D.in_degree()]
    outdeg = [d for _, d in D.out_degree()]
    return {
        'dependency_nodes': n,
        'dependency_edge_count': m,
        'dependency_density': float(nx.density(D)),
        'dependency_max_in_degree': int(max(indeg, default=0)),
        'dependency_max_out_degree': int(max(outdeg, default=0)),
        'dependency_mean_in_degree': float(sum(indeg) / max(1, len(indeg))),
        'dependency_mean_out_degree': float(sum(outdeg) / max(1, len(outdeg))),
        'dependency_longest_path': int(longest),
    }


def layer_statistics(gates: Sequence[GateRecord], depths: Sequence[int]) -> Dict[str, float]:
    layer_counts = Counter(depths)
    sizes = list(layer_counts.values())
    if not sizes:
        return {
            'layer_count': 0, 'avg_gates_per_layer': 0.0,
            'max_gates_per_layer': 0, 'parallelism_factor': 0.0,
            'avg_1q_per_layer': 0.0, 'avg_2q_per_layer': 0.0,
            'max_2q_per_layer': 0, 'two_qubit_layer_count': 0,
            'two_qubit_depth': 0, 'two_qubit_depth_ratio': 0.0,
        }
    by_layer_1q = Counter()
    by_layer_2q = Counter()
    twoq_depths = []
    for g, d in zip(gates, depths):
        if g.is_measure:
            continue
        if len(g.qubits) == 1:
            by_layer_1q[d] += 1
        elif len(g.qubits) == 2:
            by_layer_2q[d] += 1
            twoq_depths.append(d)
    layer_n = len(sizes)
    return {
        'layer_count': layer_n,
        'avg_gates_per_layer': float(sum(sizes) / layer_n),
        'max_gates_per_layer': int(max(sizes)),
        'parallelism_factor': float(len(gates) / max(1, layer_n)),
        'avg_1q_per_layer': float(sum(by_layer_1q.values()) / layer_n),
        'avg_2q_per_layer': float(sum(by_layer_2q.values()) / layer_n),
        'max_2q_per_layer': int(max(by_layer_2q.values(), default=0)),
        'two_qubit_layer_count': len(by_layer_2q),
        'two_qubit_depth': int(max(twoq_depths, default=0)),
        'two_qubit_depth_ratio': float(max(twoq_depths, default=0) / max(1, max(depths, default=0))),
    }


def parameter_stats(gates: Sequence[GateRecord]) -> Dict[str, float]:
    vals: List[float] = []
    symbolic = 0
    parameterized = 0
    for g in gates:
        if not g.params:
            continue
        parameterized += 1
        for p in g.params:
            v = safe_eval_angle(p)
            if v is None:
                symbolic += 1
            else:
                vals.append(v)
    return {
        'parameterized_gate_count': int(parameterized),
        'numeric_parameter_count': int(len(vals)),
        'symbolic_parameter_count': int(symbolic),
        'parameter_abs_mean': float(sum(abs(v) for v in vals) / len(vals)) if vals else 0.0,
        'parameter_abs_max': float(max((abs(v) for v in vals), default=0.0)),
        'parameter_mean': float(sum(vals) / len(vals)) if vals else 0.0,
        'parameter_std': float(math.sqrt(sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / len(vals))) if vals else 0.0,
    }


def entropy_from_counts(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total, 2) for c in counts if c > 0)


def get_two_qubit_stats(gates: Sequence[GateRecord]) -> Dict[str, object]:
    pair_counts: Counter[Tuple[int, int]] = Counter()
    orientation_counts: Counter[Tuple[int, int]] = Counter()
    for g in gates:
        if len(g.qubits) != 2:
            continue
        a, b = g.qubits
        pair_counts[tuple(sorted((a, b)))] += 1
        orientation_counts[(a, b)] += 1

    vals = list(pair_counts.values())
    return {
        'two_qubit_gate_types': sorted(Counter(g.gate for g in gates if len(g.qubits) == 2).items()),
        'unique_two_qubit_pairs': len(pair_counts),
        'two_qubit_pair_reuse_mean': float(sum(vals) / len(vals)) if vals else 0.0,
        'two_qubit_pair_reuse_max': int(max(vals, default=0)),
        'two_qubit_pair_reuse_std': float(math.sqrt(sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals))) if vals else 0.0,
        'two_qubit_orientation_count': len(orientation_counts),
        'two_qubit_directional_asymmetry': float(
            sum(abs(orientation_counts[(a, b)] - orientation_counts[(b, a)]) for a, b in pair_counts)
            / max(1, 2 * sum(vals))
        ) if vals else 0.0,
    }


def qubit_statistics(num_qubits: int, gates: Sequence[GateRecord]) -> Dict[str, object]:
    per_qubit = Counter()
    per_qubit_1q = Counter()
    per_qubit_2q = Counter()
    for g in gates:
        if g.is_measure:
            continue
        for q in g.qubits:
            per_qubit[q] += 1
        if len(g.qubits) == 1:
            per_qubit_1q[g.qubits[0]] += 1
        elif len(g.qubits) == 2:
            for q in g.qubits:
                per_qubit_2q[q] += 1
    active = sorted(per_qubit)
    vals = [per_qubit[q] for q in active]
    two_vals = [per_qubit_2q[q] for q in active]
    return {
        'active_qubit_count': len(active),
        'active_qubit_ids': active,
        'inactive_qubit_count': num_qubits - len(active),
        'qubit_gate_count_mean': float(sum(vals) / len(vals)) if vals else 0.0,
        'qubit_gate_count_max': int(max(vals, default=0)),
        'qubit_gate_count_min': int(min(vals, default=0)),
        'qubit_gate_count_std': float(math.sqrt(sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / len(vals))) if vals else 0.0,
        'qubit_gate_count_entropy': float(entropy_from_counts(vals)),
        'two_qubit_gate_use_mean_per_active_qubit': float(sum(two_vals) / len(two_vals)) if two_vals else 0.0,
        'two_qubit_gate_use_max_per_active_qubit': int(max(two_vals, default=0)),
        'qubit_gate_counts': {str(q): int(per_qubit[q]) for q in active},
        'qubit_1q_gate_counts': {str(q): int(per_qubit_1q[q]) for q in active},
        'qubit_2q_gate_counts': {str(q): int(per_qubit_2q[q]) for q in active},
    }


def gate_type_features(gates: Sequence[GateRecord]) -> Dict[str, object]:
    counts = Counter(g.gate for g in gates)
    non_measure = max(1, sum(1 for g in gates if not g.is_measure))
    common = ['rz', 'sx', 'x', 'h', 'cx', 'ecr', 'rzx', 'measure']
    out: Dict[str, object] = {
        'distinct_gate_types': len(counts),
        'gate_type_counts': dict(sorted(counts.items())),
    }
    for gate in common:
        out[f'{gate}_count'] = int(counts.get(gate, 0))
        out[f'{gate}_fraction'] = float(counts.get(gate, 0) / max(1, len(gates)))
    out['non_measure_gate_count'] = sum(1 for g in gates if not g.is_measure)
    out['measurement_count'] = sum(1 for g in gates if g.is_measure)
    out['one_qubit_gate_count'] = sum(1 for g in gates if len(g.qubits) == 1 and not g.is_measure)
    out['two_qubit_gate_count'] = sum(1 for g in gates if len(g.qubits) == 2 and not g.is_measure)
    out['multi_qubit_gate_count'] = sum(1 for g in gates if len(g.qubits) > 2 and not g.is_measure)
    return out


def extract_features(
    num_qubits: int,
    ec: EGraphCircuit,
    gate_defs: Optional[Dict[str, GateDef]] = None,
    level: str = 'logical',
    source_circuit_name: str = '',
) -> Dict[str, object]:
    gates = ec.gates
    depth, depths = asap_depth(num_qubits, gates)
    G = build_interaction_graph(gates)
    D = build_dependency_graph(gates)

    active = sorted({q for g in gates for q in g.qubits if not g.is_measure})
    all_touched = sorted({q for g in gates for q in g.qubits})

    f: Dict[str, object] = {
        'schema_version': '2.0',
        'source_circuit': source_circuit_name,
        'feature_level': level,
        'declared_qubits': num_qubits,
        'active_qubits': len(active),
        'active_qubit_ids': active,
        'touched_qubits': len(all_touched),
        'total_gate_count': len(gates),
        'circuit_depth': int(depth),
        'depth_to_gate_ratio': float(depth / max(1, len(gates))),
        'depth_to_non_measure_ratio': float(depth / max(1, sum(1 for g in gates if not g.is_measure))),
        'gate_count_per_active_qubit': float(sum(1 for g in gates if not g.is_measure) / max(1, len(active))),
        'egraph_enodes': ec.egraph.num_enodes,
        'egraph_eclasses_total': len(ec.egraph.classes),
        'egraph_eclasses_live': ec.egraph.num_eclasses,
        'egraph_union_merges': ec.egraph.merge_count,
        'egraph_root_count': len(ec.egraph.root_ids()),
        'egraph_enodes_per_gate': float(ec.egraph.num_enodes / max(1, len(gates))),
        'egraph_eclasses_per_gate': float(ec.egraph.num_eclasses / max(1, len(gates))),
        'egraph_compression_ratio': float(ec.egraph.num_eclasses / max(1, ec.egraph.num_enodes)),
    }
    f.update(gate_type_features(gates))
    f.update(qubit_statistics(num_qubits, gates))
    f.update(get_two_qubit_stats(gates))
    f.update(parameter_stats(gates))

    layer = layer_statistics(gates, depths)
    f.update(layer)

    f['two_qubit_fraction'] = float(f['two_qubit_gate_count'] / max(1, f['non_measure_gate_count']))
    f['entangling_gate_fraction'] = f['two_qubit_fraction']
    f['single_to_two_qubit_ratio'] = float(f['one_qubit_gate_count'] / max(1, f['two_qubit_gate_count']))

    # ECR/rzx decomposition indicators.
    f['macro_gate_definition_count'] = len(gate_defs or {})
    f['macro_gate_names'] = sorted((gate_defs or {}).keys())
    f['native_two_qubit_gate_fraction'] = float(
        sum(1 for g in gates if len(g.qubits) == 2 and g.gate in {'ecr', 'cz', 'cx', 'rzx'})
        / max(1, f['two_qubit_gate_count'])
    )

    # Graph statistics.
    f.update(graph_stats(G))
    f.update(dependency_stats(D))

    # Safe local equivalence/rewrite features.
    rewrite = local_rewrite_analysis(ec)
    f.update({k: v for k, v in rewrite.items() if k != 'rewrite_events'})

    # Explicit pair-use histogram and edge lists are retained for graph models.
    pair_counts: Counter[Tuple[int, int]] = Counter()
    for g in gates:
        if len(g.qubits) == 2:
            pair_counts[tuple(sorted(g.qubits))] += 1
    f['interaction_edges'] = sorted([list(e) for e in G.edges()])
    f['interaction_edge_weights'] = [
        {'u': int(a), 'v': int(b), 'count': int(c)}
        for (a, b), c in sorted(pair_counts.items())
    ]
    f['dependency_edge_list'] = [[int(u), int(v)] for u, v in D.edges()]
    f['gate_depths'] = [int(x) for x in depths]

    # E-graph summary of gate representatives.
    f['egraph_operation_counts'] = dict(sorted(Counter(e.op.split(':', 1)[-1] for e in ec.egraph.enodes).items()))

    return f


# ---------------------------------------------------------------------------
# Stable ML/GNN representations
# ---------------------------------------------------------------------------


def make_node_features(ec: EGraphCircuit, gate_depths: Sequence[int]) -> List[Dict[str, object]]:
    out = []
    for g, depth in zip(ec.gates, gate_depths):
        out.append({
            'gid': g.gid,
            'gate': g.gate,
            'arity': len(g.qubits),
            'qubits': list(g.qubits),
            'line_no': g.line_no,
            'parameters': list(g.params),
            'parameter_values': [safe_eval_angle(x) for x in g.params],
            'source_gate': g.source_gate,
            'source_index': g.source_index,
            'is_measure': g.is_measure,
            'depth': int(depth),
            'input_eclasses': list(g.input_eclasses),
            'output_eclasses': list(g.output_eclasses),
        })
    return out


def make_qubit_nodes(num_qubits: int, gates: Sequence[GateRecord]) -> List[Dict[str, object]]:
    qs = qubit_statistics(num_qubits, gates)
    active_ids = set(qs['active_qubit_ids'])
    return [
        {
            'qubit': q,
            'active': q in active_ids,
            'gate_count': qs['qubit_gate_counts'].get(str(q), 0),
            'one_qubit_gate_count': qs['qubit_1q_gate_counts'].get(str(q), 0),
            'two_qubit_gate_count': qs['qubit_2q_gate_counts'].get(str(q), 0),
        }
        for q in range(num_qubits)
    ]


DEFAULT_VECTOR_KEYS = [
    # topology / size
    'declared_qubits', 'active_qubits', 'inactive_qubit_count',
    'total_gate_count', 'non_measure_gate_count', 'one_qubit_gate_count',
    'two_qubit_gate_count', 'measurement_count', 'multi_qubit_gate_count',
    'two_qubit_fraction', 'single_to_two_qubit_ratio',
    # depth / parallelism
    'circuit_depth', 'depth_to_gate_ratio', 'depth_to_non_measure_ratio',
    'layer_count', 'avg_gates_per_layer', 'max_gates_per_layer',
    'parallelism_factor', 'avg_1q_per_layer', 'avg_2q_per_layer',
    'max_2q_per_layer', 'two_qubit_layer_count', 'two_qubit_depth', 'two_qubit_depth_ratio',
    # interaction graph
    'interaction_nodes', 'interaction_edges_count', 'interaction_density',
    'interaction_components', 'interaction_degree_mean',
    'interaction_degree_max', 'interaction_degree_std',
    'interaction_avg_shortest_path', 'unique_two_qubit_pairs',
    'two_qubit_pair_reuse_mean', 'two_qubit_pair_reuse_max',
    'two_qubit_pair_reuse_std', 'two_qubit_orientation_count',
    'two_qubit_directional_asymmetry',
    # qubit load
    'qubit_gate_count_mean', 'qubit_gate_count_max', 'qubit_gate_count_min',
    'qubit_gate_count_std', 'qubit_gate_count_entropy',
    'two_qubit_gate_use_mean_per_active_qubit',
    'two_qubit_gate_use_max_per_active_qubit',
    # parameters
    'parameterized_gate_count', 'numeric_parameter_count',
    'symbolic_parameter_count', 'parameter_abs_mean', 'parameter_abs_max',
    'parameter_mean', 'parameter_std',
    # dependency graph
    'dependency_nodes', 'dependency_edge_count', 'dependency_density',
    'dependency_max_in_degree', 'dependency_max_out_degree',
    'dependency_mean_in_degree', 'dependency_mean_out_degree',
    'dependency_longest_path',
    # e-graph
    'egraph_enodes', 'egraph_eclasses_total', 'egraph_eclasses_live',
    'egraph_union_merges', 'egraph_root_count', 'egraph_enodes_per_gate',
    'egraph_eclasses_per_gate', 'egraph_compression_ratio',
    # rewrite opportunities
    'rewrite_event_count', 'rewrite_estimated_gate_reduction',
    'rewrite_opportunity_density', 'rz_fusion_count',
    'x_x_cancellation_count', 'h_h_cancellation_count',
    'sx_sx_to_x_count', 'sx_x_sx_cancellation_count',
    'rewrite_parameter_abs_reduction',
    # native gate composition
    'rz_count', 'sx_count', 'x_count', 'h_count', 'cx_count',
    'ecr_count', 'rzx_count', 'native_two_qubit_gate_fraction',
]


def build_feature_vector(features: Dict[str, object], keys: Sequence[str] = DEFAULT_VECTOR_KEYS) -> Tuple[List[str], List[float]]:
    vals = []
    for k in keys:
        v = features.get(k, 0.0)
        if isinstance(v, bool):
            vals.append(float(v))
        elif isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            vals.append(0.0)
    return list(keys), vals


def build_feature_table_logical_vs_expanded(
    logical: Dict[str, object],
    expanded: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    out = {
        'logical': {
            'total_gate_count': logical['total_gate_count'],
            'one_qubit_gate_count': logical['one_qubit_gate_count'],
            'two_qubit_gate_count': logical['two_qubit_gate_count'],
            'circuit_depth': logical['circuit_depth'],
            'egraph_enodes': logical['egraph_enodes'],
            'rewrite_estimated_gate_reduction': logical['rewrite_estimated_gate_reduction'],
        }
    }
    if expanded:
        out['expanded'] = {
            'total_gate_count': expanded['total_gate_count'],
            'one_qubit_gate_count': expanded['one_qubit_gate_count'],
            'two_qubit_gate_count': expanded['two_qubit_gate_count'],
            'circuit_depth': expanded['circuit_depth'],
            'egraph_enodes': expanded['egraph_enodes'],
            'rewrite_estimated_gate_reduction': expanded['rewrite_estimated_gate_reduction'],
        }
        out['expansion_multipliers'] = {
            'gate_count': expanded['total_gate_count'] / max(1, logical['total_gate_count']),
            'depth': expanded['circuit_depth'] / max(1, logical['circuit_depth']),
            'two_qubit_gate_count': expanded['two_qubit_gate_count'] / max(1, logical['two_qubit_gate_count']),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting / main
# ---------------------------------------------------------------------------


def pretty_summary(name: str, f: Dict[str, object]) -> str:
    return '\n'.join([
        f"Circuit: {name}",
        f"Feature level: {f['feature_level']}",
        f"Declared / active qubits: {f['declared_qubits']} / {f['active_qubits']}",
        f"Total / non-measure gates: {f['total_gate_count']} / {f['non_measure_gate_count']}",
        f"1Q / 2Q / multi-Q: {f['one_qubit_gate_count']} / {f['two_qubit_gate_count']} / {f['multi_qubit_gate_count']}",
        f"Measurements: {f['measurement_count']}",
        f"Depth / layers: {f['circuit_depth']} / {f['layer_count']}",
        f"Unique 2Q pairs / max reuse: {f['unique_two_qubit_pairs']} / {f['two_qubit_pair_reuse_max']}",
        f"Interaction degree mean / max: {f['interaction_degree_mean']:.3f} / {f['interaction_degree_max']:.0f}",
        f"Dependency edges / longest path: {f['dependency_edge_count']} / {f['dependency_longest_path']}",
        f"E-graph e-nodes / live e-classes: {f['egraph_enodes']} / {f['egraph_eclasses_live']}",
        f"Rewrite opportunities / estimated reduction: {f['rewrite_event_count']} / {f['rewrite_estimated_gate_reduction']}",
        f"Gate counts: {f['gate_type_counts']}",
    ])


def write_json(path: Optional[Path], payload: object) -> None:
    if path:
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"Wrote: {path}")


def process_one_circuit(
    qasm_path: Path,
    output_dir: Optional[Path] = None,
    expanded: bool = False,
    write_nodes: bool = True,
    write_qubits: bool = True,
    write_rewrites: bool = True,
) -> Dict[str, object]:
    """Process one QASM file and optionally emit all derived artifacts."""
    num_qubits, gate_defs, logical = parse_qasm(qasm_path)
    logical_ec = build_egraph(num_qubits, logical)
    logical_features = extract_features(
        num_qubits, logical_ec, gate_defs, level='logical', source_circuit_name=qasm_path.name
    )
    logical_keys, logical_vec = build_feature_vector(logical_features)
    logical_features['feature_vector_keys'] = logical_keys
    logical_features['feature_vector'] = logical_vec

    expanded_features: Optional[Dict[str, object]] = None
    selected_ec = logical_ec
    selected_features = logical_features
    selected_level = 'logical'

    if expanded:
        primitive = expand_instructions(logical, gate_defs)
        expanded_ec = build_egraph(num_qubits, primitive)
        expanded_features = extract_features(
            num_qubits, expanded_ec, gate_defs, level='expanded', source_circuit_name=qasm_path.name
        )
        expanded_keys, expanded_vec = build_feature_vector(expanded_features)
        expanded_features['feature_vector_keys'] = expanded_keys
        expanded_features['feature_vector'] = expanded_vec
        selected_ec = expanded_ec
        selected_features = expanded_features
        selected_level = 'expanded'

    payload = dict(selected_features)
    payload.pop('rewrite_events', None)
    payload['logical_features'] = logical_features
    if expanded_features is not None:
        payload['expanded_features'] = expanded_features
        payload['logical_vs_expanded'] = build_feature_table_logical_vs_expanded(logical_features, expanded_features)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = qasm_path.stem
        (output_dir / f'{stem}_features.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        rewrite = local_rewrite_analysis(selected_ec)
        if write_rewrites:
            (output_dir / f'{stem}_rewrites.json').write_text(json.dumps({
                'schema_version': '2.0', 'circuit': qasm_path.name,
                'feature_level': selected_level, **rewrite
            }, indent=2), encoding='utf-8')
        if write_nodes:
            _, depths = asap_depth(num_qubits, selected_ec.gates)
            (output_dir / f'{stem}_nodes.json').write_text(json.dumps({
                'schema_version': '2.0', 'circuit': qasm_path.name,
                'feature_level': selected_level,
                'nodes': make_node_features(selected_ec, depths),
                'dependency_edges': selected_features['dependency_edge_list'],
                'interaction_edges': selected_features['interaction_edge_weights'],
            }, indent=2), encoding='utf-8')
        if write_qubits:
            (output_dir / f'{stem}_qubits.json').write_text(json.dumps({
                'schema_version': '2.0', 'circuit': qasm_path.name,
                'feature_level': selected_level,
                'nodes': make_qubit_nodes(num_qubits, selected_ec.gates),
            }, indent=2), encoding='utf-8')

    return payload


def scalar_row(features: Dict[str, object]) -> Dict[str, object]:
    """Return scalar ML features only, suitable for a CSV data set."""
    row = {'circuit': features.get('source_circuit', ''), 'feature_level': features.get('feature_level', '')}
    for k in DEFAULT_VECTOR_KEYS:
        v = features.get(k, 0.0)
        row[k] = float(v) if isinstance(v, (int, float, bool)) else 0.0
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='Extract E-graph circuit features from one or more OpenQASM 2.0 files.')
    ap.add_argument('qasm', type=Path, help='Input .qasm file or directory containing .qasm files.')
    ap.add_argument('--json', type=Path, default=None, help='Write main features to JSON (single-file mode only).')
    ap.add_argument('--nodes-json', type=Path, default=None, help='Write GNN-ready gate nodes (single-file mode only).')
    ap.add_argument('--qubits-json', type=Path, default=None, help='Write qubit nodes (single-file mode only).')
    ap.add_argument('--rewrite-json', type=Path, default=None, help='Write rewrite events (single-file mode only).')
    ap.add_argument('--output-dir', type=Path, default=None, help='Directory for per-circuit JSON artifacts; also enables directory/batch output.')
    ap.add_argument('--csv', type=Path, default=None, help='Write scalar feature matrix CSV. Recommended for multiple circuits.')
    ap.add_argument('--expanded', action='store_true', help='Recursively expand user-defined gates before selected feature extraction.')
    ap.add_argument('--compare', action='store_true', help='Include logical-vs-expanded comparison when --expanded is used.')
    ap.add_argument('--no-nodes', action='store_true', help='Do not write GNN gate-node JSON in batch mode.')
    ap.add_argument('--no-qubits', action='store_true', help='Do not write qubit-node JSON in batch mode.')
    ap.add_argument('--no-rewrites', action='store_true', help='Do not write rewrite-event JSON in batch mode.')
    args = ap.parse_args()

    if args.qasm.is_dir():
        files = sorted(args.qasm.glob('*.qasm'))
        if not files:
            raise SystemExit(f'No .qasm files found in {args.qasm}')
        outdir = args.output_dir or (args.qasm / 'egraph_features')
        rows: List[Dict[str, object]] = []
        failures = []
        for qasm_file in files:
            try:
                payload = process_one_circuit(
                    qasm_file, outdir, args.expanded,
                    write_nodes=not args.no_nodes,
                    write_qubits=not args.no_qubits,
                    write_rewrites=not args.no_rewrites,
                )
                selected = payload.get('expanded_features') if args.expanded else payload.get('logical_features')
                rows.append(scalar_row(selected))
                print(f'Processed: {qasm_file.name} | gates={selected["total_gate_count"]} | depth={selected["circuit_depth"]} | 2Q={selected["two_qubit_gate_count"]}')
            except Exception as exc:
                failures.append({'circuit': qasm_file.name, 'error': str(exc)})
                print(f'FAILED: {qasm_file.name}: {exc}')

        import csv
        csv_path = args.csv or (outdir / 'circuit_feature_matrix.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['circuit', 'feature_level'] + DEFAULT_VECTOR_KEYS
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote: {csv_path}')
        if failures:
            fail_path = outdir / 'failed_circuits.json'
            fail_path.write_text(json.dumps(failures, indent=2), encoding='utf-8')
            print(f'Failures: {len(failures)}; details: {fail_path}')
        else:
            print(f'Processed successfully: {len(rows)} circuit(s)')
        return

    # Single-file mode.
    payload = process_one_circuit(
        args.qasm, args.output_dir, args.expanded,
        write_nodes=not args.no_nodes, write_qubits=not args.no_qubits,
        write_rewrites=not args.no_rewrites,
    )
    selected = payload.get('expanded_features') if args.expanded else payload.get('logical_features')
    if args.compare and args.expanded:
        payload['logical_vs_expanded'] = build_feature_table_logical_vs_expanded(
            payload['logical_features'], payload['expanded_features']
        )

    print(pretty_summary(args.qasm.name, selected))
    print('\nFeature vector length:', len(selected['feature_vector']))
    print('Feature vector:')
    print(json.dumps(selected['feature_vector']))

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'Wrote: {args.json}')
    if args.nodes_json:
        qnum, qdefs, qlogical = parse_qasm(args.qasm)
        qinst = expand_instructions(qlogical, qdefs) if args.expanded else qlogical
        selected_ec = build_egraph(qnum, qinst)
        args.nodes_json.write_text(json.dumps({
            'schema_version': '2.0', 'circuit': args.qasm.name,
            'feature_level': selected['feature_level'],
            'nodes': make_node_features(selected_ec, asap_depth(qnum, selected_ec.gates)[1]),
            'dependency_edges': selected['dependency_edge_list'],
            'interaction_edges': selected['interaction_edge_weights'],
        }, indent=2), encoding='utf-8')
        print(f'Wrote: {args.nodes_json}')
    if args.qubits_json:
        qnum, qdefs, qlogical = parse_qasm(args.qasm)
        qinst = expand_instructions(qlogical, qdefs) if args.expanded else qlogical
        qec = build_egraph(qnum, qinst)
        args.qubits_json.write_text(json.dumps({
            'schema_version': '2.0', 'circuit': args.qasm.name,
            'feature_level': selected['feature_level'],
            'nodes': make_qubit_nodes(qnum, qec.gates),
        }, indent=2), encoding='utf-8')
        print(f'Wrote: {args.qubits_json}')
    if args.rewrite_json:
        qnum, qdefs, qlogical = parse_qasm(args.qasm)
        qinst = expand_instructions(qlogical, qdefs) if args.expanded else qlogical
        qec = build_egraph(qnum, qinst)
        args.rewrite_json.write_text(json.dumps({
            'schema_version': '2.0', 'circuit': args.qasm.name,
            'feature_level': selected['feature_level'],
            **local_rewrite_analysis(qec),
        }, indent=2), encoding='utf-8')
        print(f'Wrote: {args.rewrite_json}')
    if args.csv:
        import csv
        with args.csv.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=['circuit', 'feature_level'] + DEFAULT_VECTOR_KEYS)
            writer.writeheader(); writer.writerow(scalar_row(selected))
        print(f'Wrote: {args.csv}')


if __name__ == '__main__':
    main()
