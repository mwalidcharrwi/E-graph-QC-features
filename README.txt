E-Graph Quantum Circuit Feature Extraction
==========================================

Main script:
    egraph_circuit_features.py

Dependencies:
    Python 3.9+
    networkx

Single circuit, logical level:
    python egraph_circuit_features.py heisenberg_8.qasm --json features.json

Single circuit, primitive-expanded level:
    python egraph_circuit_features.py heisenberg_8.qasm --expanded --compare --json features.json

Batch directory:
    python egraph_circuit_features.py ./qasm_directory --expanded --output-dir ./egraph_features --csv ./egraph_features/feature_matrix.csv

Outputs:
  *_features.json   Full circuit features, feature vector, graph structure, and logical/expanded comparison when requested.
  *_nodes.json      Gate-level nodes with e-class IDs, parameters, depth, and dependency/interaction edges.
  *_qubits.json     Qubit-level node features.
  *_rewrites.json   Safe local equivalence rewrite opportunities.
  feature_matrix.csv Scalar fixed-length feature matrix for ML.

The e-graph representation models the evolving SSA-like state of every qubit.
The source circuit is preserved unchanged. Safe local rewrites are analyzed
conservatively without commuting gates or assuming unsafe multi-qubit identities.

Implemented safe local identities include:
  RZ(a) RZ(b) -> RZ(a+b)
  X X -> I
  H H -> I
  SX SX -> X
  SX X SX -> I
  X SX X -> SX
