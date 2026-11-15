"""
GHT-QL Parser  —  v1.0
Graph Hypothesis Testing Query Language

Parses a .ghtql file (or string) into a patterns dict compatible with
the patterns.json format.

CLI usage:
    python parser.py patterns.ghtql              # parse and print to stdout
    python parser.py patterns.ghtql -o out.json  # write result to JSON file

Python API:
    from ghtql.parser import parse_file, parse_string
    result = parse_file("patterns.ghtql")

─── TEST clause ──────────────────────────────────────────────────────────────

Every TEST clause MUST include a comparison operator and a right-hand side.
Three forms are supported:

  Form 1 — compare aggregate to the global population mean (most common):
      TEST mean(n1.n_citation) = mu
      TEST mean(|n1.n_citation - n3.n_citation|) = mu

  Form 2 — compare aggregate to a scalar constant:
      TEST mean(n1.age) > 25
      TEST mean(|n0.stars - n2.stars|) >= 0.5

  Form 3 — compare two aggregates (two pattern instances):
      TEST mean(n1.age) > mean(n3.age)
      TEST mean(|n0.stars - n2.stars|) < mean(n1.rating)

  Edge attribute target (extracts an attribute from a pattern edge):
      TEST mean(n0->n1.rating) = mu
      TEST mean(n0->n1.stars) > 3.5

Supported comparison operators: <  <=  =  >=  >  !=

The keyword `mu` represents the global (population) mean of the target
attribute across the entire dataset — it is the null-hypothesis reference
value used in one-sample t-tests.

─── JSON test field ──────────────────────────────────────────────────────────

  Global mean comparison (form 1):
      "test": {"type": "one-sample t-test", "agg": "mean",
               "op": "=", "rhs": "mu"}

  Scalar comparison (form 2):
      "test": {"type": "one-sample t-test", "agg": "mean",
               "op": ">", "rhs": 25}

  Aggregate-vs-aggregate comparison (form 3):
      "test": {"type": "two-sample t-test", "agg": "mean",
               "op": ">", "rhs": <target-dict>}
"""

from __future__ import annotations

import re
import json
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GHTQLError(Exception):
    def __init__(self, msg: str, line_no: int | None = None):
        loc = f" (line {line_no})" if line_no is not None else ""
        super().__init__(f"[GHTQLError{loc}] {msg}")

class UndeclaredVariableError(GHTQLError): pass
class DuplicateVariableError(GHTQLError):  pass
class UnknownDatasetError(GHTQLError):     pass
class MalformedTestError(GHTQLError):      pass
class MissingClauseError(GHTQLError):      pass


# ---------------------------------------------------------------------------
# Lexer (Tokenizer)
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str   # see TOKEN_PATTERNS below
    value: Any
    line: int

# "mu" is promoted to a dedicated keyword so it cannot be mistaken for a
# node variable.
KEYWORDS = {"MATCH", "ON", "TEST", "mean", "mu"}
SUPPORTED_DATASETS = {"Citation", "MovieLens", "Yelp"}

# Comparison operators — listed before MINUS / IDENT / MISMATCH so they
# are matched as atomic tokens.  Order matters: <= / >= / != before < / > / =.
# Rules listed in priority order (high to low).
TOKEN_PATTERNS: list[tuple[str, str]] = [
    ("ARROW_R", r"->"),
    ("ARROW_L", r"<-"),
    ("CMP",     r"<=|>=|!=|<|>|="),   # comparison operators
    ("STRING",  r'"[^"]*"'),
    ("FLOAT",   r"\d+\.\d+"),
    ("INT",     r"\d+"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("LBRACE",  r"\{"),
    ("RBRACE",  r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("PIPE",    r"\|"),
    ("DOT",     r"\."),
    ("COLON",   r":"),
    ("COMMA",   r","),
    ("MINUS",   r"-"),                         # standalone minus, used in diff expressions
    # Identifier: hyphens are allowed only BETWEEN alphanumerics (e.g. "Sci-Fi",
    # "popularity-high"), never as a trailing char — otherwise "n0->n1" would
    # tokenize as IDENT("n0-") CMP(">") IDENT("n1") instead of IDENT("n0")
    # ARROW_R IDENT("n1").
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*"),
    ("SKIP",    r"[ \t]+"),
    ("NEWLINE", r"\n"),
    ("MISMATCH",r"."),
]
_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_PATTERNS)
)

VALID_OPS = {"<", "<=", "=", ">=", ">", "!="}


def tokenize(src: str) -> list[Token]:
    """Tokenize a GHT-QL source string into a list of Tokens."""
    tokens: list[Token] = []
    line_no = 1
    for m in _MASTER_RE.finditer(src):
        kind  = m.lastgroup
        raw   = m.group()
        if kind == "SKIP":
            continue
        if kind == "NEWLINE":
            line_no += 1
            tokens.append(Token("NEWLINE", "\n", line_no))
            continue
        if kind == "MISMATCH":
            raise GHTQLError(f"Unexpected character: {raw!r}", line_no)
        # Upgrade identifiers that match keywords
        if kind == "IDENT" and raw in KEYWORDS:
            kind = "KEYWORD"
        # Type conversion
        if kind == "STRING":
            raw = raw[1:-1]       # strip surrounding quotes
        elif kind == "INT":
            raw = int(raw)
        elif kind == "FLOAT":
            raw = float(raw)
        tokens.append(Token(kind, raw, line_no))
    tokens.append(Token("EOF", None, line_no))
    return tokens


# ---------------------------------------------------------------------------
# Pre-processing: extract -- id comment lines
# ---------------------------------------------------------------------------

def _preprocess(src: str) -> tuple[str, list[str]]:
    """
    Strip `-- <id>` comment lines from the source and collect their IDs in
    order.  Returns (source_without_comments, [id, ...]).
    """
    lines   = src.split("\n")
    ids: list[str] = []
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--"):
            ids.append(stripped[2:].strip())
            cleaned.append("")   # preserve line numbers
        else:
            cleaned.append(line)
    return "\n".join(cleaned), ids


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos    = 0

    # ── Utilities ─────────────────────────────────────────────────────────

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        if t.kind != "EOF":
            self.pos += 1
        return t

    def skip_newlines(self) -> None:
        while self.cur.kind == "NEWLINE":
            self.advance()

    def expect(self, kind: str, value: Any = None) -> Token:
        t = self.cur
        if t.kind != kind:
            raise GHTQLError(
                f"Expected token {kind!r}, got {t.kind!r} ({t.value!r})", t.line
            )
        if value is not None and t.value != value:
            raise GHTQLError(
                f"Expected value {value!r}, got {t.value!r}", t.line
            )
        return self.advance()

    def is_kw(self, kw: str) -> bool:
        return self.cur.kind == "KEYWORD" and self.cur.value == kw

    # ── Top level: program ────────────────────────────────────────────────

    def parse_program(self) -> list[dict]:
        """Parse the entire program; return a list of entries in source order."""
        entries: list[dict] = []
        self.skip_newlines()
        while self.cur.kind != "EOF":
            entries.append(self._parse_statement())
            self.skip_newlines()
        return entries

    # ── Statement ─────────────────────────────────────────────────────────

    def _parse_statement(self) -> dict:
        """Parse a single MATCH … ON … TEST … statement."""
        # Reset edge-variable map for this statement
        self._edge_var_map: dict[str, dict] = {}

        # MATCH
        if not self.is_kw("MATCH"):
            raise MissingClauseError("Expected MATCH clause", self.cur.line)
        self.advance()
        nodes, edges = self._parse_pattern_list()
        self.skip_newlines()

        # ON
        if not self.is_kw("ON"):
            raise MissingClauseError("Expected ON clause", self.cur.line)
        self.advance()
        dataset = self._parse_dataset()
        self.skip_newlines()

        # TEST
        if not self.is_kw("TEST"):
            raise MissingClauseError("Expected TEST clause", self.cur.line)
        self.advance()
        target, agg, op, rhs = self._parse_test()
        self.skip_newlines()

        # Semantic validation
        declared = {n["id"] for n in nodes}
        self._validate_target(target, declared, edges)
        if isinstance(rhs, dict):
            # rhs is a target dict (aggregate comparison) — validate its vars too
            self._validate_target(rhs, declared, edges)

        # Determine test type
        if isinstance(rhs, dict):
            test_type = "two-sample t-test"
        elif "edge_diff" in target:
            # signed difference of two positional edges → paired t-test
            test_type = "paired t-test"
        else:
            test_type = "one-sample t-test"

        test: dict[str, Any] = {
            "type": test_type,
            "agg":  agg,
            "op":   op,
            "rhs":  rhs,
        }

        return {
            "dataset": dataset,
            "type": _infer_type(nodes, edges),
            "pattern": {"nodes": nodes, "edges": edges},
            "target": target,
            "test": test,
        }

    # ── MATCH: graph pattern list ──────────────────────────────────────────

    def _parse_pattern_list(self) -> tuple[list[dict], list[dict]]:
        """Parse comma-separated pattern clauses; return (nodes, edges)."""
        node_map: dict[str, dict] = {}
        edges: list[dict] = []

        self._parse_pattern_clause(node_map, edges)
        while True:
            self.skip_newlines()
            if self.cur.kind == "COMMA":
                self.advance()
                self.skip_newlines()
                self._parse_pattern_clause(node_map, edges)
            else:
                break

        # Sort nodes by trailing numeric suffix to match reference JSON ordering
        nodes = sorted(node_map.values(), key=_node_sort_key)
        return nodes, edges

    def _parse_pattern_clause(self, node_map: dict, edges: list) -> None:
        """Parse a single path chain: node (edge node)*

        Each edge may optionally carry a decoration:  -[var?:label? props?]->
        Examples:
            (n0)->(n1)                     — plain edge
            (n0)-[e]->(n1)                 — named edge variable
            (n0)-[:belong_to]->(n1)        — edge label/type
            (n0)-[e:belong_to]->(n1)       — variable + label
            (n0)-[e:belong_to {w:5}]->(n1) — variable + label + attribute filter
        """
        left = self._parse_node(node_map)
        while True:
            edge_dec: dict = {}
            # Edge decoration: -[var?:label? props?]->  or  <-[...-
            # The leading '-' is tokenised as MINUS, then LBRACKET, then ARROW.
            if self.cur.kind == "MINUS":
                self.advance()  # consume MINUS
                if self.cur.kind == "LBRACKET":
                    edge_dec = self._parse_edge_decoration(node_map)
                if self.cur.kind == "ARROW_R":
                    self.advance()
                    direction = "ARROW_R"
                else:
                    raise GHTQLError(
                        f"Expected '->' after edge decoration, got {self.cur.value!r}",
                        self.cur.line,
                    )
            elif self.cur.kind == "ARROW_R" or self.cur.kind == "ARROW_L":
                direction = self.advance().kind
            else:
                break

            right = self._parse_node(node_map)
            if direction == "ARROW_R":
                edge = {"from": left, "to": right, **edge_dec}
            else:
                edge = {"from": right, "to": left, **edge_dec}
            # Register edge variable for TEST clause resolution
            if "var" in edge_dec:
                self._edge_var_map[edge_dec["var"]] = edge
            edges.append(edge)
            left = right

    def _parse_edge_decoration(self, node_map: dict) -> dict:
        """Parse [var?:label? props?] between the dashes of an edge arrow."""
        self.expect("LBRACKET")
        result: dict = {}
        # Optional edge variable name
        if self.cur.kind == "IDENT":
            var_name = self.advance().value
            if var_name in node_map:
                raise DuplicateVariableError(
                    f"Edge variable {var_name!r} conflicts with a node variable "
                    f"of the same name",
                    self.cur.line,
                )
            if var_name in self._edge_var_map:
                raise DuplicateVariableError(
                    f"Edge variable {var_name!r} declared twice",
                    self.cur.line,
                )
            result["var"] = var_name
        # Optional edge label / relationship type
        if self.cur.kind == "COLON":
            self.advance()
            result["label"] = self.expect("IDENT").value
        # Optional edge attribute constraints
        if self.cur.kind == "LBRACE":
            result["attribute"] = self._parse_props()
        self.expect("RBRACKET")
        return result

    def _parse_node(self, node_map: dict) -> str:
        """
        Parse (var:label {props}) or (var).
        Returns the variable name; registers new declarations in node_map.
        """
        self.expect("LPAREN")
        var = self.expect("IDENT").value

        if self.cur.kind == "COLON":
            self.advance()
            label = self.expect("IDENT").value
            props: dict = {}
            if self.cur.kind == "LBRACE":
                props = self._parse_props()
            entry = {"id": var, "label": label, "attribute": props}
            if var in node_map:
                ex = node_map[var]
                if ex["label"] != label or ex["attribute"] != props:
                    raise DuplicateVariableError(
                        f"Variable {var!r} declared twice with conflicting definition",
                        self.cur.line,
                    )
            else:
                node_map[var] = entry
        else:
            if var not in node_map:
                raise UndeclaredVariableError(
                    f"Variable {var!r} used before declaration", self.cur.line
                )

        self.expect("RPAREN")
        return var

    def _parse_props(self) -> dict:
        """Parse {key:value, ...}; return a property dict.

        Keys may be either bare IDENT (e.g. ``state``) or quoted STRING
        (e.g. ``"Fast Food"``) — the latter is required for attribute names
        containing spaces or other non-identifier characters, which are common
        in the Yelp categories schema.
        """
        self.expect("LBRACE")
        props: dict = {}
        while self.cur.kind != "RBRACE":
            kt = self.cur
            if kt.kind == "IDENT":
                key = kt.value
            elif kt.kind == "STRING":
                key = kt.value
            else:
                raise GHTQLError(
                    f"Expected property key (identifier or string), got {kt.value!r}",
                    kt.line,
                )
            self.advance()
            self.expect("COLON")
            t = self.cur
            if t.kind in ("STRING", "INT", "FLOAT"):
                props[key] = t.value
                self.advance()
            else:
                raise GHTQLError(
                    f"Expected property value (string/int/float), got {t.value!r}", t.line
                )
            if self.cur.kind == "COMMA":
                self.advance()
        self.expect("RBRACE")
        return props

    # ── ON: dataset ────────────────────────────────────────────────────────

    def _parse_dataset(self) -> str:
        t = self.cur
        if t.kind not in ("IDENT", "KEYWORD"):
            raise UnknownDatasetError(f"Expected dataset name, got {t.value!r}", t.line)
        if t.value not in SUPPORTED_DATASETS:
            raise UnknownDatasetError(
                f"Unknown dataset {t.value!r}. Supported: {sorted(SUPPORTED_DATASETS)}",
                t.line,
            )
        self.advance()
        return t.value

    # ── TEST: test expression ─────────────────────────────────────────────

    def _parse_test(self) -> tuple[dict, str, str, Any]:
        """
        Parse the full TEST expression.

        Returns (lhs_target, agg, op, rhs) where:
          - rhs == "mu"     → compare against the global population mean
          - rhs is int/float → scalar threshold
          - rhs is dict      → rhs aggregate target (two-sample comparison)

        A comparison operator is mandatory; omitting it is a parse error.
        """
        lhs, agg = self._parse_agg_expr()

        # Comparison operator is required
        if self.cur.kind != "CMP":
            raise MalformedTestError(
                f"TEST clause requires a comparison operator "
                f"(e.g. '= mu', '> 0', '> mean(n3.age)'). "
                f"Got {self.cur.kind!r} ({self.cur.value!r}) instead.",
                self.cur.line,
            )
        op = self.advance().value   # consume the CMP token

        # RHS: mu | scalar literal | another agg expression
        t = self.cur
        if self.is_kw("mu"):
            # global population mean — the default null-hypothesis reference
            self.advance()
            return lhs, agg, op, "mu"
        elif t.kind in ("INT", "FLOAT"):
            rhs: Any = t.value
            self.advance()
            return lhs, agg, op, rhs
        elif t.kind == "MINUS":
            # negative number: -25, -0.5 …
            self.advance()
            t2 = self.cur
            if t2.kind in ("FLOAT", "INT"):
                rhs = -t2.value
            else:
                raise MalformedTestError(
                    f"Expected numeric literal after '-', got {t2.value!r}", t2.line
                )
            self.advance()
            return lhs, agg, op, rhs
        elif t.kind == "STRING":
            rhs = t.value
            self.advance()
            return lhs, agg, op, rhs
        elif self.is_kw("mean"):
            rhs_target, _rhs_agg = self._parse_agg_expr()
            return lhs, agg, op, rhs_target
        else:
            raise MalformedTestError(
                f"Expected 'mu', a scalar value, or an aggregation expression "
                f"after '{op}', got {t.value!r}",
                t.line,
            )

    def _peek(self, offset: int = 1) -> Token:
        """Look ahead `offset` tokens without consuming."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF

    def _parse_agg_expr(self) -> tuple[dict, str]:
        """
        Parse  mean(target)  and return (target_dict, agg_name).
        target is one of:
          - node target      : var.attr             (var is a node variable)
          - edge var target  : var.attr             (var is an edge variable)
          - edge target      : var->var.attr        (inline edge reference)
          - diff target      : |var.attr - var.attr|
        """
        if not self.is_kw("mean"):
            raise MalformedTestError(
                f"Expected aggregation function (e.g. 'mean'), got {self.cur.value!r}",
                self.cur.line,
            )
        agg = self.advance().value
        self.expect("LPAREN")
        if self.cur.kind == "PIPE":
            target = self._parse_diff_target()
        elif self.cur.kind == "IDENT" and self.cur.value in self._edge_var_map:
            # Edge variable reference: e.attr  (e declared via -[e:...]-> in MATCH)
            target = self._parse_edge_var_target()
        elif self.cur.kind == "IDENT" and self._peek(1).kind == "ARROW_R":
            target = self._parse_edge_target()
            # edge_diff target: mean(a->b.attr - c->d.attr) — signed difference
            # between two positional edges of the same path pattern.
            if self.cur.kind == "MINUS":
                t = self.cur
                self.advance()
                if not (self.cur.kind == "IDENT" and self._peek(1).kind == "ARROW_R"):
                    raise MalformedTestError(
                        f"Expected an edge reference (var->var.attr) after '-' "
                        f"in edge-diff expression, got {self.cur.value!r}",
                        t.line,
                    )
                target_b = self._parse_edge_target()
                if target_b["edge"]["attribute"] != target["edge"]["attribute"]:
                    raise MalformedTestError(
                        f"Both edges of an edge-diff must reference the same "
                        f"attribute; got {target['edge']['attribute']!r} vs "
                        f"{target_b['edge']['attribute']!r}",
                        t.line,
                    )
                ea, eb = target["edge"], target_b["edge"]
                if (ea["from"], ea["to"]) == (eb["from"], eb["to"]):
                    raise MalformedTestError(
                        "Both edges of an edge-diff must be distinct pattern edges.",
                        t.line,
                    )
                target = {
                    "edge_diff": {
                        "edges": [[ea["from"], ea["to"]],
                                  [eb["from"], eb["to"]]],
                        "attribute": ea["attribute"],
                    }
                }
        else:
            target = self._parse_node_target()
        self.expect("RPAREN")
        return target, agg

    def _parse_node_target(self) -> dict:
        """Parse var.attr; return {"node": {"id": var, "attribute": attr}}."""
        var  = self.expect("IDENT").value
        self.expect("DOT")
        attr = self.expect("IDENT").value
        return {"node": {"id": var, "attribute": attr}}

    def _parse_edge_var_target(self) -> dict:
        """
        Parse e.attr where *e* is a declared edge variable, and resolve it
        to the standard edge target {"edge": {"from": ..., "to": ..., "attribute": ...}}
        at parse time.  This keeps the JSON output format identical to the
        inline ``n0->n1.attr`` form.
        """
        var  = self.expect("IDENT").value
        self.expect("DOT")
        attr = self.expect("IDENT").value
        edge = self._edge_var_map[var]
        return {"edge": {"from": edge["from"], "to": edge["to"], "attribute": attr}}

    def _parse_edge_target(self) -> dict:
        """
        Parse var->var.attr; return {"edge": {"from": ..., "to": ..., "attribute": ...}}.

        The from/to pair must match a directed edge declared in the MATCH
        pattern (validated later in _validate_target).
        """
        from_var = self.expect("IDENT").value
        self.expect("ARROW_R")
        to_var   = self.expect("IDENT").value
        self.expect("DOT")
        attr     = self.expect("IDENT").value
        return {"edge": {"from": from_var, "to": to_var, "attribute": attr}}

    def _parse_diff_target(self) -> dict:
        """Parse |varA.attr - varB.attr|; return {"node_diff": {...}}."""
        self.expect("PIPE")
        var_a  = self.expect("IDENT").value
        self.expect("DOT")
        attr_a = self.expect("IDENT").value
        # Expect a standalone MINUS token (lexer gives MINUS higher priority than IDENT)
        t = self.cur
        if t.kind != "MINUS":
            raise MalformedTestError(
                f"Expected '-' in diff expression, got {t.value!r}", t.line
            )
        self.advance()
        var_b  = self.expect("IDENT").value
        self.expect("DOT")
        attr_b = self.expect("IDENT").value
        self.expect("PIPE")
        if attr_a != attr_b:
            raise MalformedTestError(
                f"Both sides of diff must reference the same attribute; "
                f"got {attr_a!r} vs {attr_b!r}",
                t.line,
            )
        return {"node_diff": {"nodes": [var_a, var_b], "attribute": attr_a}}

    # ── Semantic validation ───────────────────────────────────────────────

    def _validate_target(self, target: dict, declared: set[str],
                         edges: list[dict]) -> None:
        """
        Check that all variable / edge references in a target dict are
        declared in the MATCH pattern.

        For edge targets, both endpoints must be declared nodes AND a directed
        edge (from -> to) must exist in the pattern's edge list.
        """
        if "node" in target:
            var = target["node"]["id"]
            if var not in declared:
                raise UndeclaredVariableError(
                    f"TEST references undeclared variable {var!r}"
                )
        elif "node_diff" in target:
            for var in target["node_diff"]["nodes"]:
                if var not in declared:
                    raise UndeclaredVariableError(
                        f"TEST references undeclared variable {var!r}"
                    )
        elif "edge" in target:
            e   = target["edge"]
            frm = e["from"]
            to  = e["to"]
            for v in (frm, to):
                if v not in declared:
                    raise UndeclaredVariableError(
                        f"TEST references undeclared variable {v!r}"
                    )
            # The from->to direction must match a pattern edge exactly.
            if not any(ex["from"] == frm and ex["to"] == to for ex in edges):
                raise GHTQLError(
                    f"TEST references edge {frm!r}->{to!r} which is not declared "
                    f"in the MATCH pattern. Ensure the direction matches the "
                    f"pattern edge (use '<-' in MATCH to flip direction)."
                )
        elif "edge_diff" in target:
            for frm, to in target["edge_diff"]["edges"]:
                for v in (frm, to):
                    if v not in declared:
                        raise UndeclaredVariableError(
                            f"TEST references undeclared variable {v!r}"
                        )
                if not any(ex["from"] == frm and ex["to"] == to for ex in edges):
                    raise GHTQLError(
                        f"TEST references edge {frm!r}->{to!r} which is not "
                        f"declared in the MATCH pattern. Ensure the direction "
                        f"matches the pattern edge."
                    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _node_sort_key(node: dict) -> float:
    """Sort nodes by trailing numeric suffix; nodes without a suffix sort last."""
    m = re.search(r"(\d+)$", node["id"])
    return int(m.group(1)) if m else float("inf")


def _infer_type(nodes: list[dict], edges: list[dict]) -> str:
    """
    Infer the graph pattern type from topology (degree distribution),
    not merely from node/edge counts.

    Supported types
    ---------------
    node          : single isolated node (n=1, e=0)
    edge          : single directed edge (n=2, e=1)
    path          : linear chain — every node has degree ≤ 2 and at most one
                    node has degree 2 (the chain has no branches)
    star          : one hub with degree k≥2 connected to k leaves (degree 1),
                    total edges == k  (n = k+1, e = k)
    square        : exactly 4 nodes forming a directed 4-cycle / bipartite
                    quadrilateral  (n=4, e=4, each node degree 2)
    tailed_square : a square subgraph with one extra "tail" edge attached to
                    one of its nodes  (n=5, e=5)
    general       : anything else
    """
    n, e = len(nodes), len(edges)

    if n == 1 and e == 0:
        return "node"
    if n == 2 and e == 1:
        return "edge"

    # Build undirected degree map
    degree: dict[str, int] = defaultdict(int)
    for node in nodes:
        degree[node["id"]]  # ensure every node is present (degree 0 possible)
    for edge in edges:
        degree[edge["from"]] += 1
        degree[edge["to"]]   += 1
    deg_vals = sorted(degree.values())

    # ── path: all nodes have degree 1 or 2, exactly 2 endpoints (degree 1)
    if all(d <= 2 for d in deg_vals):
        endpoints = sum(1 for d in deg_vals if d == 1)
        if endpoints == 2 and e == n - 1:
            return "path"

    # ── star: one hub with degree == n-1, all others degree 1
    if n >= 3 and e == n - 1:
        max_deg = max(deg_vals)
        if max_deg == n - 1 and deg_vals.count(1) == n - 1:
            return "star"

    # ── square: 4 nodes, 4 edges, every node has undirected degree 2
    if n == 4 and e == 4 and all(d == 2 for d in deg_vals):
        return "square"

    # ── tailed_square: 5 nodes, 5 edges
    #    degree distribution: three nodes with degree 2, one with degree 3,
    #    one with degree 1  →  [1, 2, 2, 2, 3]
    if n == 5 and e == 5:
        if sorted(deg_vals) == [1, 2, 2, 2, 3]:
            return "tailed_square"

    return "general"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_string(src: str) -> dict[str, dict]:
    """
    Parse a GHT-QL source string; return a {pattern_id: entry} dict.
    If the number of comment IDs does not match the number of statements,
    extra statements are named pattern_0, pattern_1, … automatically.
    """
    cleaned, ids = _preprocess(src)
    tokens  = tokenize(cleaned)
    entries = _Parser(tokens).parse_program()

    result: dict[str, dict] = {}
    for i, entry in enumerate(entries):
        pid = ids[i] if i < len(ids) else f"pattern_{i}"
        entry["id"] = pid
        result[pid] = entry
    return result


def parse_file(filepath: str) -> dict[str, dict]:
    """Parse a .ghtql file; return a pattern dict."""
    with open(filepath, encoding="utf-8") as f:
        return parse_string(f.read())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="GHT-QL Parser — .ghtql → JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input",           help="input .ghtql file")
    ap.add_argument("-o", "--output",  help="output JSON file (omit to print to stdout)")
    args = ap.parse_args()

    data = parse_file(args.input)
    out  = json.dumps(data, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"✅ Parsed successfully. Written to {args.output}. {len(data)} pattern(s):")
    else:
        print(out)

    for pid, entry in data.items():
        print(f"  {entry['dataset']:12s} | {entry['type']:16s} | {pid}")


if __name__ == "__main__":
    _cli()
