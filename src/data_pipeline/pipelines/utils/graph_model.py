from typing import Dict, Optional, Any, List, Tuple
import pickle
import json


class SeqNode:
    """Node in the sequence tree."""

    def __init__(self, token: str, parent: Optional["SeqNode"] = None, depth: int = 1):
        self.token = token
        self.children: Dict[str, "SeqNode"] = {}
        self.region_counts: Dict[int, int] = {}
        self.total = 0
        self.parent = parent
        self.depth = depth
        self.is_leaf = False

    def add_region(self, region: int, count: int = 1):
        self.region_counts[region] = self.region_counts.get(region, 0) + count
        self.total += count


class SequenceTree:
    """Tree for sequence-based region prediction."""

    def __init__(
        self,
        purity_threshold: float = 0.85,
        min_support: int = 5,
        max_depth: Optional[int] = None,
    ):
        self.roots: Dict[str, SeqNode] = {}
        self.purity_threshold = purity_threshold
        self.min_support = min_support
        self.max_depth = max_depth
        self.global_region_prior = -1

    def build(self, training_data: List[Tuple[List[str], int]]) -> "SequenceTree":
        # logger.info(f"Building tree with {len(training_data)} samples")
        for tokens, region in training_data:
            if not tokens:
                continue
            first_token = tokens[0]
            if first_token not in self.roots:
                self.roots[first_token] = SeqNode(first_token, depth=1)
            current = self.roots[first_token]
            current.add_region(region)
            depth = 1
            for token in tokens[1:]:
                depth += 1
                if self.max_depth and depth > self.max_depth:
                    break
                if token not in current.children:
                    current.children[token] = SeqNode(
                        token, parent=current, depth=depth
                    )
                current = current.children[token]
                current.add_region(region)
        return self

    def _compute_purity(self, node: SeqNode) -> Tuple[float, Optional[int], int]:
        if node.total == 0:
            return 0.0, None, 0
        dominant_region, dominant_count = max(
            node.region_counts.items(), key=lambda x: x[1]
        )
        return dominant_count / node.total, dominant_region, dominant_count

    def _prune_node(self, node: SeqNode) -> bool:
        changed = False
        parent_purity, _, _ = self._compute_purity(node)
        for token in list(node.children.keys()):
            child = node.children[token]
            subtree_changed = self._prune_node(child)
            changed |= subtree_changed
            child_purity, _, _ = self._compute_purity(child)
            if child.total < self.min_support or child_purity <= parent_purity:
                del node.children[token]
                changed = True
                continue
            if child_purity >= self.purity_threshold:
                child.is_leaf = True
                child.children.clear()
                changed = True
        if node.parent is None:
            current_purity, _, _ = self._compute_purity(node)
            if (
                current_purity >= self.purity_threshold
                and node.total >= self.min_support
            ):
                node.is_leaf = True
                node.children.clear()
                changed = True
        return changed

    def prune(self, max_passes: int = 10) -> "SequenceTree":
        # logger.info("Starting pruning")
        # initial_count = self._count_nodes()
        for _ in range(max_passes):
            any_change = False
            for root in self.roots.values():
                any_change |= self._prune_node(root)
            if not any_change:
                break
        # final_count = self._count_nodes()
        # logger.info(f"Pruning done: {initial_count} -> {final_count} nodes")
        return self

    def _count_nodes(self) -> int:
        def count(node: SeqNode) -> int:
            return 1 + sum(count(child) for child in node.children.values())

        return sum(count(root) for root in self.roots.values())

    def predict(self, tokens: List[str]) -> Dict[str, Any]:
        default_res = {"region": -1, "matched_rule": "", "purity": 0.0, "support": 0}
        if not tokens:
            return default_res
        # Try sliding window starting from each position
        # print(f"tokens: {tokens}")
        for start in range(len(tokens)):
            sub_tokens = tokens[start:]
            current = self.roots.get(sub_tokens[0])
            if not current:
                continue
            matched_path = [sub_tokens[0]]
            # print(f"matched_path: {matched_path}")
            deepest = current
            for token in sub_tokens[1:]:
                if deepest.is_leaf:
                    break
                next_node = deepest.children.get(token)
                if not next_node:
                    break
                deepest = next_node
                matched_path.append(token)
            # print(f"is_leaf: {deepest.is_leaf}")
            # print(f"region_counts: {deepest.region_counts}")
            purity, dominant_region, support = self._compute_purity(deepest)
            if purity >= self.purity_threshold and support >= self.min_support:
                return {
                    "region": int(dominant_region) if dominant_region else 0,
                    "matched_rule": " -> ".join(matched_path),
                    "purity": round(purity, 3),
                    "support": support,
                }

        return default_res

    def _get_node_by_path(self, tokens: List[str]) -> Optional[SeqNode]:
        if not tokens:
            return None
        current = self.roots.get(tokens[0])
        for token in tokens[1:]:
            if not current:
                return None
            current = current.children.get(token)
        return current

    def export_rules(self) -> List[Dict[str, Any]]:
        # logger.info("Exporting rules")
        raw_rules = []

        def dfs(node: SeqNode, path: List[str]):
            if node.is_leaf:
                purity, major_region, _ = self._compute_purity(node)
                raw_rules.append(
                    {
                        "tokens": path.copy(),
                        "region": major_region,
                        "support": node.total,
                        "purity": purity,
                        "histogram": node.region_counts,
                    }
                )
            for token, child in node.children.items():
                dfs(child, path + [token])

        for token, root in self.roots.items():
            dfs(root, [token])

        optimized_rules: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for rule in raw_rules:
            tokens = rule["tokens"]
            while len(tokens) > 1:
                suffix = tokens[1:]
                suffix_node = self._get_node_by_path(suffix)
                if suffix_node:
                    s_purity, _, _ = self._compute_purity(suffix_node)
                    if s_purity >= rule["purity"]:
                        tokens = suffix
                        rule.update(
                            {
                                "purity": s_purity,
                                "support": suffix_node.total,
                                "histogram": suffix_node.region_counts,
                            }
                        )
                        continue
                break
            rule["tokens"] = tokens
            key = tuple(tokens)
            if (
                key not in optimized_rules
                or rule["purity"] > optimized_rules[key]["purity"]
            ):
                optimized_rules[key] = rule
        return list(optimized_rules.values())

    def save(self, filepath: str, format_type: str = "pickle"):
        if format_type == "json":
            state = {
                "metadata": {
                    "purity_threshold": self.purity_threshold,
                    "min_support": self.min_support,
                    "max_depth": self.max_depth,
                    "global_region_prior": self.global_region_prior,
                },
                "rules": self.export_rules(),
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str, format_type: str = "pickle") -> "SequenceTree":
        if format_type == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        if format_type == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            instance = cls(
                purity_threshold=meta.get("purity_threshold", 0.85),
                min_support=meta.get("min_support", 5),
                max_depth=meta.get("max_depth"),
            )
            instance.global_region_prior = meta.get("global_region_prior", -1)
            for rule in data.get("rules", []):
                tokens = rule.get("tokens", [])
                if not tokens:
                    continue
                hist = rule.get("histogram", {})
                if tokens[0] not in instance.roots:
                    instance.roots[tokens[0]] = SeqNode(tokens[0], depth=1)
                current = instance.roots[tokens[0]]
                if len(tokens) > 1:
                    current.add_region(0, 5)
                    current.add_region(1, 5)
                else:
                    for r_id, count in hist.items():
                        current.add_region(int(r_id), count)

                for i, token in enumerate(tokens[1:], start=2):
                    if token not in current.children:
                        current.children[token] = SeqNode(
                            token, parent=current, depth=i
                        )
                    current = current.children[token]
                    if token != tokens[-1]:
                        current.add_region(0, 5)
                        current.add_region(1, 5)
                    else:
                        for r_id, count in hist.items():
                            current.add_region(int(r_id), count)

                current.is_leaf = True
            return instance
        raise ValueError(f"Unsupported format: {format_type}")
