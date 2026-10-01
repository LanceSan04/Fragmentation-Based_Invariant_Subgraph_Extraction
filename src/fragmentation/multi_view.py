"""Utilities for assigning ensemble views to fragmentation methods.

``view_fragmentations`` is the single source of truth for model-view fragment
graphs. A one-method recipe keeps the original lean data layout; a recipe with
multiple positive-count methods stores method-prefixed fragment graph fields.
"""

from collections import OrderedDict

from src.fragmentation import build_fragmenter


def normalize_fragmentation_name(name):
    """Return a normalized fragmentation method name."""
    return str(name or "group_graph").strip().lower()


def normalized_view_fragmentations(config, include_zero=True):
    """Return an ordered ``{method: count}`` mapping from view_fragmentations."""
    compact = _config_get(config, "view_fragmentations", None)
    if compact is None:
        raise ValueError(
            "config must define view_fragmentations, for example "
            "{'group_graph': 5} or {'group_graph': 5, 'brics': 3, 'recap': 2}"
        )

    items = compact.items() if isinstance(compact, dict) else compact
    counts = OrderedDict()
    for method, count in items:
        name = normalize_fragmentation_name(method)
        count = int(count)
        if count < 0:
            raise ValueError(f"view_fragmentations count for {name!r} cannot be negative")
        build_fragmenter(name)
        if count > 0 or include_zero:
            counts[name] = count
    return counts


def expand_view_fragmentations(config):
    """Return one fragmentation method name per model view.

    Accepted config forms:

    ``view_fragmentations: {group_graph: 5, brics: 3, recap: 2}``
        Compact method/count mapping. Zero-count methods are allowed and ignored
        for view assignment.
    """
    num_views = int(_config_get(config, "num_views", 1))

    methods = []
    for method, count in normalized_view_fragmentations(config, include_zero=False).items():
        methods.extend([method] * count)

    if len(methods) != num_views:
        raise ValueError(
            "view fragmentation assignment must contain exactly num_views entries "
            f"({len(methods)} != {num_views})"
        )

    for method in methods:
        build_fragmenter(method)
    return methods


def unique_fragmentation_methods(config):
    """Return unique configured view methods in first-use order."""
    methods = OrderedDict()
    for method in expand_view_fragmentations(config):
        methods[method] = None
    return list(methods)


def uses_mixed_fragmentations(config):
    """Return True when more than one positive-count method is configured."""
    return len(unique_fragmentation_methods(config)) > 1


def primary_fragmentation_method(config):
    """Return the first positive-count fragmentation method."""
    methods = unique_fragmentation_methods(config)
    if not methods:
        raise ValueError("view_fragmentations must assign at least one view")
    return methods[0]


def fragmentation_recipe_label(config):
    """Return a path-safe label describing the configured fragmentation recipe."""
    counts = normalized_view_fragmentations(config, include_zero=False)
    positive = [(method, count) for method, count in counts.items() if count > 0]
    if len(positive) == 1:
        return positive[0][0]
    return "mixed_" + "_".join(f"{method}{count}" for method, count in positive)


class MultiViewFragmenter:
    """Run multiple registered fragmenters for the same molecule."""

    def __init__(self, methods=None):
        self.methods = []
        self.fragmenters = OrderedDict()
        for method in methods or ["group_graph"]:
            name = normalize_fragmentation_name(method)
            if name in self.fragmenters:
                continue
            self.methods.append(name)
            self.fragmenters[name] = build_fragmenter(name)

    def fragment(self, mol):
        """Return the first method's fragmentation for Fragmenter compatibility."""
        return self.fragmenters[self.methods[0]].fragment(mol)

    def fragment_all(self, mol):
        """Return ``{method_name: FragmentationResult}`` for every configured method."""
        return {method: fragmenter.fragment(mol) for method, fragmenter in self.fragmenters.items()}


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
