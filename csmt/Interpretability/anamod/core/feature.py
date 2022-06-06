"""Feature class"""
import anytree
import numpy as np
import xxhash

ATTRIBUTES = dict(
    # p-value attributes
    pvalue=1.,
    ordering_pvalue=1.,
    window_pvalue=1.,
    window_ordering_pvalue=1.,
    # effect size attributes
    effect_size=0.,
    window_effect_size=0.,
    # importance attributes
    important=False,
    ordering_important=False,
    window_important=False,
    window_ordering_important=False,
    # misc attributes
    temporal_window=None
)


# pylint: disable = too-many-instance-attributes
class Feature(anytree.Node):
    """Class representing feature/feature group"""
    aliases = dict(
        overall_pvalue="pvalue",
        overall_important="important",
        overall_effect_size="effect_size",
        importance_score="effect_size",
        window_importance_score="window_effect_size",
        window="temporal_window"
    )

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    def __str__(self):
        out = f"Name: {self.name}\nDescription: {self.description}\nIndices: {self.idx}"
        for key, value in ATTRIBUTES.items():
            out += f"\n{key.replace('_', ' ').title()}: {value}"
        return out

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.description = kwargs.get("description", "")
        self.idx = kwargs.get("idx", [])
        self.perturbable = kwargs.get("perturbable", True)
        # TODO: (Verify) Could initialize the RNG right away, since cloudpickle should still be able to pickle it
        self._rng_seed = xxhash.xxh32_intdigest(name)
        self.rng = None  # RNG used for permuting this feature - see perturbations.py: 'feature.rng'
        for key, value in ATTRIBUTES.items():
            setattr(self, key, value)

    @property
    def rng_seed(self):
        """Get RNG seed"""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, seed):
        """Set RNG seed"""
        self._rng_seed = seed

    def initialize_rng(self):
        """Initialize random number generator for feature (used for permutations)"""
        self.rng = np.random.default_rng(self._rng_seed)

    def uniquify(self, uniquifier):
        """Add uniquifying identifier to name"""
        assert uniquifier
        self.name = f"{uniquifier}->{self.name}"

    @property
    def size(self):
        """Return size"""
        return len(self.idx)

    def copy_attributes(self, other):
        """Copy attributes from other feature"""
        for key in ATTRIBUTES:
            setattr(self, key, getattr(other, key))
