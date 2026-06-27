"""Small internal utilities used across RUBIS."""


class DotDict(dict):
    """Dictionary supporting attribute-style access."""

    def __getattr__(*args):
        value = dict.get(*args)
        return DotDict(value) if type(value) is dict else value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__