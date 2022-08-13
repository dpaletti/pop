from typing import AbstractSet, Iterable, Optional

import pylru


class FixedSet(AbstractSet):
    def __init__(self, size: int, iterable: Optional[Iterable] = None):
        self.cache: pylru.lrucache = pylru.lrucache(size=size)
        if iterable is None:
            return

        for x in iterable:
            self.add(x)

    def __contains__(self, x: object) -> bool:
        return x in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __iter__(self):
        return self.cache.__iter__()

    def __str__(self):
        return set(self.cache.keys()).__str__()

    def add(self, x: object):
        self.cache[x] = None
