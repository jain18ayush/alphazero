# src/registry.py
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, key: str):
        def decorator(fn):
            if key in self._items:
                raise KeyError(f"{key} already registered in {self.name}")
            self._items[key] = fn
            return fn
        return decorator

    def get(self, key: str):
        if key not in self._items:
            available = ", ".join(self._items.keys())
            raise KeyError(
                f"Unknown {self.name}: '{key}'. Available: [{available}]"
            )
        return self._items[key]
