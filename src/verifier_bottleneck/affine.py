from dataclasses import dataclass

MOD = 101


@dataclass(frozen=True)
class AffineMap:
    a: int
    b: int
    mod: int = MOD

    def __post_init__(self):
        if self.a % self.mod == 0:
            raise ValueError("a must be non-zero modulo mod")

        object.__setattr__(self, "a", self.a % self.mod)
        object.__setattr__(self, "b", self.b % self.mod)

    def __call__(self, x: int) -> int:
        return (self.a * x + self.b) % self.mod

    def compose_after(self, inner: "AffineMap") -> "AffineMap":
        """
        self(inner(x)).
        """
        return AffineMap(
            a=self.a * inner.a,
            b=self.a * inner.b + self.b,
            mod=self.mod,
        )