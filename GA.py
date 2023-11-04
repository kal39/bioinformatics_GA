import numpy as np


class Individual:
    # for generating a new individual from a sequence (used to make generation 0)
    def new(seq: list[str], w_gap: int, w_mismatch: int) -> "Individual":
        self = Individual()
        self.w_gap = w_gap
        self.w_mismatch = w_mismatch
        self.seq = np.array([bytearray(s, "utf-8") for s in seq], dtype=int)  # convert string to ascii array
        self.gap = np.zeros(self.seq.shape, dtype=int)  # initialize gaps to 0
        return self

    # for generating a new individual from two parents (for crossover)
    def from_parents(p1: "Individual", p2: "Individual") -> "Individual":
        self = Individual()
        self.w_gap = p1.w_gap
        self.w_mismatch = p1.w_mismatch
        self.seq = p1.seq
        gap_1 = p1.gap[:, :p1.gap.shape[1] // 2]  # first half from p1
        gap_2 = p2.gap[:, p1.gap.shape[1] // 2:]  # second half from p2
        self.gap = np.hstack([gap_1, gap_2])  # combine the two halves

        return self

    def evaluate(self) -> float:
        matches = sum([
            max([*[x[1] for x in zip(*np.unique(col, return_counts=True)) if chr(x[0]) != "-"], 1]) - 1
            for col in self._cur_seq().T
        ])
        # gaps = self.gap.sum(axis=1).max() * self.gap.shape[0]
        return matches

    def mutate(self, p: float) -> None:
        idx = np.random.rand(*self.gap.shape) < p  # select indices to mutate
        self.gap[idx] = [self._random_gap() for _ in range(idx.sum())]  # mutate with rand val between 1 and 10

    def _random_gap(self) -> int:
        return np.random.randint(0, self.seq.shape[1])

    # get the sequence the individual currently represents (including the gaps)
    def _cur_seq(self) -> list[str]:
        max_len = self.gap.sum(axis=1).max()  # maximum gap count (for padding)
        stacked = np.dstack([self.seq, self.gap])  # combine seq and gap arrays, so its easier to iterate
        raw_out = np.array([
            bytearray(
                "".join(["-" * s[1] + chr(s[0]) for s in seq]) +  # '-' * gap + char
                "-" * (max_len - seq[:, 1].sum()), "utf-8"  # pad with '-'s
            ) for seq in stacked
        ], dtype=int)  # convert string back to ascii int array

        return np.array([col for col in raw_out.T if "".join([chr(c) for c in col]).strip("-") != ""]).T

    # pretty print
    def __str__(self) -> str:
        return f"Score: {self.evaluate()}\n" + "\n".join(["".join([chr(c) for c in seq]) for seq in self._cur_seq()]) + "\n"


class Population:
    def __init__(self, seq: list[str], w_gap: int, w_mismatch: int, pop_count: int) -> None:
        self.individuals = [Individual.new(seq, w_gap, w_mismatch) for _ in range(pop_count)]  # init individuals

    def clear_individuals(self):
        self.individuals.clear()

    # return the best individual
    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.evaluate())

    def next_gen(self) -> None:
        orig_len = len(self.individuals)

        # Roulette Selection
        p = np.array([ind.evaluate() for ind in self.individuals])
        p = p / p.sum() if p.sum() != 0 else np.ones(p.shape[0]) / p.shape[0]

        new_individuals = []
        for _ in range(orig_len):
            parent_1 = np.random.choice(self.individuals, p=p)
            parent_2 = np.random.choice(self.individuals, p=p)
            new_individuals.append(Individual.from_parents(parent_1, parent_2))

        self.individuals = new_individuals

    def mutate(self, p: float) -> None:
        [ind.mutate(p) for ind in self.individuals]

    # pretty print
    def __str__(self) -> str:
        self.individuals = sorted(self.individuals, key=lambda ind: ind.evaluate(), reverse=True)
        # return "\n".join([str(ind) for ind in self.individuals])
        # return "Population: " + ", ".join([str(ind.evaluate()) for ind in self.individuals])
        return str(self.best())


# some similar-ish sequences (TODO: probably a horrible example)
sequences = [
    "AAA1111CC",
    "22AAACC22",
    "333333AAA",
]

pop = Population(sequences, 1, 1, 100)

for i in range(1000):
    print(pop)
    pop.next_gen()
    pop.mutate(0.01)
