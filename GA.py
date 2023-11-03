import numpy as np


class Individual:
    # for generating a new individual from a sequence
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
        g_count = self.gap.sum()  # count the number of gaps
        m_count = sum([np.unique(i).shape[0] - 1 for i in self._cur_seq().T])  # count number of unique chars per col
        return -(self.w_gap * g_count + self.w_mismatch * m_count)

    def mutate(self, p: float) -> None:
        idx = np.random.rand(*self.gap.shape) < p  # select indices to mutate
        self.gap[idx] = [np.random.randint(1, 10) for _ in range(idx.sum())]  # mutate with rand val between 1 and 10

    # get the sequence the individual currently represents (including the gaps)
    def _cur_seq(self) -> list[str]:
        max_len = self.gap.sum(axis=1).max()  # maximum gap count (for padding)
        stacked = np.dstack([self.seq, self.gap])  # combine seq and gap arrays, so its easier to iterate
        return np.array([
            bytearray(
                "".join(["-" * s[1] + chr(s[0]) for s in seq]) +  # '-' * gap + char
                "-" * (max_len - seq[:, 1].sum()), "utf-8"  # pad with '-'s
            ) for seq in stacked
        ], dtype=int)  # convert string back to ascii int array

    # pretty print
    def __str__(self) -> str:
        return f"Score: {self.evaluate()}\n" + "\n".join(["".join([chr(c) for c in seq]) for seq in self._cur_seq()]) + "\n"


class Population:
    def __init__(self, seq: list[str], w_gap: int, w_mismatch: int, pop_count: int) -> None:
        self.individuals = [Individual.new(seq, w_gap, w_mismatch) for _ in range(pop_count)]  # init individuals

    # return the best individual
    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.evaluate())

    # TODO: this part is a bit sketchy
    def eval_and_crossover(self, p: float) -> None:
        orig_len = len(self.individuals)

        # TODO: not roulette selection
        self.individuals = sorted(self.individuals, key=lambda ind: ind.evaluate(), reverse=True)  # sort by score
        self.individuals = self.individuals[int(len(self.individuals) * p):]  # select top p% of individuals

        # generate new individuals
        self.individuals = [
            Individual.from_parents(
                np.random.choice(self.individuals),  # randomly sample a parents
                np.random.choice(self.individuals)  # randomly sample a parents
            ) for _ in range(orig_len)  # re-populate
        ]

    def mutate(self, p: float) -> None:
        [ind.mutate(p) for ind in self.individuals]

    # pretty print
    def __str__(self) -> str:
        return "==== Population ====\n\n" + "\n".join([str(ind) for ind in self.individuals])


# some similar-ish sequences (TODO: probably a horrible example)
sequences = [
    "CGTGGGTGTGTTCTGTG",
    "CGTGGTGGGTTCTGTAG",
    "CGTGGATGTGTACGGTG",
    "CGGGGATGGTGTTCTGT",
]

# for testing mutation:

pop = Population(sequences, 1, 1, 10)

print(pop)

pop.mutate(0.01)

print(pop)

# TODO: the GA should technically work (all the code is there) but in reality, not so great
