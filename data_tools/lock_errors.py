from .problems import Problem, ProblemCode

BREAKING = frozenset(
    {
        ProblemCode.LOCKED_CHANGED,
        ProblemCode.LOCKED_MISSING,
        ProblemCode.SPLIT_RULE_CHANGED,
    }
)


class LockConflict(Exception):
    def __init__(self, problems: list[Problem]):
        self.problems = problems
        super().__init__("; ".join(str(problem) for problem in problems))
