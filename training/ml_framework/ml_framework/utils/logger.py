from typing import List


class Print:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, *message: List[str]):
        if self.verbose:
            print(*message)
