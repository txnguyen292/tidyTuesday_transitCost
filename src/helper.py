import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, TypeVar, List
import logging
from random import randrange
from logzero import setup_logger

from config import CONFIG

T = TypeVar("T")
Vector = List[T]

logger = setup_logger(__file__, level=logging.DEBUG)

class kfoldCrossVal:
    """Kfold Cross Validation
    """
    def __init__(self, k: int) -> None:
        self.k: int = k
    def fit(self, df: Iterable[T]) -> None:
        """Randome generate a kfold scheme for cross validation

        Args:
            df (pd.DataFrame): input data
        """
        self.N: int = len(df)
        self.train_indices: List[List[int]] = []
        self.test_indices: List[List[int]] = []
        self.ksize: int = int(self.N / self.k)
        self.df_indices: List[int] = list(range(self.N))
    
    def split(self):
        for k in range(self.k):
            fold = list()
            while len(fold) < self.ksize:
                index = randrange(len(self.df_indices))
                fold.append(self.df_indices.pop(index))
            self.test_indices.append(fold)
            self.train_indices.append([x for x in range(self.N) if x not in fold])
        return zip(self.train_indices, self.test_indices)

if __name__ == "__main__":
    kfold = kfoldCrossVal(k=5)
    train_data = pd.read_csv(CONFIG.data / "final" / "train.csv")
    kfold.fit(train_data)
    for train_indices, test_indices in kfold.split():
        logger.info("Train and test indices:")
        logger.info(f"{len(train_indices)}, {len(test_indices)}")
        whole_indices = train_indices + test_indices
        whole_indices.sort()
        assert whole_indices == list(range(kfold.N)), "wrong split!"
    assert kfold.test_indices[0] != kfold.test_indices[1], "your kfold returns the same indices for all slits!"
    
