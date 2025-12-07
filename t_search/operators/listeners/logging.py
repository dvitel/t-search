import os
import sys
from time import time
from typing import TextIO

import torch

from t_search.base import ServiceBase

from .base import EvalListener, GenListener

# TODO: formatter
class Logging(ServiceBase, EvalListener, GenListener):
    ''' Outputs new terms into provided file/ descriptor '''

    def __init__(self, *, 
                file: TextIO = sys.stdout,
                file_path: str = "",
                autoflush: bool = True, 
                mode: str = "w",
                ):
        self.timestamp = int(time())
        self.mode = mode
        self.file_path = file_path
        self.full_file_name = ""
        self.file_stream: TextIO = file
        self.autoflush = autoflush

    def init(self):
        if self.file_path == "":
            self._prompt_start()        
            return
        self.full_file_name = self.file_path.format(self.timestamp)
        os.makedirs(os.path.dirname(self.full_file_name), exist_ok=True)
        self.file_stream = open(self.full_file_name, self.mode)
        self._prompt_start()

    def get_finalizer(self):
        self._prompt_end()
        def finalizer():
            if self.file_path != "":
                self.file_stream.close()
        return finalizer

    def _prompt_start(self):
        self.file_stream.write(f"#START time={self.timestamp}\n")

    def _prompt_end(self):
        self.file_stream.write(f"#END time={self.timestamp} end={int(time())}\n")

    def on_gen_start(self, gen: int, population: list):
        self.file_stream.write(f"#GEN_START gen={gen} pop={len(population)}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()

    def on_gen_end(self, gen: int, population: list):
        self.file_stream.write(f"#GEN_END gen={gen} pop={len(population)}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()

    def on_eval(self, terms, semantics, fitness: torch.Tensor | None = None):
        for term_i, term in enumerate(terms):
            fstr = "N/A"
            if fitness is not None:
                fstr = str(fitness[term_i].item())
            self.file_stream.write(f"{str(term)} {fstr}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()