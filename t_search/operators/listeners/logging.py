import os
import sys
from time import time
from typing import TYPE_CHECKING, TextIO

import torch

from .base import Listener

if TYPE_CHECKING:
    from t_search.solver import GPSolver

# TODO: formatter
class LoggingListener(Listener):
    ''' Outputs new terms into provided file/ descriptor '''

    def __init__(self, name="Log", *, 
                file: TextIO = sys.stdout,
                file_path: str = "",
                autoflush: bool = True, 
                mode: str = "w",
                use_timestamp: bool = True
                ):
        super().__init__(name)
        self.reset_count = -1
        self.use_timestamp = use_timestamp
        self.timestamp = int(time())
        self.mode = mode
        self.file_path = file_path
        self.full_file_name = ""
        self.file_stream: TextIO = file
        self.autoflush = autoflush

    def _prompt_start(self):
        self.file_stream.write(f"#START run={self.reset_count} start={self.timestamp}\n")

    def _prompt_end(self):
        self.file_stream.write(f"#END run={self.reset_count} start={self.timestamp} end={int(time())}\n")

    def on_start(self, solver: 'GPSolver'):
        super().on_start(solver)
        self.reset_count += 1
        self.timestamp = int(time())
        if self.file_path == "": # use stdout/file provided    
            self._prompt_start()        
            return
        if self.file_stream is not None:
            self.file_stream.close()
        self.full_file_name = self.file_path.format(self.timestamp if self.use_timestamp else self.reset_count)
        os.makedirs(os.path.dirname(self.full_file_name), exist_ok=True)
        self.file_stream = open(self.full_file_name, self.mode)
        self._prompt_start()

    def on_end(self, solver: 'GPSolver'):
        self._prompt_end()
        if self.file_path != "":
            self.file_stream.close()

    def on_gen_start(self, solver: 'GPSolver', gen: int, population: list):
        self.file_stream.write(f"#GEN_START gen={gen} pop={len(population)}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()

    def on_gen_end(self, solver: 'GPSolver', gen: int, population: list):
        self.file_stream.write(f"#GEN_END gen={gen} pop={len(population)}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()

    def on_eval(self, solver, terms, semantics, fitness: torch.Tensor | None = None):
        for term_i, term in enumerate(terms):
            fstr = "N/A"
            if fitness is not None:
                fstr = str(fitness[term_i].item())
            self.file_stream.write(f"{str(term)} {fstr}\n")
        if self.file_path != "" and self.autoflush:
            self.file_stream.flush()