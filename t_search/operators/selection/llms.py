''' LLM prompting based selection operators.

    Required environment variables:
        OPENAI_API_KEY - for OpenAI models
        GOOGLE_API_KEY - for Google Gemini models
'''

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Annotated, Sequence

import torch

from syntax import Term
from .base import Selection
from .. import llm

if TYPE_CHECKING:
    from t_search.solver import GPSolver

selection_prompt = """You are an expert in symbolic mathematics and programming. 
Given a set of LISP expressions and their evaluations,
your task is to select one most promising expression for later modification.
You should only answer with the index.

Proposed terms:
{% for s in selection %}
{{ loop.index0 }}: {{ s.term }}
{{loss_name}}={{s.fitness | round(3)}}
{% endfor %}

What term is best for modification w.r.t. loss improvement and complexity of modification?
"""

# TODO: reasoning?? 

@dataclass
class SelectionExecution:    
    selected: Annotated[int, "Index"]

@dataclass 
class SelectionTermInfo:
    term: str
    fitness: float

@dataclass 
class PromptContext: 
    selection: list[SelectionTermInfo]

class LLMS(Selection):
    ''' Similar to Tournament Selection, but uses model decition what to pick. '''

    def __init__(self, name: str = "LLMS", *, 
                    llm: llm.LLMCaller,
                    tournament_size: int = 7,
                    prompt_template: str = selection_prompt,
                    loss_name: str = "NMSE"):
        super().__init__(name)
        self.tournament_size = tournament_size
        from jinja2 import Template
        self.prompt_template: Template = Template(prompt_template)        
        self.loss_name = loss_name
        self.llm = llm

    def select(self, solver: 'GPSolver', population, selection_size: int) -> Sequence[Term]:
        fitness = solver.eval(population, return_fitness="list").fitness
        selected_ids = torch.randint(len(population), (selection_size, self.tournament_size), dtype=torch.int, device=fitness.device,
                                    generator=solver.torch_gen)
        selected_fitnesses = fitness[selected_ids]

        children = []        
        for i in range(selection_size):
            candidiates = [population[idx] for idx in selected_ids[i].tolist()]
            candidiates_fitness = selected_fitnesses[i].tolist()
            selection_info = [SelectionTermInfo(term=str(candidiates[j]), fitness=candidiates_fitness[j]) for j in range(self.tournament_size)]

            context = PromptContext(
                selection=selection_info,
                loss_name=solver.loss.name        
            )
            try:
                prompt = self.prompt_template.render(**asdict(context)) # prompt rendering 
            except Exception as e:
                print("Error rendering prompt:", e)
                self.add_metric(render_error=1)
                continue
            
            try:
                response = self.llm(prompt, SelectionExecution)
            except Exception as e:
                print("Error during LLM prompting:", e)
                self.add_metric(llm_error=1)
                continue
            selected_id = response.selected
            if selected_id < 0 or selected_id >= len(candidiates):
                self.add_metric(invalid_index=1)
                continue
            selected_term = candidiates[selected_id]
            children.append(selected_term)

        return children