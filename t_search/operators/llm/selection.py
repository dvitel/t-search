''' LLM prompting based selection operators.

    Required environment variables:
        OPENAI_API_KEY - for OpenAI models
        GOOGLE_API_KEY - for Google Gemini models
'''

from dataclasses import dataclass, asdict
from typing import Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.operators.llm.utils import LLMCaller
from t_search.operators.selection import Selection
from t_search.syntax import Term

selection_prompt = """You are an expert in symbolic mathematics. 
Given a set of LISP expressions and their evaluations,
your task is to select one most promising expression for later modification.
You should only answer with the index.

Proposed terms:
{% for s in selection %}
{{ loop.index0 }}: {{ s.term }}
{{loss_name}}={{s.fitness | round(3)}}
{% endfor %}

Select the term that is both:
1. Likely to improve loss when mutated.
2. Has a structure that allows effective small modifications.

Output only:
- reason: 1–2 short bullets explaining why
- index: integer of selected term
"""

@dataclass
class SelectionExecution:    
    reason: list[str]
    index: int 

@dataclass 
class SelectionTermInfo:
    term: str
    fitness: float

@dataclass 
class PromptContext: 
    selection: list[SelectionTermInfo]

class LLMSelection(Selection):
    ''' Similar to Tournament Selection, but uses model decition what to pick. '''

    def __init__(self, *, 
                    llm: LLMCaller,
                    fitness: Fitness,
                    torch_gen: torch.Generator,
                    tournament_size: int = 7,
                    prompt_template: str = selection_prompt,
                    loss_name: str = "NMSE",
                    **kwargs):
        super().__init__(**kwargs)
        self.tournament_size = tournament_size
        from jinja2 import Template
        self.prompt_template: Template = Template(prompt_template)        
        self.loss_name = loss_name
        self.llm = llm
        self.fitness = fitness
        self.torch_gen = torch_gen

    def __call__(self, population) -> Sequence[Term]:
        fitness = self.fitness.get_fitness(population, return_fitness="tensor")
        selected_ids = torch.randint(len(population), 
                                     (self.selection_size, self.tournament_size), 
                                     dtype=torch.int, device=fitness.device,
                                     generator=self.torch_gen)
        selected_fitnesses = fitness[selected_ids]        

        children = []        
        for i in range(self.selection_size):
            candidiates = [population[idx] for idx in selected_ids[i].tolist()]
            candidiates_fitness = selected_fitnesses[i].tolist()
            selection_info = [SelectionTermInfo(term=str(candidiates[j]), fitness=candidiates_fitness[j]) for j in range(self.tournament_size)]

            context = PromptContext(
                selection=selection_info,
                loss_name=self.loss_name        
            )
            try:
                prompt = self.prompt_template.render(**asdict(context)) # prompt rendering 
            except Exception as e:
                print("Error rendering prompt:", e)
                self.add_metrics(render_error=1)
                continue
            
            try:
                response, usage = self.llm(prompt, SelectionExecution)
                self.add_metrics(**usage)
            except Exception as e:
                print("Error during LLM prompting:", e)
                self.add_metrics(llm_error=1)
                continue
            selected_id = response.index
            if selected_id < 0 or selected_id >= len(candidiates):
                self.add_metrics(invalid_index=1)
                continue
            selected_term = candidiates[selected_id]
            children.append(selected_term)

        del fitness

        return children