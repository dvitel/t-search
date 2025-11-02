''' LLM prompting-based Mutation operator.
    Zero-shot: requests direct mutation 
    ICL based: providing several examples 
    CoT based: provide reasoning steps     
'''

from dataclasses import dataclass, asdict
import os
from typing import TYPE_CHECKING, Annotated, Sequence

import torch

from syntax import Term
from .base import TermMutation
from .. import llm

if TYPE_CHECKING:
    from t_search.solver import GPSolver

@dataclass 
class TestCaseContext:
    vars: dict[str, float]
    expected: float
    actual: float
    diff: float = None
    actual_after: float | None = None
    diff_after: float | None = None
    
@dataclass 
class GoodMutationContext:
    term: str 
    term_after: str
    tests: Sequence[TestCaseContext]
    fitness_diff: float

@dataclass 
class PromptContext: 
    previous_good_mutations: Sequence[GoodMutationContext]
    term: str
    tests: Sequence[TestCaseContext]
    fitness: float
    op_desc: dict[str, str]        
    free_vars: Sequence[str]
    num_mutations: int = 3
    num_positions: int = 2
    loss_name: str

# DONE: ICL --> best mutations and their effect 
# DONE: CoT --> select position explain why and then mutate there
# DONE: structured format: return explicitly the subexpression to mutate
# DONE: __init__ params 
# DONE: context initialization 
# DONE: parsing 
# DONE: evaluation and output of the best
# DONE: preserving good mutations into ICL 

# DONE: ICL exemplar adjustment in prompt - maybe better to show improvement instead of output actual values?
# DONE: metrics

# TODO: simpler version of prompt to reduce number of tokens
# TODO: Generation of sketch (arbitrary constants) instead of concrete term
# TODO: management of requests for same term?

tests_prompt = """You are an expert in symbolic mathematics and programming. 
Given a LISP expression presenting a original mathematical term, 
your task is to provide a mutated/changed version of that term to bring its outputs closer to expected outputs on given tests and minimize {{loss_name}}.
The expected term values are provided in a form of tests: free variable values, expected result.
Ensure that your modification forms the correct LISP expression from the free variables 
{%- for free_var in free_vars -%} {{free_var}}, {% endfor -%}, 
constants and following operations:
{% for key, value in op_desc.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Try to keep expression depths below {{max_depth}}.
Number of constants in one expression should not exceed {{max_constants}}.

{% for example in previous_good_mutations -%}
{% if loop.first %}
Examples of improvement:
{% endif %}
    {{ loop.index }}. Original term:
    {{example.term}}
    After modification:
    {{example.term_after}}
    {% for test in example.tests %}
        {% if loop.first %}
        Tests with highest improvement:
        {% endif %}
        {{ loop.index }}. {% for free_var, free_var_value in test.vars.items() -%} {{ free_var }} = {{ free_var_value | round(3) }} {%- endfor %}
        Distance change to expected outcome:
        {%- if test.diff < 0 -%}
        output was lower by {{ test.diff | round(3) }}, 
        {%- else -%}
        output was higher by {{ test.diff | round(3) }}, 
        {%- endif -%}
        {%- if test.diff_after < 0 -%}
        became lower by {{ test.diff_after | round(3) }}
        {%- else -%}
        became higher by {{ test.diff_after | round(3) }}
        {%- endif -%}            
    {%- endfor %}
    {{loss_name}} decreased by {{ example.fitness_diff | round(5) }}.
{%- endfor %}

Given term:
{{term}}
Hardest tests:
    {% for test in tests %}    
        {{ loop.index }}. {% for free_var, free_var_value in test.vars.items() -%} {{ free_var }} = {{ free_var_value | round(3) }} {%- endfor %}
        Distance to expected outcome:
        {%- if test.diff < 0 -%}
        output is lower by {{ test.diff | round(3) }}, 
        {%- else -%}
        output is higher by {{ test.diff | round(3) }}, 
        {%- endif -%}
    {% endfor %}
{{loss_name}} is {{ fitness | round(5) }}.

Generate {{num_mutations}} alternative mutations for the current term. 
For each mutation:
1. First, select from one to {{num_positions}} subexpressions to modify and provide a short reason for selection.
2. Then, decide the change that improves tests and loss (operator replacement, constant adjustment, subtree replacement, etc.).
3. Finnaly, apply the mutation and provide the resulting LISP expression.
Prefer minimal modifications that reduce error without bloating the term.
"""

@dataclass
class MutationPositionExecution:
    reason: Annotated[str, "Short explanation why this subexpression was selected (e.g., contributes most to error)"]
    position: Annotated[str, "The select LISP subexpression."]
    mutation_type: Annotated[str, "Type of mutation applied"]

@dataclass
class MutationExecution:
    all_selected_positions: Annotated[list[MutationPositionExecution], "List of positions selected for mutation."]
    final_mutated_term: Annotated[str, "Final LISP expression after applying the mutations to the selected positions."]

@dataclass
class AllMutationExecutions:
    mutations: Annotated[list[MutationExecution], "List of all mutation executed independently."]

class LLMM(TermMutation):

    def __init__(self, name = "LLMM", *, 
                 llm: llm.LLMCaller,
                 rate = 0.8, 
                #  prompt_template_path: str = "",
                 prompt_template: str = tests_prompt,
                 op_desc: dict[str, str] = llm.ops_descriptions,
                 num_mutations: int = 3,
                 max_num_positions: int = 2,
                 max_num_examples: int = 3,
                 max_num_tests: int = 3,
                 max_num_demo_tests: int = 2,
                 loss_name: str = "NMSE",
                 **kwargs):
        super().__init__(name, rate=rate, **kwargs)
        self.num_mutations = num_mutations
        self.max_num_positions = max_num_positions
        self.max_num_examples = max_num_examples
        self.max_num_tests = max_num_tests
        self.max_num_demo_tests = max_num_demo_tests
        self.loss_name = loss_name
        from jinja2 import Template
        self.prompt_template: Template = Template(prompt_template)
        self.llm = llm
        self.op_desc = op_desc        
        self.good_mutations: list[GoodMutationContext] = []

    def select_exemplars(self, solver: 'GPSolver', term: str):
        ''' Currently we pick just last N good mutations 
            TODO: consider similarity, diversity metrics and NMSE improvement
                  (article on salient aspects)
        '''
        if len(self.good_mutations) <= self.max_num_examples:
            return self.good_mutations
        rand_good_mutations = solver.rnd.choice(
            self.good_mutations, 
            size=self.max_num_examples, 
            replace=False
        )
        return rand_good_mutations

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        '''
            Note that term may have inf or nan in outputs - invalid term,
            this should be handled before calling this method.
        '''

        evals = solver.eval(term, return_outputs="list", return_fitness='list')        
        outcomes = evals.outputs[0]
        fitness = evals.fitness[0].item()
        target = solver.target

        # selecting hardest tests from this term 
        outcome_diffs = torch.abs(target - outcomes)
        test_ids = torch.argsort(outcome_diffs)
        hardest_test_ids = test_ids[-self.max_num_tests:]
        del test_ids
        expected = target[hardest_test_ids].tolist()
        actual = outcomes[hardest_test_ids].tolist()
        var_values = {var_name:var_vals[hardest_test_ids].tolist() for var_name, var_vals in solver.var_binding}
        tests = [TestCaseContext(vars = {var_name: var_values[var_name][i] for var_name in solver.vars.keys()},
                                  expected = expected[i],
                                  actual = actual[i],
                                  diff = actual[i] - expected[i]
                                  ) for i in range(len(hardest_test_ids))]

        demonstrations = self.select_exemplars(solver, term)
        context = PromptContext(
            num_mutations=self.num_mutations,
            num_positions=self.max_num_positions,
            term = str(term),
            tests = tests,
            fitness = fitness,
            op_desc = self.op_desc,
            free_vars=list(solver.vars.keys()),
            loss_name=self.loss_name,
            previous_good_mutations = demonstrations
        )
        
        try:
            prompt = self.prompt_template.render(**asdict(context)) # prompt rendering 
        except Exception as e:
            print("Error rendering prompt:", e)
            self.add_metric(render_error=1)
            return None

        try:
            response = self.llm(prompt, AllMutationExecutions)
        except Exception as e:
            print("Error during LLM prompting:", e)
            self.add_metric(llm_error=1)
            return None

        candidates = []
        for mutation in response.mutations:
            new_term = solver.parse_term_str(mutation.final_mutated_term)
            if new_term is None:
                self.add_metric(llm_syn_invalid=1)
                continue
            if not solver.is_valid(new_term):
                self.add_metric(llm_constr_invalid=1)
                continue
            candidates.append(new_term)

        if len(candidates) == 0:
            return None
        
        outputs, fitnesses = solver.eval(candidates, return_outputs="list", return_fitness="list")

        best_id = torch.argmin(fitnesses).item()
        best_term = candidates[best_id]
        best_fitness = fitnesses[best_id].item()
        best_outcomes = outputs[best_id]

        good_mutation_count = len(f for f in fitnesses if f < fitness)
        bad_mutation_count = len(f for f in fitnesses if f > fitness)
        neutral_mutation_count = len(f for f in fitnesses if f == fitness)
        self.add_metric(
            good_mutations=good_mutation_count, 
            bad_mutations=bad_mutation_count, 
            neutral_mutations=neutral_mutation_count
        )

        if best_fitness < fitness: # good mutation - record it 

            # selecting tests with best outcome improvements 
            outcome_diffs_after = torch.abs(target - best_outcomes)
            outcome_improvements = outcome_diffs - outcome_diffs_after
            test_ids = torch.argsort(outcome_improvements)
            best_test_ids = test_ids[-self.max_num_demo_tests:]
            best_test_id_ids, = torch.where(outcome_improvements[best_test_ids] > 0)
            final_best_test_ids = best_test_ids[best_test_id_ids]
            best_expected = target[final_best_test_ids].tolist()
            best_actual = outcomes[final_best_test_ids].tolist()
            best_actual_after = best_outcomes[final_best_test_ids].tolist()
            best_free_vars = {var_name:var_values[final_best_test_ids].tolist() for var_name, var_values in solver.var_binding}

            tests = [TestCaseContext(vars={var_name: best_free_vars[var_name][i] for var_name in solver.vars.keys()},
                                     expected=best_expected[i],
                                     actual=best_actual[i],
                                     diff=best_actual[i] - best_expected[i],
                                     actual_after=best_actual_after[i],
                                     diff_after=best_actual_after[i] - best_expected[i]
                                     )
                     for i in final_best_test_ids]
            good_mutation = GoodMutationContext(
                term = context.term, 
                term_after = str(best_term),
                fitness_diff = fitness - best_fitness,
                tests = tests
            )
            self.good_mutations.append(good_mutation)
        
        return best_term 