''' LLM prompting-based Mutation operator.
    Zero-shot: requests direct mutation 
    ICL based: providing several examples 
    CoT based: provide reasoning steps     
'''

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimizer import Optimizer
from t_search.evaluators.semantics import Semantics
from t_search.operators.llm.utils import ops_descriptions, LLMCaller
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term
from t_search.syntax.str import term_to_const_skeleton_str
from t_search.syntax.syntax import Syntax

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
    loss_name: str = "nmse"

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

# DONE: simpler version of prompt to reduce number of tokens
# DONE: Generation of sketch (arbitrary constants) instead of concrete term

tests_prompt = """You are an expert in symbolic mathematics. 
Given a LISP expression for a mathematical term, produce a mutated version that reduces {{loss_name}} on the tests below.
Allowed variables {%- for v in free_vars -%} {{v}}{% if not loop.last %}, {% endif %}{% endfor %}.
Use C as a placeholder constant.
Allowed operations:
{% for k,v in op_desc.items() %}- {{k}}: {{v}}
{% endfor %}
Constraints: depth ≤ {{max_depth}}, constants ≤ {{max_constants}}.

{% for e in previous_good_mutations -%}
    {% if loop.first %}
    Examples of improving mutations:
    {% endif %}
    {{loop.index}}. {{e.term}} → {{e.term_after}}  (Δ{{loss_name}} = {{e.fitness_diff|round(4)}})
    {% for t in e.tests %}
    test{{loop.index}}: {% for var,val in t.vars.items() -%}{{var}}={{val|round(3)}}, {% endfor %}
    Δ={{t.diff|round(3)}} → Δ={{t.diff_after|round(3)}}
    {% endfor %}
{% endfor %}

Current term:
{{term}}
Hard tests:
{% for t in tests %}
  test{{loop.index}}: {% for v,val in t.vars.items() -%}{{v}}={{val|round(3)}}, {% endfor %}
  Δ={{t.diff|round(3)}}
{% endfor %}
{{loss_name}}={{fitness|round(5)}}

Output (only these fields):
- reasoning: 2–3 short bullets (why this change). 
- mutated_term: final LISP expression

Prefer minimal edits; keep syntax valid.
"""

# @dataclass
# class MutationPositionExecution:
#     reasoning: Annotated[str, "Short explanation why this subexpression was selected (e.g., contributes most to error)"]
#     position: Annotated[str, "The select LISP subexpression."]
#     mutation_type: Annotated[str, "Type of mutation applied"]

# @dataclass
# class MutationExecution:
#     all_selected_positions: Annotated[list[MutationPositionExecution], "List of positions selected for mutation."]
#     final_mutated_term: Annotated[str, "Final LISP expression after applying the mutations to the selected positions."]

@dataclass
class MutationExecution:
    reasoning: list[str]
    mutated_term: str 

class LLMMutation(TermMutation):

    def __init__(self, *,     
                 llm: LLMCaller,
                 syntax: Syntax,
                 var_bindings: dict[str, torch.Tensor],
                 evaluator: Evaluator,
                 fitness: Fitness,
                 semantics: Semantics,
                 optimizer: Optimizer,
                 target: torch.Tensor,
                 add_metrics: Callable,
                #  prompt_template_path: str = "",
                 prompt_template: str = tests_prompt,
                 op_desc: dict[str, str] = ops_descriptions,
                 max_num_examples: int = 3,
                 max_num_tests: int = 3,
                 max_num_demo_tests: int = 2,
                 loss_name: str = "NMSE",
                 **kwargs):
        super().__init__(**kwargs)
        self.max_num_examples = max_num_examples
        self.max_num_tests = max_num_tests
        self.max_num_demo_tests = max_num_demo_tests
        self.loss_name = loss_name
        from jinja2 import Template
        self.prompt_template: Template = Template(prompt_template)
        self.llm = llm
        self.op_desc = op_desc        
        self.good_mutations: list[GoodMutationContext] = []
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.target = target
        self.syntax = syntax
        self.var_bindings = var_bindings
        self.add_metrics = add_metrics

    def select_exemplars(self, term: str):
        ''' Currently we pick just last N good mutations 
            TODO: consider similarity, diversity metrics and NMSE improvement
                  (article on salient aspects)
        '''
        if len(self.good_mutations) <= self.max_num_examples:
            return self.good_mutations
        rand_good_mutations = self.rnd.choice(
            self.good_mutations, 
            size=self.max_num_examples, 
            replace=False
        )
        return rand_good_mutations

    def mutate_term(self, term: Term) -> Term | None:
        '''
            Note that term may have inf or nan in outputs - invalid term,
            this should be handled before calling this method.
        '''

        outcomes = self.evaluator.eval(term, ensure_dims=True)
        fitness = self.fitness.get_fitness(term)

        # selecting hardest tests from this term 
        outcome_diffs = torch.abs(self.target - outcomes)
        test_ids = torch.argsort(outcome_diffs)
        hardest_test_ids = test_ids[-self.max_num_tests:]
        del test_ids
        expected = self.target[hardest_test_ids].tolist()
        actual = outcomes[hardest_test_ids].tolist()
        var_values = {var_name:var_vals[hardest_test_ids].tolist() for var_name, var_vals in self.var_bindings}
        tests = [TestCaseContext(vars = {x: var_values[x][i] for x in self.var_bindings.keys()},
                                  expected = expected[i],
                                  actual = actual[i],
                                  diff = actual[i] - expected[i]
                                  ) for i in range(len(hardest_test_ids))]

        demonstrations = self.select_exemplars(term)
        context = PromptContext(
            term = term_to_const_skeleton_str(term),
            tests = tests,
            fitness = fitness,
            op_desc = self.op_desc,
            free_vars=[x.var_id for x in vars],
            loss_name=self.loss_name,
            previous_good_mutations = demonstrations
        )
        
        try:
            prompt = self.prompt_template.render(**asdict(context)) # prompt rendering 
        except Exception as e:
            print("Error rendering prompt:", e)
            self.add_metrics(render_error=1)
            return None

        try:
            response, usage = self.llm(prompt, MutationExecution)
            self.add_metrics(**usage)
        except Exception as e:
            print("Error during LLM prompting:", e)
            self.add_metrics(llm_error=1)
            return None

        new_term = self.syntax.parse_const_skeleton(response.mutated_term)
        if new_term is None:
            self.add_metrics(llm_syn_invalid=1)
            return None
        if not self.syntax.is_valid(new_term):
            self.add_metrics(llm_constr_invalid=1)
            return None
        
        optimized_term = self.optimizer.optimize(new_term)
        
        new_outcomes = self.evaluator.eval(optimized_term, ensure_dims=True)
        new_fitness = self.fitness.get_fitness(optimized_term)

        good_mutations=1 if new_fitness < fitness else 0, 
        bad_mutations=1 if new_fitness > fitness else 0,
        neutral_mutations=1 if new_fitness == fitness else 0

        self.add_metrics(
            good_mutations=good_mutations,
            bad_mutations=bad_mutations,
            neutral_mutations=neutral_mutations
        )

        if good_mutations == 1:
            outcome_diffs_after = torch.abs(self.target - new_outcomes)
            outcome_improvements = outcome_diffs - outcome_diffs_after
            test_ids = torch.argsort(outcome_improvements)
            best_test_ids = test_ids[-self.max_num_demo_tests:]
            best_test_id_ids, = torch.where(outcome_improvements[best_test_ids] > 0)
            final_best_test_ids = best_test_ids[best_test_id_ids]
            best_expected = self.target[final_best_test_ids].tolist()
            best_actual = outcomes[final_best_test_ids].tolist()
            best_actual_after = new_outcomes[final_best_test_ids].tolist()
            best_free_vars = {var_name:var_values[final_best_test_ids].tolist() for var_name, var_values in self.var_bindings.items()}

            tests = [TestCaseContext(vars={var_name: best_free_vars[var_name][i] for var_name in self.var_bindings.keys()},
                                        expected=best_expected[i],
                                        actual=best_actual[i],
                                        diff=best_actual[i] - best_expected[i],
                                        actual_after=best_actual_after[i],
                                        diff_after=best_actual_after[i] - best_expected[i]
                                        )
                        for i in final_best_test_ids]
            good_mutation = GoodMutationContext(
                term = context.term, 
                term_after = term_to_const_skeleton_str(optimized_term),
                fitness_diff = fitness - new_fitness,
                tests = tests
            )
            self.good_mutations.append(good_mutation)
        
        return optimized_term 