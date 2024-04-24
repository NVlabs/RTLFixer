from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory, agent_factory
from verilog_eval.evaluation import estimate_pass_at_k
from typing import List
from copy import deepcopy
from executors.executor_utils import simple_syntax_fixer


def react_fix_compile(
        dataset: List[dict],
        model_name: str,
        language: str,
        max_iters: int,
        max_budgets: int,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        agent_feedback: str = None,
        num_samples: int = 10,
        compiler: str = ""
    ) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    method = agent_feedback
    code_agent = agent_factory('rtlfixer_code', model_name, exe=exe, max_iters=10, toolset=['compiler'], method=method, compiler=compiler)
    fix_agent = agent_factory('rtlfixer_fix', model_name, exe=exe, max_iters=20, toolset=['examine'])
    

    print_v = make_printv(verbose)
    num_correct_list = []
    num_samples_list = []
    
    for i, item in enumerate_resume(dataset, log_path):


        num_correct = 0
        num_budgets = 0
        org_item = deepcopy(item)
            
        for j in range(num_samples):
            
            # init
            is_solved = False
            item = org_item
            func_impl = None
            
            reflections = []
            implementations = []
            test_feedback = []
            history = []

            code_agent.toolkit.initialize(item)
            fix_agent.toolkit.initialize(item)
            code_agent.reset_logs()
            fix_agent.reset_logs()
            
#             while code_agent.toolkit.num_simulat < max_budgets:
            for _ in range(1):
        
                func_impl = simple_syntax_fixer(item['solution'], item)
                exec_result = exe.evaluate(
                    item if language == "verilog" else item['entry_point'],
                    func_impl,
                    item["test"],
                    timeout = 20,
#                     compile_only=compiler
                )
                compile_log = exec_result['feedback']['compiler_log']
                feedback = exec_result['feedback']
                
                if not exec_result['passed']:
                
                    # code generation
                    code_prompt = gen.prepare_prompt(
                        item if language == "verilog" else item['prompt'],
                        model,
                        "dual_agent_code",
                        func_impl,
                        test_feedback,
                        reflections,
                    )

                    import dataclasses
                    from typing import List, Union, Optional, Literal
                    MessageRole = Literal["system", "user", "assistant"]
                    @dataclasses.dataclass()
                    class Message():
                        role: MessageRole
                        content: str

                    code_prompt.append(Message(
                        role="assistant",
                        content=func_impl,
                    ))
                    code_prompt.append(Message(
                        role="user",
                        content="Run the compiler and fix the error if encountered.",
                    ))
                    
                    func_impl = code_agent.generate_chat(code_prompt)
                    func_impl = simple_syntax_fixer(func_impl, item)

                

                # simulate
                assert isinstance(func_impl, str)
                exec_result = exe.evaluate(
                    item if language == "verilog" else item['entry_point'],
                    func_impl,
                    item["test"],
                    timeout = 20 if is_leetcode else 20,
#                     compile_only=compiler
                )
                if 'wave.vcd' in exec_result:
                    exec_result.pop('wave.vcd')
                if 'test_output' in exec_result:
                    exec_result.pop('test_output')

                if exec_result['passed']:
                    is_solved = True
                    num_correct += 1


                
            import pdb
            pdb.set_trace()
#             item['result'] = result
            item["solution"] = func_impl
#             item["reflections"] = reflections
#             item["implementations"] = implementations
            item["feedback"] = exec_result['feedback']
            item["is_solved"] = is_solved
            item["num_simulat"] = code_agent.toolkit.num_simulat 
            item["num_compile"] = code_agent.toolkit.num_compile 
            item['agent_history'] = code_agent.serialize(history)
            item['error_logs'] = code_agent.serialize({
                'code': code_agent.error_logs,
                'fix': fix_agent.error_logs
            })
            write_jsonl(log_path, [item], append=True)
            num_budgets += code_agent.toolkit.num_simulat

        num_correct_list.append(num_correct)
        num_samples_list.append(num_budgets)

    print(f"pass@{pass_at_k}: {estimate_pass_at_k(num_samples_list, num_correct_list, pass_at_k)}")
