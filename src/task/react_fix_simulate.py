from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory, agent_factory
from verilog_eval.evaluation import estimate_pass_at_k
from typing import List, Tuple
from copy import deepcopy
import numpy as np
from executors.executor_utils import simple_syntax_fixer


def parse_mismatches(log: str) -> int:
    for line in log.split('\n'):
        if 'Mismatches' in line:
            st = len("Mismatches:")
            ed = line.find("in")
            return int(line[st: ed].strip())
    return 999


def restart_strategy(implementations: list, test_feedback: list, func_impl: str, patient: int = 3) -> Tuple[bool, str, str]:
    
    assert len(implementations) == len(test_feedback)
    if len(implementations) <= 1:
        return False, "First", func_impl
    
    if 'give up' in test_feedback[-1]['compiler_log']:
        return True, "compiler give up", ""
    
    miss_match_list = [parse_mismatches(i['test_output']) for i in test_feedback]
    
    # early stop
#     if len(implementations) >= patient:
#         # no improvment in rounds (equal ascending orders)
#         if miss_match_list[-patient:] == np.sort(miss_match_list[-patient:]).tolist():
#             return True, "patient", ""
        
    last_miss_match = miss_match_list[-1]
    best_idx = np.argmin(miss_match_list)
    best_miss_match = miss_match_list[best_idx]
    best_impl = implementations[best_idx]
    
    if last_miss_match < best_miss_match:
        # has improve
        return False, "improved", func_impl
    elif last_miss_match > best_miss_match:
        # has worsen
        return False, "worsen", best_impl
    else:
        return False, "no_change", func_impl
        


def react_fix_simulate(
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
        num_samples: int = 10
    ) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    nofeedback = True if 'nofeedback' in agent_feedback else False
    code_agent = agent_factory('rtlfixer_code', model_name, exe=exe, max_iters=20, toolset=['compiler'])
    fix_agent = agent_factory('rtlfixer_fix', model_name, exe=exe, max_iters=20, toolset=[])
    

    print_v = make_printv(verbose)
    num_correct_list = []
    num_samples_list = []
    
    for i, item in enumerate_resume(dataset, log_path):

        num_correct = 0
        num_budgets = 0
        org_item = deepcopy(item)
        
        code_agent.toolkit.initialize(item)
        fix_agent.toolkit.initialize(item)
            
        # for j in range(num_samples):
        while code_agent.toolkit.num_simulat < max_budgets:
            
            # init
            is_solved = False
            item = org_item
            func_impl = None
            
            reflections = []
            implementations = []
            test_feedback = []
            history = []

            code_agent.reset_logs()
            fix_agent.reset_logs()
            
            func_impl = item['solution']
            n_compile_error = 0
            n_simulate_error = 0

            while code_agent.toolkit.num_simulat < max_budgets:
                
                func_impl = simple_syntax_fixer(func_impl, item)
                
                if func_impl not in implementations:
                    
                    # simulate
                    assert isinstance(func_impl, str)
                    exec_result = exe.evaluate(
                        item if language == "verilog" else item['entry_point'],
                        func_impl,
                        item["test"],
                        timeout = 20 if is_leetcode else 20
                    )
                    if 'wave.vcd' in exec_result:
                        exec_result.pop('wave.vcd')
                    if 'test_output' in exec_result:
                        exec_result.pop('test_output')
                    
                    
                    code_agent.toolkit.num_simulat += 1
                    if hasattr(code_agent, 'intermediate_steps'):
                        history.append(code_agent.intermediate_steps)
                    implementations.append(func_impl)
                    test_feedback.append(exec_result['feedback'])
                    if exec_result['passed']:
                        is_solved = True
                        num_correct += 1
                        break
                    elif exec_result['feedback']['compiler_log']:
                        n_compile_error += 1
                    else:
                        n_simulate_error += 1
                    
                    # track impl and performance
                    restart, reason, func_impl = restart_strategy(implementations, test_feedback, func_impl)
                    if restart:
                        history.append({
                            'restart': {
                                'reason': reason
                            }
                        })
                        func_impl = ""
                    
                if func_impl:  # did not restart
                    # prepare for examine tool
                    item['feedback'] = exec_result['feedback']
                    item['solution'] = func_impl

                    # fix
                    if nofeedback:
                        err_msg = "There is an error during the simulation. Please check the logic and fix the code."
                    else:
                        err_msg = exec_result['feedback']['test_output'] + f"\nWaveform:\n{fix_agent.toolkit.examine._run('waveform')}\n"
                        
                    fix_prompt = gen.prepare_prompt(
                        item if language == "verilog" else item['prompt'],
                        model,
                        "react_fix",
                        func_impl,
                        err_msg,
                    )
                    reflexion = fix_agent.generate_chat(fix_prompt)
                    history.append(fix_agent.intermediate_steps)
                    reflections.append(reflexion)


                # code generation
                org_temperature = code_agent.llm.temperature
                while True:
                    code_prompt = gen.prepare_prompt(
                        item if language == "verilog" else item['prompt'],
                        model,
                        "react_code",
                        func_impl,
                        test_feedback,
                        reflections,
                    )
                    func_impl = code_agent.generate_chat(code_prompt)
                    if func_impl is None:
                        func_impl = implementations[-1]
                    if func_impl not in implementations:
                        code_agent.llm.temperature = org_temperature
                        break
                    else:
                        if code_agent.llm.temperature > 1.0:
                            func_impl = ""
                        else:
                            code_agent.llm.temperature += 0.1


            import pdb
            pdb.set_trace()
            item["solution"] = func_impl
            item["reflections"] = reflections
            item["implementations"] = implementations
            item["feedbacks"] = test_feedback
            item["is_solved"] = is_solved
            item["num_simulat"] = code_agent.toolkit.num_simulat 
            item["num_compile"] = code_agent.toolkit.num_compile
            item["n_compile_error"] = n_compile_error
            item["n_simulate_error"] = n_simulate_error
            item['agent_history'] = code_agent.serialize(history)
            item['error_logs'] = code_agent.serialize({
                'code': code_agent.error_logs,
                'fix': fix_agent.error_logs
            })
            write_jsonl(log_path, [item], append=True)
            num_budgets += code_agent.toolkit.num_simulat

        num_correct_list.append(num_correct)
        num_samples_list.append(num_budgets)

    pass_k_list = estimate_pass_at_k(num_samples_list, num_correct_list, pass_at_k)
    print(f"pass@{pass_at_k}: {np.mean(pass_k_list)} {pass_k_list}")
