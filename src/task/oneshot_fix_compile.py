import numpy as np
from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory, agent_factory
from verilog_eval.evaluation import estimate_pass_at_k
from typing import List
from copy import deepcopy
from generators.model import Message
from generators.agents.langchain_tools import RAGTool, LocalizeTool
from executors.executor_utils import simple_syntax_fixer
from tqdm import tqdm


def oneshot_fix_compile(
        dataset: List[dict],
        model_name: str,
        language: str,
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
    

    print_v = make_printv(verbose)
    num_correct_list = []
    num_samples_list = []
    
    for i, item in tqdm(enumerate_resume(dataset, log_path), total=len(dataset)):

        num_correct = 0
        num_budgets = 0
        org_item = deepcopy(item)
            
        for j in tqdm(range(num_samples)):
            
            # init
            is_solved = False
            item = org_item
            func_impl = simple_syntax_fixer(item['solution'], item)
            
            exec_result = exe.evaluate(
                item if language == "verilog" else item['entry_point'],
                func_impl,
                item["test"],
                timeout = 20,
#                 compile_only=compiler
            )
            compile_log = exec_result['feedback']['compiler_log']
            feedback = exec_result['feedback']
            msg = ""
            numbered_impl = ""
            rag = ""
            output = ""
            
            if not exec_result['passed']:
                
                code_prompt = gen.prepare_prompt(
                    item if language == "verilog" else item['prompt'],
                    model,
                    "simple",
                    func_impl,
                )
                    
                code_prompt.append(Message(
                    role="assistant",
                    content=func_impl,
                ))
                

                if 'rag' in agent_feedback:
                    rag = RAGTool(compiler)
                    rag = rag._run(compile_log)

                    
                if "nofeedback" in agent_feedback or ('dangling input' in compile_log and not rag):
                    msg = "There is an error in the code that was cause compile error."
                else:
                    compile_log = compile_log.replace('I give up.', '')
                    msg = f"{numbered_impl}compile result: {compile_log}\n{rag}\n"

                code_prompt.append(Message(
                    role="user",
                    content=f"{msg}\nFix the error and give me the correct code.",
                ))
                
                output = model.generate_chat(code_prompt)
                func_impl = simple_syntax_fixer(output, item)
                    
            # simulate
            assert isinstance(func_impl, str)
            exec_result = exe.evaluate(
                item if language == "verilog" else item['entry_point'],
                func_impl,
                item["test"],
                timeout = 20,
                compile_only=compiler
            )
            if exec_result['passed']:
                is_solved = True
                num_correct += 1
                
            item['output'] = output
            item['reflexion'] = msg
            item["solution"] = func_impl
            item["feedbacks"] = exec_result['feedback']['compiler_log']
            item["test_output"] = exec_result['feedback']['test_output']
            item['completion'] = exec_result['feedback']['completion']
            item["is_solved"] = is_solved
            write_jsonl(log_path, [item], append=True)

        num_correct_list.append(num_correct)
        num_samples_list.append(num_samples)

    pass_k_list = estimate_pass_at_k(num_samples_list, num_correct_list, pass_at_k)
    print(f"pass@{pass_at_k}: {np.mean(pass_k_list)} {pass_k_list}")
