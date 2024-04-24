import re


def timeout_handler(_, __):
    raise TimeoutError()

import os, json
def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)

from threading import Thread
class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
    
    
def simple_syntax_fixer(code_completion: str, problem: dict = {}):
    
    try:
        # parse code block
        code_completion = parse_markdown_code_block(code_completion)

        # add endmodule
        if code_completion and 'endmodule' not in [i for i in code_completion.strip().split('\n') if i][-1]:
            code_completion += "\nendmodule"
#         elif not code_completion.strip().endswith('endmodule'):
#             code_completion += "\nendmodule"

        # remove lines
        impl = []
        for i in code_completion.split('\n'):
            i = i.strip()
            if not i:
                continue
    #         elif i.startswith('//'):
    #             continue
            elif 'timescale' in i:
                continue
            impl.append(i.strip())
        code_completion = "\n".join(impl)
        
    
        
        # add module header
        if 'module top_module' not in code_completion:
            code_completion = problem['prompt'] + '\n' + code_completion

        # force correct module header
        elif re.sub('\s+', '', problem['prompt']) not in re.sub('\s+', '', code_completion) and 'top_module' in code_completion and 'full_module' not in code_completion:
            st = None
            ed = None
            for e, line in enumerate(code_completion.split('\n')):
                
                if 'top_module' in line:
                    st = e
                elif ');' in line:
                    ed = e+1
                    break
            code_completion = "\n".join(code_completion.split('\n')[:st] + problem['prompt'].split('\n') + code_completion.split('\n')[ed:])
            

        
    #     if 'clk' not in problem['prompt'] and 'clk' in code_completion:
    #         code_completion = code_completion.replace('posedge clk', '*')
    
    except Exception:
        import traceback
        print(traceback.format_exc())

    return code_completion


def parse_markdown_code_block(text: str, ext: str = 'verilog'):
    try:
        cleaned_output = text.strip()
        if f"```{ext}" in cleaned_output:
            _, cleaned_output = cleaned_output.split(f"```{ext}")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith(f"```{ext}"):
            cleaned_output = cleaned_output[len(f"```{ext}"):]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```"):]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        return cleaned_output.strip()
    except Exception:
        return text
    

def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]
    
# Py tests

# if __name__ == "__main__":
#     formatter = PySubmissionFormatter()
#     leetcode_1 = 'class Solution:\n    def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n        '
#     humaneval_1 = 'def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n'

#     assert leetcode_1 == formatter.to_leetcode(humaneval_1)
#     assert humaneval_1 == formatter.to_humaneval(leetcode_1)




