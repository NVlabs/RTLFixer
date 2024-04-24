import json
from langchain.tools import tool
from langchain.tools import Tool, BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, List, Union, Type, Any, Optional
from copy import deepcopy
from visualize import tabular_via_callback, tabular_via_dataframe
from vcdvcd import VCDVCD
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils import verilog_compile_has_error
from executors.executor_utils import simple_syntax_fixer





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


class RAG(BaseModel):
    compile_error_message: str = Field(description="error message from the verilog compiler.")


class VerilogSimulation(BaseModel):
    code_completion: str = Field(description="valid verilog code that can complete the module in the given problem.")
        
        
class VerilogCompiler(BaseModel):
    code_completion: str = Field(description="valid verilog code that can complete the module in the given problem.")
        
        
class CodeRepair(BaseModel):
    corrected_code_block_prefix_with_line_number: str = Field(description="The code block after correction with the corresponding line number as prefix. Ex. 18 next_state = 3'b001;\n19 end else begin\n20 next_state = 3'b000;")


class SelfVerify(BaseModel):
    answer: str = Field(description="The answer to how to fix the code error.")
        

class ExpertOpinion(BaseModel):
    question: str = Field(description="The question to ask the expert.")


class ExamineFile(BaseModel):
    
    class FileEnum(str, Enum):
        testbench = 'testbench'
        waveform = 'waveform'
        module = 'module'
    
    file: Optional[str] = Field(description="Should be one of [waveform, testbench, module].")
        
        
class RAGTool(BaseTool):
    name = "error_lookup"
    description = """
    Look up the compile error for some solutions.
    """
    args_schema: Type[BaseModel] = RAG
        
    def __init__(self, *args, **kwargs):
#         if 'compiler' in kwargs:
#             self.compiler = kwargs.pop('compiler')
        super().__init__()
            

    def _run(self, compile_error_message: str, compiler: str='quartus'):
        if not compile_error_message:
            return ""

        extra_msg = []
        
        if compiler == 'iverilog':
            if 'Unable to bind wire/reg/memory' in compile_error_message:
                extra_msg.append("* replace 'posedge clk' with '*'")
            if 'dangling input port' in compile_error_message and '(clk) floating' in compile_error_message:
                if 'posedge clk' in compile_error_message:
                    extra_msg.append("* replace 'posedge clk' with '*'")
                else:
                    extra_msg.append("* remove clk from input")
            if 'not a valid l-value' in compile_error_message:
                extra_msg.append('* next_state is defined as a wire not reg, it is not a valid l-value. Use assign statements instead of always block if possible.')
    #         if not selff._verify_module_header(code_completion, self.test_suite['prompt']):
    #             extra_msg.append(f"* The implementation must be based on the following module header: {self.test_suite['prompt']}")
            if "invalid module instantiation" in compile_error_message.lower():
                extra_msg.append("* Implement all logic inside top_module. Do not instantiation module inside top_module. Please correct the code and run compile again.")
    #         if 'give up' in compile_error_message:
    #             extra_msg.append("* Ignore the previous implementation and create a new implementation from scratch.")
            if "Extra digits given for sized binary constant." in compile_error_message:
                extra_msg.append("* Please make sure that your binary numbers have exactly digits, consisting of 0s and 1s, as specified in the declaration (e.g., 9'b0000000001). Extra digits are not allowed.")
                
            if extra_msg:
                extra_msg = ['Fix suggestions:'] + extra_msg
            else:
                extra_msg = ["I should explain what the compile error is about. Then correct the code syntax in top_module and run compile again."]


        elif compiler == "quartus":
            if '10161' in compile_error_message:
                if 'clk' in compile_error_message:
                    extra_msg.append("Replace 'posedge clk' with '*'")
                else:
                    extra_msg.append("Rewrite all undeclared variables.")
            elif '10232' in compile_error_message:
                extra_msg.append("Use binary string to do the indexing instead of parameter.")
            elif 'z_next' in compile_error_message:
                extra_msg.append('Remove the z_next register')
            elif '10028' in compile_error_message:
                extra_msg.append('Combined the state machine and output logic into a single always block.')
            elif '10170' in compile_error_message:
                extra_msg.append('Rewrite all variables.')
                extra_msg.append('Remove duplicated endmodule at the end.')
            elif '12007' in compile_error_message:
                extra_msg.append('Top-level design entity is undefined.')
            elif '10029' in compile_error_message:
                extra_msg.append('Check if driver by two always blocks which is illegal for synthesis.')
            elif '10759' in compile_error_message:
                extra_msg.append('Check if multiple declaration on one variable.')
            elif '13069' in compile_error_message:
                extra_msg.append('Replace the int with integer.')
            elif '293007' in compile_error_message:
                extra_msg.append('Cannot reset block rams using a single initial block.')
            elif '275062' in compile_error_message:
                extra_msg.append('Find the error in the code.')
            elif '293001' in compile_error_message:
                extra_msg.append('Instance should be defined as a signal name or another logic function.')
            elif '20268' in compile_error_message:
                extra_msg.append('Functional simulation is off but it is the only supported netlist type for this device.')
            

        extra_msg = "\n".join(extra_msg)
        return extra_msg
    
    
class WaveformTabular():
    
    def _run(self, test_output: str, waveform: str):
              
        def parse_mismatch(test_output: str):
            mismatch = {}
            prefix = "First mismatch occurred at time"
            for line in test_output.split('\n'):
                if prefix in line:

                    # signal name
                    st = line.find("Output '")
                    ed = line.find("' ")
                    signal_name = line[st+8:ed]

                    # timestep
                    st = line.find(prefix)
                    mismatch_timestep = int(line[st+len(prefix):-1].strip())

                    mismatch[signal_name] = mismatch_timestep

            first_mismatch_timestep = min(mismatch.values())
            return list(mismatch.keys()), first_mismatch_timestep

        def get_tabular(method: str):
            vcd_path = 'tmp.vcd'
            with open(vcd_path, "w") as f:
                f.write(waveform)
                f.seek(0)

                mismatch_columns, offset = parse_mismatch(test_output)
                window_size = 20

                gen_func = {
                    'callback': tabular_via_callback,
                    'dataframe': tabular_via_dataframe,
                }.get(method)
                if gen_func is None:
                    raise Exception(f"get tabular do not support {method} method.")

                return gen_func(vcd_path, offset, mismatch_columns, window_size)

        tabular = get_tabular('dataframe')
        return tabular
    
    
class LocalizeTool(BaseTool):
    name = "localize_tool"
    description = ""
    
    def _run(self):
        pass
    
    def _numberize_impl(self, feedback: dict, winsize: int, task_id: str):
        numbered_impl = [f"{e:<5}{i}" for e, i in enumerate(feedback['verilog_test'].split('\n'))]
        error_lines = []
        prefix = f'{task_id}.sv:'
        for line in feedback['compiler_log'].split('\n'):
            if prefix in line:
                err_num = int(line[len(prefix):].split(':')[0])
                error_lines.append(err_num)

        st = max(min(error_lines) - winsize, 0)
        ed = max(error_lines) + winsize
        numbered_impl = "\n".join(numbered_impl[st: ed])
        return numbered_impl

    def _reorder_numberize_impl(self, code_completion: str, feedback: dict, task_id: str, winsize: int = 5):
        error_lines = []
        prefix1 = f'{task_id}.sv:'
        prefix2 = f'{task_id}2.sv:'
        
        
        for line in feedback['compiler_log'].split('\n'):
            if prefix1 in line:
                err_num = int(line[len(prefix1):].split(':')[0])
                error_lines.append(err_num)
            elif prefix2 in line:
                err_num = int(line[len(prefix2):].split(':')[0])
                error_lines.append(err_num)

        # reorder
        reorder_error_lines = []
        offset = len(feedback['verilog_test'].split('\n')) - len(code_completion.split('\n'))
        for i in list(set(error_lines)):
            new_num = (i - offset - 1)
            if new_num < 0:
                new_num = " "
            else:
                reorder_error_lines.append(new_num)
            feedback['compiler_log'] = feedback['compiler_log'].replace(str(i), str(new_num))

        st = None
        ed = None
        if reorder_error_lines:
            st = max(min(reorder_error_lines)-winsize, 0)
            ed = max(reorder_error_lines) + winsize
            
            ed = min(ed, st + 20)
        
        numbered_impl = [f"{e:<5}{i}" for e, i in enumerate(code_completion.split('\n'))]
        numbered_impl = "\n".join(numbered_impl[st:ed])
        return numbered_impl, feedback['compiler_log']


class VerilogToolkit():
    
    def __init__(self, exe, toolset = None, llm = None, method: str = "", compilername: str = ""):
        
        
        class ExpertTool(BaseTool):
            name = "get_verilog_expert_suggestions"
            description = """
                Use this tool to get suggestions from verilog expert.
                Three things are required: verilog module code, simulation results, waveform.
            """
            args_schema: Type[BaseModel] = ExpertOpinion
                
            def _run(selff, question: str):
                return "I have read the code, simulation result and waveform. I can figure out what the error is and how to modify the code."
                prompt = f"""
                What is the error in the code?
                ```
                {question}
                ```
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                answer = self.llm(messages)
                return f"{answer.content}\nNow try to fix the code accordingly."
            
        
        class CheckCorrectnessTool(BaseTool):
            name = "identify_error"
            description = """
            This tool can help to check if there is any potential issue in the code.
            """
            args_schema: Type[BaseModel] = VerilogSimulation
                
            def _run(selff, code_completion: str):
                
                code_completion = simple_syntax_fixer(code_completion, self.test_suite)
                
                prompt = f"""
                Problem description:
                {self.test_suite['detail_description']}
                
                Implementation:
                ```
                {code_completion}
                ```
                """
                
                if 'feedback' in self.test_suite:
                    
                    if 'wave.vcd' in self.test_suite['feedback']:
                        prompt += f"""
                        Simulation result:
                        {self.test_suite['feedback']['test_output']}
                        
                        Waveform:
                        {WaveformTabular()._run(self.test_suite['feedback']['test_output'], self.test_suite['feedback']['wave.vcd'])}
                        """
                    else:
                        prompt += f"""
                        Compile result:
                        {self.test_suite['feedback']['compiler_log']}
                        """
                
                prompt += "Explain the problem in the code and show me how to fix it."
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                answer = self.llm(messages)
                return answer.content + '\nReturn the explanation and solution to the user.'
        
        
        class VerifyTool(BaseTool):
            name = "verify"
            description = """
                Use this tool to verify if a fix to the code error is provided.
            """
            args_schema: Type[BaseModel] = SelfVerify
                
            def _run(selff, prompt: str, fail_msg: str):
                assert self.llm is not None
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                self_verify = self.llm(messages)
                if "YES" in self_verify.content.upper():
                    return "Pass verification."
                else:
                    return fail_msg
                
        class CheckWaveform(BaseTool):
            name = "check_waveform"
            description = """
                Use this tool to get the simulation waveform.
                """
            def _run(selff):
                return self.examine._run('waveform')
            
        class CheckImplementation(BaseTool):
            name = "check_implementation"
            description = """
                Use this tool to get the verilog module code implementation.
                """
            def _run(selff):
                return self.examine._run('module')
                
        
        class ExamineFileTool(BaseTool):
            name = "examine"
            description = """
            Use this tool to inspect the current implementation. Specifiy which file you need to examine.
            testbench: the testbench for simulation.
            waveform: the signal waveform where mismatch occurs during the simulation.
            module: Current code implementation for the problem.
            """
            args_schema: Type[BaseModel] = ExamineFile
                
            def _run(selff, file: str):
                try:
                    if "testbench" in file:
                        return self.test_suite["test"]
                    elif "wave" in file:
                        
                        if 'feedback' not in self.test_suite:
                            return ""

                        if "test_output" not in self.test_suite['feedback']:
                            return "Run the verilog simulation first before examine files."

                        waveform = self.test_suite['feedback'].get('wave.vcd')

                        if waveform:
                            test_output = self.test_suite['feedback']["test_output"]

                            def parse_mismatch(test_output: str):
                                mismatch = {}
                                prefix = "First mismatch occurred at time"
                                for line in test_output.split('\n'):
                                    if prefix in line:

                                        # signal name
                                        st = line.find("Output '")
                                        ed = line.find("' ")
                                        signal_name = line[st+8:ed]

                                        # timestep
                                        st = line.find(prefix)
                                        mismatch_timestep = int(line[st+len(prefix):-1].strip())

                                        mismatch[signal_name] = mismatch_timestep

                                first_mismatch_timestep = min(mismatch.values())
                                return list(mismatch.keys()), first_mismatch_timestep

                            def get_tabular(method: str):
                                vcd_path = 'tmp.vcd'
                                with open(vcd_path, "w") as f:
                                    f.write(waveform)
                                    f.seek(0)

                                    mismatch_columns, offset = parse_mismatch(test_output)
                                    window_size = 20

                                    gen_func = {
                                        'callback': tabular_via_callback,
                                        'dataframe': tabular_via_dataframe,
                                    }.get(method)
                                    if gen_func is None:
                                        raise Exception(f"get tabular do not support {method} method.")

                                    return gen_func(vcd_path, offset, mismatch_columns, window_size)

                            tabular = get_tabular('dataframe')
                            return tabular
                        else:
                            return 'No waveform is dumped in this simulation. Add the dump statement in the code and run simulation again.'

                    elif 'module' in file:
                        return self.test_suite["solution"]
                    else:
                        return f"Cannot choose the file type {file}. Only choose from [testbench, waveform, module]"
                except Exception:
                    
                    import traceback
                    print(traceback.format_exc())
                    
                    import pdb
                    pdb.set_trace()
        
        class VerilogSimulatorTool(BaseTool):
            name = "verilog_simulator"
            description = """
                Input valid verilog code that can complete the module.
                Return the compile and test results.
                """
            args_schema: Type[BaseModel] = VerilogSimulation
            
            def _explain_error(selff, impl: str, error_log: str) -> str:
                prompt = f"""
                Explain this compiler error in detail and how to fix it.
                ```
                {impl}
                ```
                compile result:
                {error_log}
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                compile_error_explaination = self.llm(messages)
                return compile_error_explaination.content + '\n'

            def _run(selff, code_completion: str):
                code_completion = simple_syntax_fixer(code_completion)

                feedback = self.exe.evaluate(
                    self.test_suite,
                    code_completion,
                    self.test_suite["test"],
                    timeout = 20
                ).get('feedback')
                
                self.test_suite['solution'] = code_completion
                self.feedback = deepcopy(feedback)
                if 'wave.vcd' in feedback:
                    feedback.pop('wave.vcd')
                self.test_suite['feedback'] = feedback
                if feedback['test_output']:
                    self.num_simulat += 1
                if feedback['compiler_log']:
                    self.num_compile += 1
                feedback.pop('verilog_test')
                feedback.pop('verilog_test2')

                if feedback['passed']:
                    return f"feedback:{feedback}. Passed. Stop simulation and output implementation."
                elif verilog_compile_has_error(feedback['compiler_log']):
                    return f"feedback:{feedback}. I should correct the code syntax and run simulation again."
                else:
                    # no feedback
#                     out = "There are mismatches while running the simulation testbench. I should find the error and correct the code. Explain the code line by line and find the error."
                    
                    # react
                    # out = f"feedback:{feedback}. I should think about why the code has error. If I know how to fix it, run the simulation again with the fixed implementation. If not, I will check out the waveform to and find the issue. After I figure out what the problem is, I will run the verilog simulator again."

                    # explain
                    compile_error_explaination = selff._explain_error(numbered_impl, feedback['compiler_log'])
                    out = f"feedback:{feedback}. {compile_error_explaination} Correct the code and run the verilog simulator again."
                    return out
                    
                
                
        class VerilogCompilerTool(BaseTool):
            name = "verilog_compiler"
            description = """
                Input valid verilog code that can complete the module.
                Return the compile results.
                """
            args_schema: Type[BaseModel] = VerilogCompiler
            
            
            def _explain_error(selff, prev_impl: str, compiler_log: str) -> str:
                prompt = f"""
                Explain this compiler error in detail and how to fix it.
                ```
                {prev_impl}
                ```
                compile result:
                {compiler_log}
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                compile_error_explaination = self.llm(messages)
                return compile_error_explaination.content + '\n'
            
            def _explain_code_by_line(selff, impl: str) -> str:
                prompt = f"""
                Explain the code in the module line by line.
                ```
                {impl}
                ```
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                code_explaination = self.llm(messages)
                return code_explaination.content + '\n'
            
            def _verify_module_header(selff, impl: str, header: str):
                
                import re
                pattern=r"\s+"
                if re.sub(pattern, "", header) not in re.sub(pattern, "", impl):
                    
                    if 'Agent stopped due to iteration limit or time limit.' in impl:
                        return True
                    elif 'module ' not in impl:
                        return True
                    
                    return False
                else:
                    return True
                
                prompt = f"""
                    This is the defined module header:
                    ```
                    {header}
                    ```

                    This is your implementation:
                    ```
                    {impl}
                    ```

                    If the implementation has a module header but is different from the defined module header, answer no.
                    Otherwise, answer yes.
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                explanation = self.llm(messages).content
                print(explanation)
                if 'YES' in explanation.upper():
                    return True
                else:
                    return False
            
            def _match_impl_with_description(selff, impl: str, description: str):
                
                explaination = selff._explain_code_by_line(impl)
                
                prompt = f"""
                    Problem description:
                    ```
                    {description}
                    ```

                    Code implementation:
                    ```
                    {impl}
                    ```

                    The written code explained line by line:
                    ```
                    {explaination}
                    ```

                    If the implementation is aligned with the problem description, answer YES.
                    If not, explain how the code should be modified in order to be aligned with the problem description.
                """
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                explaination = self.llm(messages).content
                if 'YES' in explaination.upper():
                    return ""
                else:
                    return explaination + '\n'

            def _run(selff, code_completion: str):
                
                if not code_completion:
                    import pdb
                    pdb.set_trace()
                
                code_completion = simple_syntax_fixer(code_completion, self.test_suite)
                self.last_impl = code_completion
                
                feedback = self.exe.evaluate(
                    self.test_suite,
                    code_completion,
                    self.test_suite["test"],
                    timeout = 20,
                    compile_only='quartus'
                ).get('feedback')
                self.num_compile += 1
                
                if feedback['compiler_log']:
                    
                    if self.method == "nofeedback":
                        return "I should correct the code syntax if there are syntax error in the code."
                        # return f"compile result: {feedback['compiler_log']}\nI should correct the code syntax and run compile again."

                    numbered_impl = ""
                    compile_error_explaination = ""
                        
                    if "localize" in self.method:
                        localizer = LocalizeTool()
                        numbered_impl, feedback['compiler_log'] = localizer._reorder_numberize_impl(code_completion, feedback, self.test_suite['task_id'])
                    
                    if 'rag' in self.method:
                        extra_msg = self.rag._run(feedback['compiler_log'], self.compilername)
                        feedback['compiler_log'] += f"\n{extra_msg}\n"
                        
                    if 'explain' in self.method:
                        compile_error_explaination = selff._explain_error(numbered_impl, feedback['compiler_log'])
                        
                    if 'align' in self.method:
                        impl_align_explaination = selff._match_impl_with_description(code_completion, self.test_suite['detail_description'])
                    
                    
                    return f"{numbered_impl}\n\ncompile result: {feedback['compiler_log']}\n{compile_error_explaination}\nFix the code and run compile again."
                else:
#                     return f"Success. I should return this implementation."
                    return f"The code has no compile error. I should give this implementation to the user."
                
                
        class CodeRepairTool(BaseTool):
            name = "code_repair"
            description = """
                Input the verilog code block after correction with the corresponding line number as prefix.
                Ex.
                ```
                18 next_state = 3'b001;
                19 end else begin
                20 next_state = 3'b000;
                ```
                """
            args_schema: Type[BaseModel] = CodeRepair
                
            def _run(selff, corrected_code_block_prefix_with_line_number: str):
                
                try:
                    last_impl = self.last_impl.split('\n')
                    for line in corrected_code_block_prefix_with_line_number.split('\n'):
                        line_num, line = line.split(None, 1)
                        last_impl[int(line_num)] = line
                except Exception as e:
                    return str(e)

                self.last_impl = "\n".join(last_impl)
                return self.compiler._run(self.last_impl)
                

        self.llm = llm
        self.exe = exe
        self.method = method
        self.compilername = compilername
        self.test_suite = None
        self.toolset_map = {
            'simulator': VerilogSimulatorTool,
            'examine': ExamineFileTool,
            'compiler': VerilogCompilerTool,
            'verify': VerifyTool,
            'repair': CodeRepairTool,
            'waveform': CheckWaveform,
            'module': CheckImplementation,
            'expert': ExpertTool,
            'rag': RAGTool,
            'correctness': CheckCorrectnessTool
        }
        self.toolset = toolset if toolset is not None else list(self.toolset_map.keys())
        self.initialize(None)
        
    def initialize(self, item: dict):
        self.test_suite = item
        self.num_simulat = 0
        self.num_compile = 0
        self.last_impl = ""
        self.tools = [
            self.toolset_map[tool_class]() for tool_class in self.toolset
        ]
        self.verify = self.toolset_map['verify']()
        self.compiler = self.toolset_map['compiler']()
        self.simulator = self.toolset_map['simulator']()
        self.examine = self.toolset_map['examine']()
        self.rag = self.toolset_map['rag']()
        
    def set_test_suite(self, item: dict):
        self.initialize(item)
        
    @tool("verilog_simulator", return_direct=False, args_schema=VerilogSimulation)
    def simulate(code_completion: str):
        """Input the code completion and the simulator will return compile and test result."""
        return self.exe.evaluate(
            self.test_suite['entry_point'],
            code_completion,
            self.test_suite["test"],
            timeout = 20
        )