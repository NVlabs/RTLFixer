from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection, func_impl_prepare_prompt, self_reflection_prepare_prompt

from typing import Optional, List, Union
import ast
import re
from .parse import parse_code_block, add_code_block


USE_VERILOG_CODEBLOCK_INSTRUCTION = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."

VG_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
VG_REFLEXION_COMPLETION_INSTRUCTION = ""
VG_SELF_REFLECTION_COMPLETION_INSTRUCTION = ""

VG_SIMPLE_CHAT_INSTRUCTION = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
VG_SIMPLE_FEW_SHOT = (
    ('user', """This Verilog module is a simple multiplexer. It takes two inputs, a and b, as well as a selector input, sel. It then outputs the value of either a or b, depending on the value of sel. If sel is 1, the output will be b, and if sel is 0, the output will be a. This module is useful for selecting between two different values, depending on the value of the selector. Implement the above description in the following module.

module top_module (
	input a,
	input b,
	input sel,
	output out
);
"""),
    ('assistant', """
assign out = sel ? b : a;

endmodule
"""),
    ('user', """
Build a circuit that always outputs a LOW.

module top_module(
output zero);
"""),
    ('assistant', """
assign zero = 0;
endmodule
""")
)

VG_REFLEXION_CHAT_INSTRUCTION = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions. You will be given your past implementation, compiler logs, and test results in order to change the implementation appropriately." #+ "If you cannot give a propriate solution right away, you can add more $display and $dumpfile to gather necessary information to achieve the correct solution."
VG_REFLEXION_FEW_SHOT_ADD = '''
Example 1:
[problem description]:
Create a 2-1 multiplexer. When sel=0, choose a. When sel=1, choose b.

[previous implementation]:
```
assign out = a & b & sel;

endmodule
```


[test results from previous implementation]:
VCD info: dumpfile wave.vcd opened for output.
test.sv:49: $finish called at 570 (1ps)
Hint: Output 'out' has 114 mismatches. First mismatch occurred at time 5.
Hint: Total mismatched samples is 114 out of 114 samples

Simulation finished at 570 ps
Mismatches: 114 in 114 samples

[reflection on previous impl]:
The previous implementation used a bitwise AND operation between 'a', 'b', and 'sel'. This results in an incorrect behavior since the output 'out' will never equal 'a' or 'b' directly. Instead, it'll produce a value based on individual bits of 'a', 'b', and 'sel'. The multiplexer is supposed to select either 'a' or 'b' based on 'sel', not perform a bitwise operation among them.

[improved implementation]:
```
assign out = sel ? b : a;

endmodule
```
END OF EXAMPLES
'''
VG_REFLEXION_FEW_SHOT = ''''''

VG_SELF_REFLECTION_CHAT_INSTRUCTION = "Your goal is to find out the error in the verilog code. Figure out how to fix the code error and tell me how to fix this error as detail as possible step by step."
VG_SELF_REFLECTION_FEW_SHOT = """
Example 1:
[problem description]:
Build a circuit that always outputs a LOW.


[module implementation]:
```
module top_module(
output zero);

endmodule
```


[simulation results]:
VCD info: dumpfile wave.vcd opened for output.
zero2.sv:37: $finish called at 102 (1ps)
Hint: Output 'zero' has 20 mismatches. First mismatch occurred at time 5.
Hint: Total mismatched samples is 20 out of 20 samples
Simulation finished at 102 ps\nMismatches: 20 in 20 samples.


[self-reflection]:
The implementation is wrong because:
Based on the problem description, you are tasked with building a circuit that always outputs a LOW signal. However, the provided Verilog module top_module does not meet this requirement, as indicated by the simulation results.
The simulation results show that there are mismatches between the expected behavior (always output LOW) and the actual behavior of the module, resulting in a total of 20 mismatches in 20 samples. This indicates that the module is not behaving as expected.
To analyze why the implementation is wrong step by step:
Module Declaration: The Verilog module top_module is declared with an output port named zero. This suggests that the module should produce a signal called zero.
Expected Behavior: The problem description specifies that the circuit should always output a LOW signal. This means that the zero output should be consistently LOW, regardless of any input or clock.
Simulation Results: The simulation results show that there are mismatches between the expected LOW output and the actual behavior of the module. The mismatches occur at various time points, and they indicate that the zero signal is not consistently LOW.
To identify why the implementation is wrong, you should examine the code of the top_module in more detail and look for any factors that might cause it to produce unexpected output. Common issues that could lead to incorrect behavior include uninitialized signals, unintentional clocking, or incorrect logic in the module.
Without the full code of the module and additional context, it's challenging to pinpoint the exact issue. However, you should carefully review the module's logic and ensure that it consistently produces a LOW output as required by the problem description.



Example 2:
[problem description]:
The module has 3 inputs. y input is a 4 bits input, w is a 1 bit input, Y2 is a 1 bit reg output.\nIt is an always_comb block, not an always_ff block.
So there is no state transitions and clock.
Every time a new input comes, the output will be changed immediately.
It is a case statement, the left hand side is a combination of inputs(y, w).
The right hand side is the output(Y2).\nIf the input is 4'h0, the output will be 0.
If the input is 4'h1, the output will be 0.
If the input is 4'h2, the output will be 1.
If the input is 4'h3, the output will be 1.
If the input is 4'h4, the output will be 0.
If the input is 4'h5, the output will be 1.
If the input is 4'h6, the output will be 0.
If the input is 4'h7, the output will be 0.
If the input is 4'h8, the output will be 0.
If the input is 4'h9, the output will be 1.
If the input is 4'ha, the output will be 1.
If the input is 4'hb, the output will be 1.


[module implementation]:
```
module top_module(
	input [3:1] y,
	input w,
	output reg Y2);

always_comb begin
    case(y)
        4'h0: Y2 = 1;
        4'h1: Y2 = 1;
        4'h2: Y2 = 0;
        4'h3: Y2 = 0;
        4'h4: Y2 = 1;
        4'h5: Y2 = 0;
        4'h6: Y2 = 1;
        4'h7: Y2 = 1;
        4'h8: Y2 = 1;
        4'h9: Y2 = 0;
        4'ha: Y2 = 0;
        4'hb: Y2 = 0;
        default: Y2 = 0; // It's a good practice to always have a default case
    endcase
end

endmodule
```


[simulation results]:
VCD info: dumpfile wave.vcd opened for output.
test.sv:41: $finish called at 501 (1ps)
Hint: Output 'Y2' has 36 mismatches. First mismatch occurred at time 15.
Hint: Total mismatched samples is 36 out of 100 samples


[self-reflection]:
The provided implementation has incorrect mappings between the 'y' input values and the 'Y2' output values. This is evident as there's a discrepancy between the expected and given output values for each 'y' input case in the always_comb block. Consequently, the simulation reports 36 mismatches out of 100 samples, indicating that the current logic doesn't align with the desired behavior.


END OF EXAMPLES
"""

VG_TEST_GENERATION_FEW_SHOT = """"""
VG_TEST_GENERATION_COMPLETION_INSTRUCTION = f""""""
VG_TEST_GENERATION_CHAT_INSTRUCTION = """"""


class VerilogGenerator(Generator):
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=VG_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=VG_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "verilog"),
            self_reflection_few_shot=VG_SELF_REFLECTION_FEW_SHOT
        )
    
    def prepare_prompt(
        self,
        func_sig: dict,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        
        if "react" in strategy:
            return self_reflection_prepare_prompt(
                func=func_sig,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=VG_SELF_REFLECTION_CHAT_INSTRUCTION,
                self_reflection_completion_instruction=VG_SELF_REFLECTION_COMPLETION_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "verilog"),
                self_reflection_few_shot=VG_SELF_REFLECTION_FEW_SHOT
            )
        else:
            return func_impl_prepare_prompt(
                func_sig=func_sig,
                model=model,
                strategy=strategy,
                prev_func_impl=prev_func_impl,
                feedback=feedback,
                self_reflection=self_reflection,
                num_comps=num_comps,
                temperature=temperature,
                reflexion_chat_instruction=VG_REFLEXION_CHAT_INSTRUCTION,
                reflexion_few_shot=VG_REFLEXION_FEW_SHOT_ADD,
                simple_chat_instruction=VG_SIMPLE_CHAT_INSTRUCTION,
                reflexion_completion_instruction=VG_REFLEXION_COMPLETION_INSTRUCTION,
                simple_completion_instruction=VG_SIMPLE_COMPLETION_INSTRUCTION,
                code_block_instruction=USE_VERILOG_CODEBLOCK_INSTRUCTION,
                parse_code_block=lambda x: parse_code_block(x, "verilog"),
                add_code_block=lambda x: add_code_block(x, "verilog"),
                lang="verilog"
            )

    def func_impl(
        self,
        func_sig: dict,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        return generic_generate_func_impl(
            func_sig=func_sig,
            model=model,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=VG_REFLEXION_CHAT_INSTRUCTION,
            reflexion_few_shot=VG_REFLEXION_FEW_SHOT_ADD,
            simple_chat_instruction=VG_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=VG_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=VG_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_VERILOG_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "verilog"),
            add_code_block=lambda x: add_code_block(x, "verilog"),
            lang="verilog"
        )

    def internal_tests(self, func_sig: dict, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]
        """
        Generates tests for a function.
        """
        return func_sig['test']
        
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=VG_TEST_GENERATION_FEW_SHOT,
            test_generation_chat_instruction=VG_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_completion_instruction=VG_TEST_GENERATION_COMPLETION_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=vg_is_syntax_valid,
        )


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


def vg_is_syntax_valid(code: str) -> bool:
    try:
        from threading import Timer
        cmd = "iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o test.vvp {}.sv"
        """
        adding timeout options for Popen. something breaks if not using timeout. seems to be working for now.
        not really sure if its the best/correct way. let me know if anyone has a better solution.
        https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
        """
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timer = Timer(timeout, p.kill)
        try:
            timer.start()
            out, err = p.communicate()
        finally:
            timer.cancel()
        out, err = out.decode("utf-8"), err.decode("utf-8") 
        if len(out) > 0 or len(err) > 0:
            return False
        return True
    except Exception:
        return False
