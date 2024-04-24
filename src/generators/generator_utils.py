from generators.model import ModelBase, Message
import random

from typing import Union, List, Optional, Callable


def func_impl_prepare_prompt(
    func_sig: dict,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflexion_chat_instruction: str,
    reflexion_few_shot: str,
    simple_chat_instruction: str,
    reflexion_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str],
    simple_few_shot: str = "",
    lang: str = None,
) -> Union[str, List[str]]:
    prompt = ""
    org_func_sig = func_sig
    if isinstance(func_sig, dict):
        prompt = func_sig['prompt']
        if lang == "verilog":
            func_sig = f"{func_sig['detail_description']}\nImplement the above description in the following module.\n{func_sig['prompt']}"
        else:
            func_sig = func_sig['prompt']
    
#     if strategy != "reflexion" and strategy != "simple":
#         raise ValueError(
#             f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "reflexion":
            message = f"{reflexion_few_shot}\n[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}"
            print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
                Message(
                    role="user", # TODO: check this
                    content=reflexion_few_shot,
                ),
                Message(
                    role="assistant",
                    content=add_code_block(prev_func_impl),
                ),
                Message(
                    role="user",
                    content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                ),
                Message(
                    role="assistant",
                    content=self_reflection,
                ),
                Message(
                    role="user",
                    content=f"[improved impl]:\n{func_sig}",
                ),
            ]
        elif "rtlfixer" in strategy:
            if not prev_func_impl:
                system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}\nI want you to run the compiler to ensure the correctness of the syntax before answering.\n"
                print_messages(system_prompt, func_sig)
                messages = [
                    Message(
                        role="system",
                        content=system_prompt,
                    ),
                ] + [
                    Message(
                        role=role,
                        content=content,
                    )
                    for role, content in simple_few_shot
                ] + [
                    Message(
                        role="user",
                        content=f"{func_sig}",
                    ),
                ]
            else:
                feedback = feedback[-1]['test_output'] if feedback[-1]['test_output'] else feedback[-1]['compiler_log']
#                 system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}\nRun the compiler to ensure the correctness of the syntax.\n"
                system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}\n"

#                 message = f"{reflexion_few_shot}\n[previous implementation]:\n{prev_func_impl}\n\n[simulation results from previous implementation]:\n{feedback}\n\n[reflection on previous implementation]:\n{self_reflection}\n\n[improved implementation]:\n"
#                 print_messages(system_prompt, message)
#                 messages = [
#                     Message(
#                         role="system",
#                         content=system_prompt,
#                     ),
#                     Message(
#                         role="user",
#                         content=reflexion_few_shot,
#                     ),
#                     Message(
#                         role='user',
#                         content=f"[problem description]:\n{org_func_sig['detail_description']}\n[previous implementation]:"
#                     ),
#                     Message(
#                         role="assistant",
#                         content=prev_func_impl,
#                     ),
#                     Message(
#                         role="user",
#                         content=f"[simulation results from previous implementation]:\n{feedback}\n\n[reflection on previous implementation]:",
#                     ),
#                     Message(
#                         role="assistant",
#                         content=self_reflection,
#                     ),
#                     Message(
#                         role="user",
#                         content=f"[improved implementation]:\n",
#                     ),
#                 ]


                reflexion = self_reflection[-1] if self_reflection else ""
                messages = [
                    Message(
                        role="system",
                        content=system_prompt,
                    ),
                    Message(
                        role='user',
                        content=f"{func_sig}"
                    ),
                    Message(
                        role="assistant",
                        content=prev_func_impl,
                    ),
                    Message(
                        role="user",
                        content=f"{feedback}\n{reflexion}\n{func_sig}",
                    ),
                ]
                print_messages(system_prompt, "\n".join([i.content for i in messages]))
        else:
            if lang == "verilog":
                system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}\n"
                print_messages(system_prompt, func_sig)
                messages = [
                    Message(
                        role="system",
                        content=system_prompt,
                    ),
                ] + [
                    Message(
                        role=role,
                        content=content,
                    )
                    for role, content in simple_few_shot
                ] + [
                    Message(
                        role="user",
                        content=f"{func_sig}",
                    ),
                ]
            
#                 system_prompt = f"{simple_chat_instruction}"
#                 print_messages(system_prompt, func_sig)
#                 messages = [
#                     Message(
#                         role="system",
#                         content=system_prompt,
#                     ),
#                     Message(
#                         role="user",
#                         content=f"{code_block_instruction}\n// {func_sig}",
#                     ),
#                 ]

#                 system_prompt = f"{simple_chat_instruction}"
#                 print_messages(system_prompt, func_sig)
#                 messages = [
#                     Message(
#                         role="system",
#                         content=f"{simple_chat_instruction}\n",
#                     ),
#                     Message(
#                         role="user",
#                         content=code_block_instruction,
#                     ),
#                     Message(
#                         role="assistant",
#                         content=func_sig,
#                     ),
#                 ]
            else:
                system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
                print_messages(system_prompt, func_sig)
                messages = [
                    Message(
                        role="system",
                        content=f"{simple_chat_instruction}\n{code_block_instruction}",
                    ),
                    Message(
                        role="user",
                        content=func_sig,
                    ),
                ]
        
        return messages
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
        return prompt
    


def generic_generate_func_impl(
    func_sig: dict,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflexion_chat_instruction: str,
    reflexion_few_shot: str,
    simple_chat_instruction: str,
    reflexion_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str],
    simple_few_shot: str = "",
    lang: str = None,
) -> Union[str, List[str]]:
    
    prompt = ""
    if isinstance(func_sig, dict):
        prompt = func_sig['prompt']
        if lang == "verilog":
            func_sig = f"{func_sig['detail_description']}\n{func_sig['prompt']}"
        else:
            func_sig = func_sig['prompt']
    
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")
        
    prompt = func_impl_prepare_prompt(
        func_sig,
        model,
        strategy,
        prev_func_impl,
        feedback,
        self_reflection,
        num_comps,
        temperature,
        reflexion_chat_instruction,
        reflexion_few_shot,
        simple_chat_instruction,
        reflexion_completion_instruction,
        simple_completion_instruction,
        code_block_instruction,
        parse_code_block,
        add_code_block,
        simple_few_shot,
        lang,
    )
    
    if model.is_chat:
        func_bodies = model.generate_chat(messages=prompt, num_comps=num_comps, temperature=temperature)
    else:
        func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        func_body_str = parse_code_block(func_bodies)
        # func_body_str = func_body_str.replace(prompt, "")  # completion only
        if 'endmodule' not in func_body_str:
            func_body_str += '\nendmodule'
        print_generated_func_body(func_body_str)
        return func_body_str
    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies


def generic_generate_internal_tests(
        func_sig: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
            print(f'React test generation output: {output}')
        else:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[unit tests]:",
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = model.generate(prompt, max_tokens=1024)
    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)


def self_reflection_prepare_prompt(
    func: str,
    feedback: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str] = None,
    lang: str = "verilog"
) -> str:
    
    if isinstance(func, dict):
        if lang == "verilog":
            description = func['detail_description']
            func = func['solution']
        else:
            description = ""
            func = func['prompt']
    
    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
#                     content=f'{self_reflection_few_shot}\n\n[problem description]:\n{description}\n\n[module implementation]:\n{add_code_block(func)}\n\n[simulation results]:\n{feedback}\n\n[self-reflection]:',
                    content=f'[problem description]:\n{description}\n\n[module implementation]:\n{add_code_block(func)}\n\n[simulation results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            print_messages(self_reflection_chat_instruction, "\n".join([i.content for i in messages[1:]]))
        else:
            messages = [
                Message(
                    role="system",
                    content=f"{self_reflection_chat_instruction}\nI want you to run the verify tool to ensure the correctness before answering. If you believe the implementation matches the expected behavior, fix the code according to the waveform and ignore the description.",
                ),
                Message(
                    role="user",
                    content=f'[problem description]:\n{description}\n\n[function impl]:\n{add_code_block(func)}\n\n[simulation results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
    else:
        messages = f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:'
    return messages


def generic_generate_self_reflection(
    func: str,
    feedback: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str] = None,
) -> str:
    
    messages = self_reflection_prepare_prompt(
        func,
        feedbac,
        model,
        self_reflection_chat_instruction,
        self_reflection_completion_instruction,
        add_code_block,
        self_reflection_few_shot,
    )
    if model.is_chat:
        reflection = model.generate_chat(messages=messages)
    else:
        reflection = model.generate(messages)
    return reflection  # type: ignore


def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)

def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
