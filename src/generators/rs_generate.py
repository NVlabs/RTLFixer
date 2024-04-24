from generators.model import ModelBase
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection
from .parse import parse_code_block, add_code_block

from typing import List, Optional, Union

RS_SIMPLE_COMPLETION_INSTRUCTION = "// Write the body of this function only."
RS_REFLEXION_COMPLETION_INSTRUCTION = "You are a Rust writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature).\n\n-----"
RS_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Rust writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_RUST_CODEBLOCK_INSTRUCTION = "Use a Rust code block to write your response. For example:\n```rust\nfn main() {\n    println!(\"Hello\");\n}\n```"

RS_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with Rust code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
RS_REFLEXION_CHAT_INSTRUCTION = "You are an AI Rust assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
RS_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Rust programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."

RS_REFLEXION_COMPLETION_INSTRUCTION = "You are a Rust programming assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Apply the changes below by writing the body of this function only.\n\n-----"
RS_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Rust programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"

RS_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```rust
fn add(a: i32, b: i32) -> i32 {
    // Given integers a and b, return the total value of a and b.
    a - b
}
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert_eq!(add(1, 2), 3); // output: -1
assert_eq!(add(1, 2), 4); // output: -1

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
```rust
fn add(a: i32, b: i32) -> i32 {
    // Given integers a and b, return the total value of a and b.
    a + b
}
```

END EXAMPLES
'''

RS_TEST_GENERATION_FEW_SHOT = """For example:

func signature:
/// Add three numbers together.
/// This function takes three numbers as input and returns the sum of the three numbers.
fn add3Numbers(x: i32, y: i32, z: i32) -> i32 {

unit tests:
assert_eq!(add3Numbers(1, 2, 3), 6);
assert_eq!(add3Numbers(-1, 2, 3), 4);
assert_eq!(add3Numbers(1, -2, 3), 2);
assert_eq!(add3Numbers(1, 2, -3), 0);
assert_eq!(add3Numbers(-3, -2, -1), -6);
assert_eq!(add3Numbers(0, 0, 0), 0);
"""

RS_SELF_REFLECTION_FEW_SHOT = '''Example 1:
[function impl]:
```rust
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
// Given an array of strings strs, group the anagrams together. You can return the answer in any order.
// An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
  use std::collections::HashMap;
  let mut map: HashMap<[u8;26], Vec<String>> = HashMap::with_capacity(strs.len());
  let offset = 'a' as usize;

  for str in strs.into_iter() {
    let mut chars: [u8; 26] = [0; 26];

    for char in str.chars() {
      chars[char.to_ascii_lowercase() as usize - offset] += 1;
    }

    // Flaw: using str.len() instead of chars in the hashmap key
    map.entry(str.len())
      .and_modify(|v| v.push(str.clone()))
      .or_insert(vec![str]);
  }
  
  let mut arr: Vec<Vec<String>> = Vec::new();
  for v in map.into_values() {
    arr.push(v);
  }
  arr
}
```

[unit test results]:
Tested passed:
assert_eq!(func(vec![""]), vec![vec![""]]);
assert_eq!(func(vec!["a"]), vec![vec!["a"]]);

Tests failed:
assert_eq!(func(vec!["eat", "tea", "tan", "ate", "nat", "bat"]), vec![vec!["bat"], vec!["nat", "tan"], vec!["ate", "eat", "tea"]]); # output:  [["bat", "tan", "nat"], ["eat", "tea", "ate"]]

[self-reflection]:
The implementation failed to group the anagrams together correctly. Instead, it grouped words by their length, which is not the intended behavior. The issue lies in using the length of the input strings (str.len()) as the key for the hashmap, rather than the count of each character in the strings (chars). To overcome this error, I should change the hashmap key to the character count array (chars). This will ensure that words with the same character counts (anagrams) are grouped together, which is the desired output. Next time I approach the problem, I will make sure to use the correct hashmap key to group the anagrams.

END EXAMPLES

'''
RS_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are a Rust programming assistant, an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring.

{RS_TEST_GENERATION_FEW_SHOT}"""

RS_TEST_GENERATION_CHAT_INSTRUCTION = """You are a Rust programming assistant, an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""


def dump_tests(tests: List[str]) -> str:
    """
    Dumps the tests to a string.
    """
    return "\n".join(tests)


def parse_tests(tests: str) -> List[str]:
    """
    Parses the tests from a string.
    """
    return [test.strip() for test in tests.splitlines() if "assert" in test]

# TODO: type-check generated unit tests?


class RsGenerator(Generator):
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=RS_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=RS_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "rust"),
            self_reflection_few_shot=RS_SELF_REFLECTION_FEW_SHOT,
        )

    def func_impl(
        self,
        func_sig: str,
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
            reflexion_chat_instruction=RS_REFLEXION_CHAT_INSTRUCTION,
            simple_chat_instruction=RS_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=RS_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=RS_SIMPLE_COMPLETION_INSTRUCTION,
            reflexion_few_shot=RS_REFLEXION_FEW_SHOT_ADD,
            parse_code_block=lambda x: parse_code_block(x, "rust"),
            add_code_block=lambda x: add_code_block(x, "rust"),
        )

    def internal_tests(
            self,
            func_sig: str,
            model: ModelBase,
            max_num_tests: int = 5
    ) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test + ";" for test in tests.split(";")]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=RS_TEST_GENERATION_FEW_SHOT,
            test_generation_chat_instruction=RS_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_completion_instruction=RS_TEST_GENERATION_COMPLETION_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=(lambda x: True)  # TODO: for now. typecheck maybe?
        )
