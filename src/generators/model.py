import os
from typing import List, Union, Optional, Literal
import dataclasses

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    
    if 'azure' in model:
        
        openai.api_type = "azure"
        openai.api_base = ""
        openai.api_version = "2023-07-01-preview"
        openai.api_key = ""
        os.environ['OPENAI_API_BASE'] = ""
        os.environ['OPENAI_API_KEY'] = ""
        os.environ['OPENAI_API_VERSION'] = "2023-07-01-preview"
        os.environ['OPENAI_API_TYPE'] = "azure"
        engine = ""

        model = model.replace('azure-', '')
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            stop='endmodule'
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            stop='endmodule'
        )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)


class GPT4(GPTChat):
    def __init__(self, model_name: str):
        super().__init__(model_name)


class GPT35(GPTChat):
    def __init__(self, model_name: str):
        super().__init__(model_name)


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 2048, temperature: float = 0.2, num_comps: int = 1, output_only: bool = False) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)
        
        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings
            ),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )
        
        if output_only:
            processed_outputs = []
            for i in range(outputs.shape[0]):
                input_length = len(prompt[i])
                processed_outputs.append(outputs[i][input_length:])
            import torch
            outputs = torch.stack(processed_outputs)
            # will still write entire module
        
        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)
            
#         print(messages)
#         print(prompt)
#         print(outs)
#         import pdb
#         pdb.set_trace()

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError
        
        
class CodeGen(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "/tmp/test-clm2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=''
        )
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        
        tokenizer = AutoTokenizer.from_pretrained(
            "/tmp/test-clm2",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)
        
    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, output_only: bool = True) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)
        import torch
        import transformers
        from optimum.pipelines import pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            # accelerator="bettertransformer"
        )

        completion = pipeline(
            prompt,
            do_sample=True,
            top_p=0.75,
            top_k=40,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
            max_length=min(
                max_tokens, self.model.config.max_position_embeddings
            ),
            return_full_text=False
        )[0]['generated_text']
        return completion

    def prepare_prompt(self, messages: List[Message]):
        return "\n".join([i.content for i in messages])

    def extract_output(self, output: str, prompt: str = "") -> str:
        return output


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            # "HuggingFaceH4/starchat-beta",
            "bigcode/starcoder",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=''
        )
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        
        tokenizer = AutoTokenizer.from_pretrained(
            # "HuggingFaceH4/starchat-beta",
            "bigcode/starcoder"
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)
        
    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, output_only: bool = True) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096)
        # remove eos token from last message
        inputs = inputs[..., :-1]
        
        # Generate
        generate_ids = self.model.generate(
            inputs.to(self.model.device),
            max_length=min(
                max_tokens, self.model.config.max_position_embeddings
            ),
            do_sample=True,
            top_p=0.75,
            top_k=40,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )
        completion = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
#         print(messages)
#         print(prompt)
#         print(completion)
#         import pdb
#         pdb.set_trace()

        completion = self.extract_output(completion, prompt)
        return completion

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        prompt += f"<|{messages[0].role}|>\n{messages[0].content}\n<|end|>\n"
        prompt += f"<|{messages[1].role}|>\n{messages[1].content}\n<|end|>\n"
        prompt += f"<|assistant|>\n{messages[2].content}\n"
        return prompt
        
        
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str, prompt: str = "") -> str:
        output = output.replace(prompt, "").strip()
        st = output.find("endmodule")
        output = output[:st+len("endmodule")]
        return output
        
        
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]
        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages

        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
    
    
class CodeLlama2(CodeLlama):
    
    """
    Remember input text in order to remove them from decoded output.
    """
        
#     def prepare_prompt(self, messages: List[Message]):
#         if messages[0].role != "system":
#             messages = [
#                 Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
#             ] + messages
#         messages = [
#             Message(role=messages[1].role, content=self.B_SYS +
#                     messages[0].content + self.E_SYS + messages[1].content)
#         ] + messages[2:]
#         assert all([msg.role == "user" for msg in messages[::2]]) and all(
#             [msg.role == "assistant" for msg in messages[1::2]]
#         ), (
#             "model only supports 'system', 'user' and 'assistant' roles, "
#             "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
#         )
#         messages_tokens: List[str] = [
#             f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} "
#             for prompt, answer in zip(
#                 messages[::2],
#                 messages[1::2],
#             )
#         ]
#         messages_tokens += [f"{(messages[-1].content).strip()}"]
#         assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
#         return "\n".join(messages_tokens)
    
    def prepare_prompt(self, messages: List[Message]):
        messages_tokens: List[str] = [
            f"{self.B_SYS} {(sys.content).strip()} {self.E_SYS} {self.B_INST}\n{(user.content).strip()}\n{self.E_INST}\n// {(assist.content).strip()} "
            for sys, user, assist in zip(
                messages[::3],
                messages[1::3],
                messages[2::3],
            )
        ]
        return "\n".join(messages_tokens)
    
    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, output_only: bool = True) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096)
        # remove eos token from last message
        inputs = inputs[..., :-1]
        
        # Generate
        generate_ids = self.model.generate(
            inputs.to(self.model.device),
            max_length=min(
                max_tokens, self.model.config.max_position_embeddings
            ),
            do_sample=True,
            top_p=0.75,
            top_k=40,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )
        completion = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if output_only:
            completion = completion.replace(prompt, "").split("\n\n\n")[0].strip()
            
        st = completion.find("endmodule")
        completion = completion[:st+len("endmodule")]
        return completion
    
    def extract_output(self, output: str) -> str:
        return output
    
    
class CodeLlama3(CodeLlama2):
    
    """
    Transormers pipeline to handle return_full_text=false
    """
    
    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1, output_only: bool = True) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        import torch
        import transformers
        from optimum.pipelines import pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            # accelerator="bettertransformer"
        )

        completion = pipeline(
            prompt,
            do_sample=True,
            top_p=0.75,
            top_k=40,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
            max_length=min(
                max_tokens, self.model.config.max_position_embeddings
            ),
            return_full_text=False
        )[0]['generated_text']

        return completion

    
class PhindCodeLlama(CodeLlama2):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = "You are an intelligent programming assistant."

    def __init__(self, version: Literal["v1", "v2"] = "v2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"Phind/Phind-CodeLlama-34B-{version}",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            f"Phind/Phind-CodeLlama-34B-{version}",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super(CodeLlama, self).__init__("PhindCodeLlama", model, tokenizer)
