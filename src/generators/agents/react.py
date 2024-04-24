import re
import json
from typing import Dict, List, Union, Type, Any
import langchain
from copy import deepcopy
from pydantic import BaseModel

# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser, ReActJsonSingleInputOutputParser
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.agents import tool, load_tools, OpenAIFunctionsAgent, AgentExecutor, OpenAIMultiFunctionsAgent, initialize_agent, AgentType, AgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from uuid import uuid4


from .langchain_callback import MyCallbackHandler
from .langchain_tools import VerilogToolkit
from .react_json_single_input_parser import RobustReActJsonSingleInputOutputParser
from ..model import GPTChat, Message
langchain.debug = True


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
        return
    


class ReAct(GPTChat):

    def __init__(
        self,
        model_name: str,
        exe,
        max_iters: int,
        toolset: list = [],
        system_prompt: str = None,
        temperature: int = 0.2,
        memory_key: str = None
    ):
        super().__init__(model_name)
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=2048, top_p=1.0)
        self.toolkit = VerilogToolkit(exe, toolset=toolset)
        self.agent_executor = initialize_agent(
            self.toolkit.tools,
            self.llm,
            memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, input_key='input', output_key='output') if memory_key else None,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            callback_manager=None,
            max_iterations=max_iters,
#             max_execution_time=25,
            handle_parsing_errors=self._handle_error,
            return_intermediate_steps=True,
            early_stopping_method="generate",
            agent_kwargs={
                'system_message': SystemMessage(content=system_prompt) if system_prompt else None,
                'extra_prompt_messages': [MessagesPlaceholder(variable_name=memory_key)] if memory_key else None,
                'output_parser': RobustReActJsonSingleInputOutputParser(),
#                 'input_variables': None,
#                 "prefix": None,
#                 'format_instructions': None,
#                 "output_parser": self.output_parser,
            },
        )
        self.uuid = uuid4().hex
        self.callbacks = [
#             HumanApprovalCallbackHandler(),
            MyCallbackHandler(self.agent_executor)  # must be last one
        ]
        
    def _handle_error(self, error: str):
        try:
            with open(f'error_log/{self.uuid}.parse_error_log', 'a') as f:
                print(error, file=f)
        except Exception:
            return "Function arguments is not in valid json format. I should fix it and try again."
        
    def postprocess(self, text: str):
        tmp = parse_markdown_code_block(text, 'verilog')
        if tmp:
            return tmp
        return text
    
    def serialize(self, history):
        try:
            from langchain.load.serializable import Serializable
            from pydantic import BaseModel
            from uuid import UUID
            
            def todict(obj, classkey=None):
                if isinstance(obj, dict):
                    data = {}
                    for (k, v) in obj.items():
                        data[k] = todict(v, classkey)
                    return data
                elif isinstance(obj, Serializable):
                    return obj.to_json()
                elif isinstance(obj, BaseModel):
                    return obj.json()
                elif isinstance(obj, UUID):
                    return obj.hex
                elif hasattr(obj, "_ast"):
                    return todict(obj._ast())
                elif hasattr(obj, "__iter__") and not isinstance(obj, str):
                    return [todict(v, classkey) for v in obj]
                elif hasattr(obj, "__dict__"):
                    data = dict([(key, todict(value, classkey)) 
                        for key, value in obj.__dict__.items() 
                        if not callable(value) and not key.startswith('_')])
                    if classkey is not None and hasattr(obj, "__class__"):
                        data[classkey] = obj.__class__.__name__
                    return data
                else:
                    return obj
            return todict(history)
        except Exception:
            import traceback
            print(traceback.format_exc())
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(history)).decode()

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        completion = {}
        try:
            messages = self.adapt(messages)
            completion = self.agent_executor(messages, callbacks=self.callbacks)
        except Exception as e:
            err_log = str(e)
            if 'maximum context length' in str(e):
                pass
            elif 'Could not parse' in str(e):
                pass
            else:
                import traceback
                err_log = traceback.format_exc()
                with open(f'error_log/{self.uuid}.chat_error_log', 'a') as f:
                    print(err_log, file=f)
            self.callbacks[-1].history.append(
                ('Exception', err_log)
            )
        finally:
            self.intermediate_steps = self.serialize(completion.get('intermediate_steps', []))
            self.agent_history = self.serialize(self.callbacks[-1].history)
            completion = self.postprocess(completion.get('output', ""))
            return completion
        
    def adapt(self, messages: List[Message]):
        output = []
        for i in messages:
            if i.role == "user":
                output.append(HumanMessage(content=i.content))
            elif i.role == "assistant":
                output.append(AIMessage(content=i.content))
            elif i.role == "system":
                output.append(SystemMessage(content=i.content))
        return output


if __name__ == "__main__":
    react = ReAct("Your name is react.")
    messages = [
        HumanMessage(content="I love programming."),
        AIMessage(content="I love programming too."),
        HumanMessage(content="What is the weather in LA and SF?"),
    ]
    print(react.agent_executor.run(messages))
