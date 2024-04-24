import os
import re
import json
from typing import Dict, List, Union, Type, Any
import langchain
from copy import deepcopy
from pydantic import BaseModel

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
from .utils import parse_markdown_code_block
langchain.debug = True



class CodeAgent(GPTChat):

    def __init__(
        self,
        model_name: str,
        exe,
        max_iters: int,
        toolset: list = None,
        system_prompt: str = None,
        temperature: int = 0.4,
        memory_key: str = None,
        method: str = "",
        compiler: str = ""
    ):
        super().__init__(model_name)
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=2048, top_p=1.0)
#         memory_key = "memory"
        self.toolkit = VerilogToolkit(exe, toolset=toolset, llm=self.llm, method=method, compilername=compiler)
        self.agent_executor = initialize_agent(
            self.toolkit.tools,
            self.llm,
            memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, input_key='input', output_key='output') if memory_key else None,
            agent=AgentType.OPENAI_FUNCTIONS,
#             agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            verbose=False,
            callback_manager=None,
            max_iterations=max_iters,
            handle_parsing_errors=self._handle_error,
            return_intermediate_steps=True,
#             early_stopping_method="generate",
            agent_kwargs={
                'system_message': SystemMessage(content=system_prompt) if system_prompt else None,
                'extra_prompt_messages': [MessagesPlaceholder(variable_name=memory_key)] if memory_key else None,
            },
        )
        self.uuid = uuid4().hex
        self.callbacks = [
            # HumanApprovalCallbackHandler(),
            MyCallbackHandler(self.agent_executor)  # must be last one
        ]
        self.agent_history = []
        self.error_logs = []
        self.max_iters = max_iters
        
    def reset_logs(self):
        self.agent_history = []
        self.error_logs = []
        
    def _handle_error(self, error: str):
        try:
            self.error_logs.append(error)
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
            self.error_logs.append(traceback.format_exc())
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(history)).decode()
        
    def self_verify(self, output: str):
        prompt = f"""
        ```
        {output}
        ```
        Is the logic implemented in the code?
        If yes, answer YES.
        If not, answer NO.
        """
        verify_result = self.toolkit.verify._run(prompt, "The logic is not yet implemented.")
        intermediate_step = {
            'self-verify': {
                'question': prompt,
                'result': verify_result,
            }
        }
        return verify_result, intermediate_step

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        completion = {}
        try:
            messages = self.adapt(messages)
            completion = {'output': "", 'intermediate_steps': []}
            random_trials_if_compiler_cannot_pass = 3
            # completion = self.agent_executor(messages, callbacks=self.callbacks)
            

            for _ in range(random_trials_if_compiler_cannot_pass):
                try:
                    for step in self.agent_executor.iter(messages):
                        
                        if output := step.get("intermediate_steps"):
#                             assert completion['intermediate_steps'] == step['intermediate_steps']
#                             completion['intermediate_steps'] = output
                            action, value = output[-1]
                            if action.tool == "verilog_compiler" and "Success" in value:
                                # agent invoke compiler & compile success
#                                 completion['output'] = action.tool_input['code_completion']
                                break
                        if output := step.get("intermediate_step"):
                            completion['intermediate_steps'] += output
                            action, value = output[-1]
                            if action.tool == "verilog_compiler" and "give" in value:
                                # agent invoke compiler & compile success
                                completion['output'] = action.tool_input['code_completion']
                                break
#                         if output := step.get("output"):
#                             completion['output'] = output

#                         if self.toolkit.num_compile > self.max_iters:
#                             break


                    if completion['output']:
                        compiler_log = self.toolkit.tools[0].run(completion['output'])
                        if f"The code has no compile error." in compiler_log:
    #                     if "Success" in compiler_log:
                            verify_result, intermediate_step = self.self_verify(completion['output'])
                            completion['intermediate_steps'].append(intermediate_step)
                            if "pass" in verify_result.lower():
                                break
                            else:
                                messages.append(AIMessage(content=completion['output']))
                                messages.append(HumanMessage(content="The logic is not implemented yet. Complete it instead of comments."))
    #                         break

                        
                    # Syntax error cannot fix. I give up. Generate a new sample.
                except Exception as e:
                    import traceback
                    self.error_logs.append(traceback.format_exc())
                    
            # 5 sample did not pass
            
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
            self.error_logs.append(err_log)
        finally:
            self.intermediate_steps = completion.get('intermediate_steps', [])
            self.agent_history.append(self.intermediate_steps)
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



class FixAgent(GPTChat):

    def __init__(
        self,
        model_name: str,
        exe,
        max_iters: int,
        toolset: list = None,
        system_prompt: str = None,
        temperature: int = 0.4,
        memory_key: str = None,
        method: str = False
    ):
        super().__init__(model_name)
        if 'azure' in model_name:
            import openai
            openai.api_type = "azure"
            openai.api_base = "https://testinstance1.openai.azure.com/"
            openai.api_version = "2023-07-01-preview"
            openai.api_key = "1854446716704e61a5d76c807c895d45"
            
            os.environ['OPENAI_API_BASE'] = "https://testinstance1.openai.azure.com/"
            os.environ['OPENAI_API_KEY'] = "1854446716704e61a5d76c807c895d45"
            os.environ['OPENAI_API_VERSION'] = "2023-07-01-preview"
            os.environ['OPENAI_API_TYPE'] = "azure"
            
            self.llm = AzureChatOpenAI(
                deployment_name="Morris-16k-for-sum",
                model_name="gpt-35-turbo-16k",
                temperature=temperature,
                max_tokens=2048,
                top_p=1.0
            )
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=2048, top_p=1.0)
        
        self.toolkit = VerilogToolkit(exe, toolset=toolset, llm=self.llm, method=method)
        self.agent_executor = initialize_agent(
            self.toolkit.tools,
            self.llm,
            memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, input_key='input', output_key='output') if memory_key else None,
            # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            callback_manager=None,
            max_iterations=max_iters,
            handle_parsing_errors=self._handle_error,
            return_intermediate_steps=True,
            early_stopping_method="generate",
            agent_kwargs={
                'system_message': SystemMessage(content=system_prompt) if system_prompt else None,
                'extra_prompt_messages': [MessagesPlaceholder(variable_name=memory_key)] if memory_key else None,
#                 'output_parser': RobustReActJsonSingleInputOutputParser(),
            },
        )
        self.uuid = uuid4().hex
        self.callbacks = [
            # HumanApprovalCallbackHandler(),
            MyCallbackHandler(self.agent_executor)  # must be last one
        ]
        self.agent_history = []
        self.error_logs = []
        
    def reset_logs(self):
        self.agent_history = []
        self.error_logs = []
        
    def _handle_error(self, error: str):
        try:
            self.error_logs.append(error)
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
            self.error_logs.append(traceback.format_exc())
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(history)).decode()
        
    def self_verify(self, output: str):
        prompt = f"""
        ```
        {output}
        ```
        Is the above answer complete ?
        If yes, answer YES.
        If not, answer NO.
        """
        verify_result = self.toolkit.verify._run(prompt, "Continue to debug using tools and must follow the action format instruction.")
        intermediate_step = {
            'self-verify': {
                'question': prompt,
                'result': verify_result,
            }
        }
        return verify_result, intermediate_step

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        completion = {}
        try:
            messages = self.adapt(messages)
            completions = {'output': "", 'intermediate_steps': []}
            
            for _ in range(3):
                try:
                    completion = self.agent_executor(messages, callbacks=self.callbacks)
                    completions['intermediate_steps'] = completion['intermediate_steps']
                    break
                    
                    # verify if need to continue
#                     verify_result, intermediate_step = self.self_verify(completion['output'])
#                     completions['intermediate_steps'].append(intermediate_step)
#                     if "pass" in verify_result.lower():
#                         break
#                     messages[-1].content += f"{completion['output']}\n{verify_result}"
                        
                except Exception as e:
                    import traceback
                    self.error_logs.append(traceback.format_exc())


#             random_trials_if_cannot_pass = 5
#             stop = False
#             for _ in range(random_trials_if_cannot_pass):
#                 try:
#                     for step in self.agent_executor.iter(messages):
                        
#                         if output := step.get("intermediate_steps"):
#                             assert completion['intermediate_steps'] == step['intermediate_steps']
#                             completion['intermediate_steps'] = output
#                             action, value = output[-1]
#                             if action.tool == "verify" and "good" in value:
#                                 # agent invoke self verify & success
#                                 completion['output'] = action.tool_input['answer']
#                         if output := step.get("intermediate_step"):
#                             completion['intermediate_steps'] += output
#                             action, value = output[-1]
#                             if action.tool == "verify" and "good" in value:
#                                 # agent invoke self verify & success
#                                 completion['output'] = action.tool_input['answer']
#                         if output := step.get("output"):
#                             completion['output'] = output
                    
#                     verify_result = self.toolkit.tools[-1].run(output)
#                     if "good" in verify_result:
#                         break
#                     else:
#                         messages[-1].content += f"\n{output}"
                        
#                     import pdb
#                     pdb.set_trace()
                    
#                 except Exception as e:
#                     import traceback
#                     self.error_logs.append(traceback.format_exc())

#             if "good" not in verify_result:
#                 import pdb
#                 pdb.set_trace()


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
            self.error_logs.append(err_log)
        finally:
            self.intermediate_steps = completions.get('intermediate_steps', [])
            self.agent_history.append(self.intermediate_steps)
            completion = completion.get('output', "")
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


def RTLFixer(agent_name: str, model_name: str, **kwargs):
    if "code" in agent_name:
        return CodeAgent(model_name, **kwargs)
    elif "fix" in agent_name:
        return FixAgent(model_name, **kwargs)
