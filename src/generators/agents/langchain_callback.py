from typing import Dict, List, Union, Type, Any
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult
)
from langchain.callbacks import StdOutCallbackHandler


class MyCallbackHandler(StdOutCallbackHandler):
    
    def __init__(self, agent, max_iter: int = 20):
        super().__init__()
        self.agent = agent
        self.max_iter = max_iter
        self.history = []
        
    def log_history(function):
        def wrap_function(self, *args, **kwargs):
            log = (function.__name__, args, kwargs)
            
#             print("================================================================")
#             print(log)
#             print("================================================================")
#             import pdb
#             pdb.set_trace()
            
            self.history.append(log)
            return function(self, *args, **kwargs)
        return wrap_function
        
    @log_history
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    @log_history
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

    @log_history
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    @log_history
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    @log_history
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    @log_history
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        self.history = []

    @log_history
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    @log_history
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    @log_history
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        
    @log_history
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        
    @log_history
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
  
    @log_history
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    @log_history
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
