import json
import re
from uuid import uuid4
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"


class RobustReActOutputParser(AgentOutputParser):
    """Output parser for the ReAct agent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        action_prefix = "Action:"
        if action_prefix not in text:
            return AgentFinish({"output": text}, text)
            raise OutputParserException(f"Could not parse LLM Output (no action): {text}")

        st = text.find(action_prefix) + len(action_prefix)
        action_str = text[st:]

        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise OutputParserException(
                f"Could not parse action directive: {action_str}"
            )
        action, action_input = re_matches.group(1), re_matches.group(2)
        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)


class RobustReActJsonSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input in json format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action:
    ```
    {
        "action": "search",
        "action_input": "what is the temperature in SF"
    }
    ```
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
    """Regex pattern to parse the output."""
    
    parse_fail_record = []

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                # Fast fail to parse Final Answer.
                raise ValueError("action not found")
            action = found.group(1)
            response = json.loads(action.strip().replace('\n', ''))
            includes_action = "action" in response
            if includes_answer and includes_action:
                raise OutputParserException(
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
            return AgentAction(
                response["action"], response.get("action_input", {}), text
            )

        except Exception:
            if not includes_answer:
                try:
                    return RobustReActOutputParser().parse(text)
                except Exception:
                    self.parse_fail_record.append(text)
                    raise OutputParserException(f"Could not parse LLM output: {text}")
            output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, text)

    @property
    def _type(self) -> str:
        return "robust-react-json-single-input"

