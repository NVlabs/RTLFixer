from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .verilog_generate import VerilogGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci, PhindCodeLlama, CodeLlama2, CodeLlama3, CodeGen
from .agents import ReAct, PlanAndExecute, OpenAIFunc, RTLFixer


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    elif lang == "vg" or lang == "verilog":
        return VerilogGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")
        
        
def agent_factory(agent_name: str, model_name: str, **kwargs):
    if agent_name == "cot":
        return model_factory(model_name)
    elif agent_name == "react":
        return ReAct(model_name, **kwargs)
    elif "rtlfixer" in agent_name:
        return RTLFixer(agent_name, model_name, **kwargs)
    elif agent_name == "openaifunc":
        return OpenAIFunc(model_name, **kwargs)
    elif agent_name == "planexec":
        return PlanAndExecute(model_name, **kwargs)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")


def model_factory(model_name: str) -> ModelBase:
    if 'gpt-4' in model_name:
        return GPT4(model_name)
    elif "gpt-3.5" in model_name:
        return GPT35(model_name)
    elif "starchat" in model_name:
        return StarChat()
    elif "codegen" in model_name:
        return CodeGen()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "B-" in model_name:
            kwargs["version"] = model_name.split("-")[-1]
        elif "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
            
        if "Phind" in model_name:
            MODEL = PhindCodeLlama
        elif model_name.startswith("codellama2"):
            MODEL = CodeLlama2
        elif model_name.startswith("codellama3"):
            MODEL = CodeLlama3
        else:
            MODEL = CodeLlama

        return MODEL(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
