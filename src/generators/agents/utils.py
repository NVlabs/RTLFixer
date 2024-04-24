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
