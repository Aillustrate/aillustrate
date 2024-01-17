from pydantic.dataclasses import dataclass


@dataclass
class PromptConfig:
    prompt_prefix: str = "{concept}"
    concept_name: str = "interiors"
    design: str = "design style"
    example: str = ""
    shot_length: str = ""
    extra_aspects: str = ""
