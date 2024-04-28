import os, ast

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, gear=False):
        self.llm = LLMWrapper()
        self.model_name = GPT4

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, "./assets/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/prompt_yolo_get_class.txt"), "r") as f:
            self.prompt_yolo_get_class = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def set_model(self, model_name):
        self.model_name = model_name

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill
    
    def get_class(self, task_description: str):
        prompt = self.prompt_yolo_get_class.format(task_description=task_description)
        result = self.llm.request(prompt, GPT3)
        return ast.literal_eval(result)

    def plan(self, task_description: str, scene_description: str = None, error_message: str = None, previous_response: str = None, execution_status: str = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()
        prompt = self.prompt_plan.format(system_skill_description_high=self.high_level_skillset,
                                             system_skill_description_low=self.low_level_skillset,
                                            #  minispec_syntax=self.minispec_syntax,
                                             guides=self.guides,
                                             plan_examples=self.plan_examples,
                                             error_message=error_message,
                                             scene_description=scene_description,
                                             task_description=task_description,
                                             previous_response=previous_response,
                                             execution_status=execution_status)
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt, self.model_name)
    
    def probe(self, question: str) -> MiniSpecValueType:
        prompt = self.prompt_probe.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt, self.model_name)), False