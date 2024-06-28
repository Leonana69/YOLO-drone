import sys
sys.path.append("..")
from controller.llm_controller import LLMController
from controller.minispec_interpreter import MiniSpecInterpreter, MiniSpecProgram

controller = LLMController()

MiniSpecInterpreter.low_level_skillset = controller.low_level_skillset
MiniSpecInterpreter.high_level_skillset = controller.high_level_skillset
interpreter = MiniSpecInterpreter()

# print(interpreter.execute("8{_1=mr(50);?_1!=False{g('tiger');->True}tc(45)}"))
# print(interpreter.execute("g('person')"))

# interpreter.execute("8{_1=mr(50);?_1!=False{g('tiger');->True;}tc(45)};")
interpreter.execute('?sa("edible object")!=False{tc(45)}tc(180);')