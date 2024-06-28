from typing import List, Tuple, Union
import re
from enum import Enum
import time
from typing import Optional
from threading import Thread
from queue import Queue
from openai import ChatCompletion, Stream

from .skillset import SkillSet
from .utils import split_args

MiniSpecValueType = Union[int, float, bool, str, None]

def evaluate_value(value: str) -> MiniSpecValueType:
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit():
        return float(value)
    elif value == 'True':
        return True
    elif value == 'False':
        return False
    elif value == 'None' or len(value) == 0:
        return None
    else:
        return value.strip('\'"')

class MiniSpecReturnValue:
    def __init__(self, value: MiniSpecValueType, replan: bool):
        self.value = value
        self.replan = replan

    def from_tuple(t: Tuple[MiniSpecValueType, bool]):
        return MiniSpecReturnValue(t[0], t[1])
    
    def default():
        return MiniSpecReturnValue(None, False)
    
    def __repr__(self) -> str:
        return f'value={self.value}, replan={self.replan}'
    
class ParsingState(Enum):
    CODE = 0
    ARGUMENTS = 1
    CONDITION = 2
    LOOP_COUNT = 3
    SUB_STATEMENTS = 4

class MiniSpecProgram:
    def __init__(self, env: Optional[dict] = None) -> None:
        self.statements: List[Statement] = []
        self.depth = 0
        self.finished = False
        if env is None:
            self.env = {}
        else:
            self.env = env
        self.current_statement = Statement(self.env)

    def parse(self, code_instance: Stream[ChatCompletion.ChatCompletionChunk] | List[str], exec: bool = False) -> bool:
        for chunk in code_instance:
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            if code == None or len(code) == 0:
                continue
            for c in code:
                if c == '{':
                    self.depth += 1
                elif c == '}':
                    if self.depth == 0:
                        self.finished = True
                        return True
                    self.depth -= 1
                    
                if self.current_statement.parse(c, exec):
                    if len(self.current_statement.action) > 0:
                        self.statements.append(self.current_statement)
                    self.current_statement = Statement(self.env)
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        print(f'Executing program: {self} {self.finished}')
        ret_val = MiniSpecReturnValue.default()
        count = 0
        while not self.finished:
            if len(self.statements) <= count:
                time.sleep(0.1)
                continue
            ret_val = self.statements[count].eval()
            if ret_val.replan:
                return ret_val
            count += 1
        if count < len(self.statements):
            for i in range(count, len(self.statements)):
                ret_val = self.statements[i].eval()
                if ret_val.replan:
                    return ret_val
        return ret_val
    
    def __repr__(self) -> str:
        s = ''
        for statement in self.statements:
            s += f'{statement}; '
        return s

class Statement:
    execution_queue: Queue['Statement'] = None
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self, env: dict) -> None:
        self.code_buffer: str = ''
        self.parsing_state: ParsingState = ParsingState.CODE
        self.condition: Optional[str] = None
        self.loop_count: Optional[int] = None
        self.action: str = ''
        self.allow_digit: bool = False
        self.executable: bool = False
        self.sub_statements: Optional[MiniSpecProgram] = None
        self.env = env

    def get_env_value(self, var) -> MiniSpecValueType:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        for c in code:
            match self.parsing_state:
                case ParsingState.CODE:
                    if c == '?':
                        self.action = 'if'
                        self.parsing_state = ParsingState.CONDITION
                    elif c == ';' or c == '}' or c == ')':
                        if c == ')':
                            self.code_buffer += c
                        self.action = self.code_buffer
                        self.executable = True
                        if exec and self.action != '':
                            self.execution_queue.put(self)
                        return True
                    else:
                        if c.isalpha() or c == '_':
                            self.allow_digit = True
                        self.code_buffer += c
                    if c.isdigit() and not self.allow_digit:
                        self.action = 'loop'
                        self.parsing_state = ParsingState.LOOP_COUNT
                case ParsingState.CONDITION:
                    if c == '{':
                        print(f'Parse Condition: {self.code_buffer}')
                        self.condition = self.code_buffer
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c
                case ParsingState.LOOP_COUNT:
                    if c == '{':
                        print(f'Parse Loop count: {self.code_buffer}')
                        self.loop_count = int(self.code_buffer)
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c
                case ParsingState.SUB_STATEMENTS:
                    if self.sub_statements.parse([c]):
                        return True
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        print(f'Executing: {self} {self.action} {self.condition} {self.loop_count}')
        while not self.executable:
            time.sleep(0.1)
        if self.action == 'if':
            cond = self.eval_condition(self.condition)
            if cond:
                print(f'Executing Condition statement: {self.sub_statements}')
                return self.sub_statements.eval()
        elif self.action == 'loop':
            print(f'Executing Loop statement: {self.loop_count} {self.sub_statements}')
            ret_val = MiniSpecReturnValue.default()
            for _ in range(self.loop_count):
                ret_val = self.sub_statements.eval()
                if ret_val.replan:
                    return ret_val
            return ret_val
        else:
            return self.eval_action(self.action)
    
    def eval_action(self, action: str) -> MiniSpecReturnValue:
        print(f'Action: {action}')
        action = action.strip().split('=', 1)
        if len(action) == 2:
            var, func = action
            print(f'Var: {var}, Func: {func}')
            ret_val = self.eval_function(func)
            if not ret_val.replan:
                self.env[var.strip()] = ret_val.value
            return ret_val
        elif len(action) == 1:
            return self.eval_function(action[0])
        else:
            raise Exception('Invalid function call statement')
        
    def eval_function(self, func: str) -> MiniSpecReturnValue:
        print(f'Function: {func}')
        # append to execution state queue
        func = func.split('(', 1)
        name = func[0].strip()
        if len(func) == 2:
            args = func[1].strip()[:-1]
            args = split_args(args)
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip('\'"')
                if args[i].startswith('_'):
                    args[i] = self.get_env_value(args[i])
        else:
            args = []

        if name == 'int':
            return MiniSpecReturnValue(int(args[0]), False)
        elif name == 'float':
            return MiniSpecReturnValue(float(args[0]), False)
        elif name == 'str':
            return MiniSpecReturnValue(args[0], False)
        else:
            skill_instance = Statement.low_level_skillset.get_skill(name)
            if skill_instance is not None:
                print(f'Executing low-level skill: {skill_instance.get_name()} {args}')
                return MiniSpecReturnValue.from_tuple(skill_instance.execute(args))

            skill_instance = Statement.high_level_skillset.get_skill(name)
            if skill_instance is not None:
                interpreter = MiniSpecProgram()
                interpreter.parse([skill_instance.execute(args)])
                print(f'Executing high-level skill: {skill_instance.get_name()}', args, skill_instance.execute(args))
                val = interpreter.eval()
                val = MiniSpecReturnValue.default()
                if val.value == 'rp':
                    return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
                return val
            raise Exception(f'Skill {name} is not defined')

    def eval_var(self, var: str) -> MiniSpecReturnValue:
        var = var.strip()
        if len(var) == 0:
            raise Exception('Empty operand')
        if var.startswith('_'):
            return MiniSpecReturnValue(self.get_env_value(var), False)
        elif var == 'True' or var == 'False':
            return MiniSpecReturnValue(evaluate_value(var), False)
        elif var[0].isalpha():
            return self.eval_action(var)
        else:
            return MiniSpecReturnValue(evaluate_value(var), False)

    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        if '&' in condition:
            conditions = condition.split('&')
            cond = True
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan:
                    return ret_val
                cond = cond and ret_val.value
            return MiniSpecReturnValue(cond, False)
        if '|' in condition:
            conditions = condition.split('|')
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan:
                    return ret_val
                if ret_val.value == True:
                    return MiniSpecReturnValue(True, False)
            return MiniSpecReturnValue(False, False)
        
        operand_1, comparator, operand_2 = re.split(r'(>|<|==|!=)', condition)
        operand_1 = self.eval_var(operand_1)
        if operand_1.replan:
            return operand_1
        operand_2 = self.eval_var(operand_2)
        if operand_2.replan:
            return operand_2
        
        if type(operand_1.value) != type(operand_2.value):
            if comparator == '!=':
                return MiniSpecReturnValue(True, False)
            elif comparator == '==':
                return MiniSpecReturnValue(False, False)
            else:
                raise Exception(f'Invalid comparator: {operand_1.value}:{type(operand_1.value)} {operand_2.value}:{type(operand_2.value)}')
            
        if comparator == '>':
            cmp = operand_1.value > operand_2.value
        elif comparator == '<':
            cmp = operand_1.value < operand_2.value
        elif comparator == '==':
            cmp = operand_1.value == operand_2.value
        elif comparator == '!=':
            cmp = operand_1.value != operand_2.value
        else:
            raise Exception(f'Invalid comparator: {comparator}')
        
        return MiniSpecReturnValue(cmp, False)


    def __repr__(self) -> str:
        s = ''
        if self.action == 'if':
            s += f'if {self.condition}'
        elif self.action == 'loop':
            s += f'[{self.loop_count}]'
        else:
            s += f'{self.action}'
        if self.sub_statements is not None:
            s += ' {'
            for statement in self.sub_statements.statements:
                s += f'{statement}; '
            s += '}'
        return s

class MiniSpecInterpreter:
    def __init__(self):
        self.env = {}
        self.ret = False
        self.code_buffer: str = ''
        self.execution_status = []
        if Statement.low_level_skillset is None or \
            Statement.high_level_skillset is None:
            raise Exception('Statement: Skillset is not initialized')
        
        Statement.execution_queue = Queue()
        self.execution_thread = Thread(target=self.executor)
        self.execution_thread.start()

    def execute(self, code: Stream[ChatCompletion.ChatCompletionChunk] | List[str]) -> MiniSpecReturnValue:
        program = MiniSpecProgram()
        program.parse(code, True)
        print("Program: ", program, len(program.statements))

    def executor(self):
        while True:
            if not Statement.execution_queue.empty():
                statement = Statement.execution_queue.get()
                print(f'Queue get statement: {statement}')
                statement.eval()
            else:
                time.sleep(0.1)

    def get_env_value(self, var) -> MiniSpecValueType:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def execute_old(self, code) -> MiniSpecReturnValue:
        # print(f'Executing: {code}, depth={self.depth}')
        statements = self.split_statements(code)
        for statement in statements:
            if not statement:
                continue
            elif statement.startswith('->'):
                # return statement, won't trigger replan
                return self.evaluate_return(statement)
            elif re.match(r'^\d', statement):
                # loop statement, may trigger replan
                ret_val = self.execute_loop(statement)
            elif statement.startswith('?'):
                # conditional statement, may trigger replan
                ret_val = self.execute_conditional(statement)
            else:
                # function call, may trigger replan
                ret_val = self.execute_function_call(statement)

            if ret_val.replan:
                return ret_val
            
            if self.ret:
                return ret_val
        return MiniSpecReturnValue.default()

    def evaluate_return(self, statement) -> MiniSpecReturnValue:
        _, value = statement.split('->')
        if value.startswith('_'):
            value = self.get_env_value(value)
        else:
            value = evaluate_value(value.strip())
        self.ret = True
        return MiniSpecReturnValue(value, False)

    def execute_function_call(self, statement) -> MiniSpecReturnValue:
        splits = statement.split('=', 1)
        split_count = len(splits)
        if split_count == 2:
            var, func = splits
            ret_val = self.call_function(func)
            if not ret_val.replan:
                self.env[var.strip()] = ret_val.value
            return ret_val
        elif split_count == 1:
            return self.call_function(statement)
        else:
            raise Exception('Invalid function call statement')

    def evaluate_condition(self, condition) -> MiniSpecReturnValue:
        if '&' in condition:
            conditions = condition.split('&')
            cond = True
            for c in conditions:
                ret_val = self.evaluate_condition(c)
                if ret_val.replan:
                    return ret_val
                cond = cond and ret_val.value
            return MiniSpecReturnValue(cond, False)
        if '|' in condition:
            conditions = condition.split('|')
            for c in conditions:
                ret_val = self.evaluate_condition(c)
                if ret_val.replan:
                    return ret_val
                if ret_val.value == True:
                    return MiniSpecReturnValue(True, False)
            return MiniSpecReturnValue(False, False)
        
        operand_1, comparator, operand_2 = re.split(r'(>|<|==|!=)', condition)
        operand_1 = self.evaluate_operand(operand_1)
        if operand_1.replan:
            return operand_1
        operand_2 = self.evaluate_operand(operand_2)
        if operand_2.replan:
            return operand_2
        
        if type(operand_1.value) != type(operand_2.value):
            if comparator == '!=':
                return MiniSpecReturnValue(True, False)
            elif comparator == '==':
                return MiniSpecReturnValue(False, False)
            else:
                raise Exception(f'Invalid comparator: {operand_1.value}:{type(operand_1.value)} {operand_2.value}:{type(operand_2.value)}')
        
        if comparator == '>':
            cmp = operand_1.value > operand_2.value
        elif comparator == '<':
            cmp = operand_1.value < operand_2.value
        elif comparator == '==':
            cmp = operand_1.value == operand_2.value
        elif comparator == '!=':
            cmp = operand_1.value != operand_2.value
        else:
            raise Exception(f'Invalid comparator: {comparator}')
        
        return MiniSpecReturnValue(cmp, False)
        
    def evaluate_operand(self, operand) -> MiniSpecReturnValue:
        operand = operand.strip()
        if len(operand) == 0:
            raise Exception('Empty operand')
        if operand.startswith('_'):
            # variable
            return MiniSpecReturnValue(self.get_env_value(operand), False)
        elif operand == 'True' or operand == 'False':
            # boolean
            return MiniSpecReturnValue(evaluate_value(operand), False)
        elif operand[0].isalpha():
            # function call
            return self.execute_function_call(operand)
        else:
            # value
            return MiniSpecReturnValue(evaluate_value(operand), False)

    def call_function(self, func) -> MiniSpecReturnValue:
        self.execution_status.append(func)
        splits = func.split('(', 1)
        name = splits[0].strip()
        if len(splits) == 2:
            args = splits[1].strip()[:-1]
            args = split_args(args)
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip('\'"')
                if args[i].startswith('_'):
                    args[i] = self.get_env_value(args[i])
        else:
            args = []

        if name == 'int':
            return MiniSpecReturnValue(int(args[0]), False)
        elif name == 'float':
            return MiniSpecReturnValue(float(args[0]), False)
        elif name == 'str':
            return MiniSpecReturnValue(args[0], False)
        else:
            skill_instance = MiniSpecInterpreter.low_level_skillset.get_skill(name)
            if skill_instance is not None:
                return MiniSpecReturnValue.from_tuple(skill_instance.execute(args))

            skill_instance = MiniSpecInterpreter.high_level_skillset.get_skill(name)
            if skill_instance is not None:
                interpreter = MiniSpecInterpreter()
                val = interpreter.execute(skill_instance.execute(args))
                if val.value == 'rp':
                    return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
                return val
            raise Exception(f'Skill {name} is not defined')