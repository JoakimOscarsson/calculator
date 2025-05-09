from dataclasses import dataclass
from enum import IntEnum, Enum, auto
from typing import Protocol, Any

class TokenType(Enum):
    ADD = " + "
    SUBTRACT = " - "
    MULTIPLY = " * "
    DIVIDE = " / " 
    POWER = " ** "
    OPEN_PARENTHESES = " ("
    CLOSE_PARENTHESES = ") "
    NUMBER = "num" 
    UNARY_PLUS = " +"
    UNARY_MINUS = " -"
    EOF = "EOF"


@dataclass
class Token:
    token_type: TokenType
    value: Any = None


class PeakIter:
    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.buffert = None
        
    def __iter__(self):
        return self.iterator

    def __next__(self):
        return self.iterator.__next__()

    def iter(self):
        return self.__iter__()

    def next(self):
        if self.buffert:
            b = self.buffert
            self.buffert = None
            return b
        else:
            return self.iterator.__next__()

    def peak(self):
        self.buffert = self.next()
        return self.buffert


class Lexer:
    def build_number(number: str, iter: PeakIter):
        try:
            peaked = iter.peak()
        except StopIteration:
            return number
        if peaked in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]:
            return Lexer.build_number(number+iter.next(), iter)
        else:
            return number

    def tokenize(str):
        tokens = []
        iter = PeakIter(str)
        while True:
            try:
                char = iter.next()
                match char: 
                    case "+":
                        tokens.append(Token(TokenType.ADD))
                    case "-":
                        tokens.append(Token(TokenType.SUBTRACT))
                    case "*":
                        if iter.peak() == "*":
                            iter.next()
                            tokens.append(Token(TokenType.POWER))
                        else:
                            tokens.append(Token(TokenType.MULTIPLY))
                    case "/":
                        tokens.append(Token(TokenType.DIVIDE))
                    case "!":
                        raise SyntaxError("! not yet implemented")  # Todo!
                    case "(":
                        tokens.append(Token(TokenType.OPEN_PARENTHESES))
                    case ")":
                        tokens.append(Token(TokenType.CLOSE_PARENTHESES))
                    case " ":
                        pass
                    case "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                        tokens.append(
                            Token(
                                TokenType.NUMBER, float(Lexer.build_number(char, iter))
                            )
                        )
                    case _:
                        raise SyntaxError(f"Invalid Expression {char}")
            except StopIteration:
                break
        tokens.append(Token(TokenType.EOF))
        return tokens
        

class Expression(Protocol):
    def to_str(self) -> str:
        ...


@dataclass
class Literal():
    token: Token
    def to_str(self) -> str:
        return str(self.token.value)
 

@dataclass
class BinaryOperator():
    left: Expression
    operator: Token
    right: Expression 
    def to_str(self) -> str:
        return f"({self.left.to_str()} {self.operator.token_type.value} {self.right.to_str()})"


@dataclass
class UnaryOperator():
    operator: Token
    operand: Expression
    def to_str(self) -> str:
        return f"{self.operator.token_type.value}{self.operand.to_str()}"


class Parser:
    def parse(tokens: [Token]) -> Expression:
        ast = Parser.parse_additive(tokens)
        if tokens[0].token_type != TokenType.EOF:
            raise SyntaxError("Error, expected end of file!")
        return ast

    def parse_unary(tokens: [Token]) -> Expression:
        match tokens.pop(0).token_type:
            case TokenType.ADD:
                operator = Token(TokenType.UNARY_PLUS)
            case TokenType.SUBTRACT:
                operator = Token(TokenType.UNARY_MINUS)
            case _:
                raise SyntaxError("Unknown Unary operator")    
        operand = Parser.parse_power(tokens)
        return UnaryOperator(operator, operand)
    
    def parse_parenthesised(tokens: [Token]) -> Expression:
        tokens.pop(0)
        expr = Parser.parse_additive(tokens)
        if tokens[0].token_type != TokenType.CLOSE_PARENTHESES:
            raise SyntaxError("Error, parentheses not closed")
        tokens.pop(0)
        return expr

    def parse_additive(tokens: [Token]) -> Expression:
        left = Parser.parse_multiplicative(tokens) 
        while tokens[0].token_type == TokenType.ADD or tokens[0].token_type == TokenType.SUBTRACT:
            operator = tokens.pop(0)
            right = Parser.parse_multiplicative(tokens)
            left = BinaryOperator(left, operator, right)
        return left

    def parse_multiplicative(tokens: [Token]) -> Expression:
        left = Parser.parse_power(tokens)
        while tokens[0].token_type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.OPEN_PARENTHESES]:
            if tokens[0].token_type ==  TokenType.OPEN_PARENTHESES:
                operator = Token(TokenType.MULTIPLY)
                right = Parser.parse_parenthesised(tokens)
            else:
                operator = tokens.pop(0)
                right = Parser.parse_power(tokens)
                if operator.token_type is TokenType.DIVIDE and right == 0:
                    raise ZeroDivisionError()
            left = BinaryOperator(left, operator, right)
        return left

    def parse_power(tokens: [Token]) -> Expression:
        left = Parser.parse_primitive(tokens) 
        while tokens[0].token_type == TokenType.POWER:
            operator = tokens.pop(0)
            right = Parser.parse_power(tokens)
            left = BinaryOperator(left, operator, right)
        return left

    def parse_primitive(tokens: [Token]) -> Expression:
        match tokens[0].token_type:
            case TokenType.NUMBER:
                return Literal(tokens.pop(0))
            case TokenType.OPEN_PARENTHESES:
                return Parser.parse_parenthesised(tokens)
            case TokenType.ADD | TokenType.SUBTRACT:
                return Parser.parse_unary(tokens)
            case _:
                raise SyntaxError(f"Unexpected token when parsing primitive {tokens[0].token_type}")


class Evaluator():
    OPERATIONS = {
        TokenType.ADD: lambda l, r: l + r, 
        TokenType.SUBTRACT: lambda l,r: l - r,
        TokenType.MULTIPLY: lambda l,r: l * r,
        TokenType.DIVIDE: lambda l,r: l / r,
        TokenType.POWER: lambda l,r: l**r,
        TokenType.UNARY_PLUS: lambda o: o,
        TokenType.UNARY_MINUS: lambda o: -o
    }

    def evaluate(ast: Expression):
        if type(ast) is Literal:
            return ast.token.value
        elif type(ast) is BinaryOperator:
            return Evaluator.OPERATIONS[ast.operator.token_type](Evaluator.evaluate(ast.left), Evaluator.evaluate(ast.right))
        elif type(ast) is UnaryOperator:
            return Evaluator.OPERATIONS[ast.operator.token_type](Evaluator.evaluate(ast.operand))
        else:
            raise SyntaxError("Unable to evaluate")


class Compiler:
    def __init__(self):
        self.OPERATIONS = {
            TokenType.ADD: lambda l, r: l + r + [Op.ADD],
            TokenType.SUBTRACT: lambda l, r: l + r + [Op.SUB],
            TokenType.MULTIPLY: lambda l, r: l +  r + [Op.MUL],
            TokenType.DIVIDE: lambda l, r: l + r + [Op.DIV],
            TokenType.UNARY_PLUS: lambda o: o,
            TokenType.UNARY_MINUS: lambda o: [Op.PUSH] + [-1] + o + [Op.MUL],
        }
    def compile(self, ast:Expression):
        return self._compile(ast)+[Op.PRINT, Op.HALT]

    def _compile(self, ast: Expression):
        if type(ast) is Literal:
            return [Op.PUSH, ast.token.value]
        elif type(ast) is BinaryOperator:
            return self.OPERATIONS.get(ast.operator.token_type)(self._compile(ast.left), self._compile(ast.right))
        elif type(ast) is UnaryOperator:
            return  self.OPERATIONS.get(ast.operator.token_type)(self._compile(ast.operand))
        else:
            raise SyntaxError("Error - Unknown expression")

# Opcodes:
class Op(IntEnum):
    HALT = 0x01
    PUSH = 0x02
    PRINT = 0x03
    ADD = 0x04
    SUB = 0x05
    MUL = 0x06
    DIV = 0x07

class Vm:
    IMMEDIATES = {
        Op.HALT: 0,
        Op.PUSH: 1,
        Op.PRINT: 0,
        Op.ADD: 0,
        Op.SUB: 0,
        Op.MUL: 0,
        Op.DIV: 0,
    }

    def print_program(self):
        self.ip = 0
        while self.ip < self.program_length:
            instruction = Op(self.memory[self.ip])
            n_immediates = Vm.IMMEDIATES[instruction]
            immediates = self.memory[self.ip+1:self.ip+1+n_immediates]
            print(instruction.name, *immediates)
            self.ip += (1+n_immediates)

    def __init__(self, memory_size=1024):
        self.dispatcher = {
            Op.HALT: self.exec_halt,
            Op.PUSH: self.exec_push,
            Op.PRINT: self.exec_print,
            Op.ADD: self.exec_add,
            Op.SUB: self.exec_sub,
            Op.MUL: self.exec_mul,
            Op.DIV: self.exec_div
        }

        self.memory = [0]*memory_size
        self.ip = 0
        self.sp = 1

    def load_program(self, program):
        self.sp = self.program_length = len(program)
        self.memory[0:self.sp] = program

    def execute(self):
        self.ip = 0
        self.running = True
        while self.running and self.ip < self.program_length:
            # Load instruction
            instruction = Op(self.memory[self.ip])
            n_immediates = Vm.IMMEDIATES[instruction]
            immediates = self.memory[self.ip+1:self.ip+1+n_immediates]

            # Precalculate instruction pointer
            next_ip = self.ip + 1 + n_immediates
            if next_ip > self.program_length:
                raise Exception("Instruction pointer out of bounds!")

            # Execute instruction
            if len(immediates) > 0:
                self.dispatcher[instruction](immediates)
            else:
                self.dispatcher[instruction]()

            # Advance instruction pointer
            if self.running:
                self.ip = next_ip

    def push_to_stack(self, value):
        self.memory[self.sp] = value
        self.sp += 1
        if self.sp > len(self.memory):
            raise Exception("Memory overflow!")

    def pop_from_stack(self):
        self.sp -= 1
        if self.sp < self.program_length:
            raise Exception("Memory underflow!")
        return self.memory[self.sp]

    def exec_halt(self):
        self.running = False

    def exec_push(self, immediates):
        self.push_to_stack(immediates[0])

    def exec_print(self):
        print(self.memory[self.sp-1])

    def exec_add(self):
        r = self.pop_from_stack()
        l = self.pop_from_stack()
        self.push_to_stack(l+r)

    def exec_sub(self):
        r = self.pop_from_stack()
        l = self.pop_from_stack()
        self.push_to_stack(l-r)

    def exec_mul(self):
        r = self.pop_from_stack()
        l = self.pop_from_stack()
        self.push_to_stack(l*r)

    def exec_div(self):
        r = self.pop_from_stack()
        l = self.pop_from_stack()
        self.push_to_stack(l/r)
        

def test_evaluator():
    eval_expr = lambda s: Evaluator.evaluate(Parser.parse(Lexer.tokenize(s)))

    # Basic literals
    assert eval_expr("1") == 1
    assert eval_expr("42") == 42
    assert eval_expr("3.5") == 3.5

    # Basic arithmetic
    assert eval_expr("1 + 2") == 3
    assert eval_expr("5 - 2") == 3
    assert eval_expr("4 * 3") == 12
    assert eval_expr("8 / 2") == 4

    # Operator precedence
    assert eval_expr("2 + 3 * 4") == 14 
    assert eval_expr("2 * 3 + 4") == 10
    assert eval_expr("2 + 3 * 4 - 1") == 13

    # Parentheses override precedence
    assert eval_expr("(2 + 3) * 4") == 20
    assert eval_expr("2 * (3 + 4)") == 14
    assert eval_expr("(2 + 3) * (1 + 1)") == 10

    # Nested parentheses
    assert eval_expr("((1 + 2) * (3 + 4))") == 21
    assert eval_expr("(2 + (3 * 4))") == 14

    # Division
    assert eval_expr("10 / 2 + 3") == 8
    assert eval_expr("10 / (2 + 3)") == 2

    # Indirect multiplication
    assert eval_expr("2(3 + 4)") == 14
    assert eval_expr("(1 + 2)(3 + 4)") == 21
    assert eval_expr("2(3 + 4) + 1") == 15
    assert eval_expr("2(3 + 4)(5 + 1)") == 84
    assert eval_expr("(2 + 3)(4 + 1)(6)") == 150
    
    # Power operator
    assert eval_expr("2 ** 3") == 8
    assert eval_expr("4 ** 0.5") == 2.0
    assert eval_expr("5 ** 1") == 5
    assert eval_expr("9 ** 0") == 1
    assert eval_expr("2**3**2") == 512

    # Power with precedence over multiplication/addition
    assert eval_expr("2 + 3 ** 2") == 11
    assert eval_expr("2 * 3 ** 2") == 18
    assert eval_expr("2 ** 3 * 4") == 32
    assert eval_expr("2 * 3 ** 2 + 1") == 19

    # Parentheses with power
    assert eval_expr("(2 + 1) ** 2") == 9
    assert eval_expr("2 ** (1 + 2)") == 8
    assert eval_expr("(2 ** 3) ** 2") == 64
    assert eval_expr("2 ** (3 ** 2)") == 512

    # Nested and mixed operations
    assert eval_expr("(2 + 1)(3 ** 2)") == 27
    assert eval_expr("2(3 ** 2)") == 18
    assert eval_expr("(1 + 2)(2 ** 2)(1 + 1)") == 24

    # Edge and invalid cases
    assert eval_expr("2 ** -1") == 0.5
    assert eval_expr("4 ** -0.5") == 0.5
    # Unary plus
    assert eval_expr("+1") == 1
    assert eval_expr("+(2 + 3)") == 5
    assert eval_expr("+(+4)") == 4
    assert eval_expr("+(+(-5))") == -5

    # Unary minus
    assert eval_expr("-1") == -1
    assert eval_expr("-(2 + 3)") == -5
    assert eval_expr("-(-4)") == 4
    assert eval_expr("-(-(-6))") == -6

    # Unary operators with power
    assert eval_expr("-2 ** 2") == -4
    assert eval_expr("(-2) ** 2") == 4
    assert eval_expr("-(2 ** 2)") == -4
    assert eval_expr("+2 ** 3") == 8

    # Unary with multiplication and division
    assert eval_expr("-2 * 3") == -6
    assert eval_expr("2 * -3") == -6
    assert eval_expr("-2 * -3") == 6
    assert eval_expr("4 / -2") == -2
    assert eval_expr("-4 / -2") == 2

    # Unary with parentheses and nesting
    assert eval_expr("-(1 + 2) * 3") == -9
    assert eval_expr("-(1 + 2 * (3 + 4))") == -15
    assert eval_expr("+(((-3)))") == -3

    # Multiple unary operators
    assert eval_expr("--1") == 1
    assert eval_expr("---1") == -1
    assert eval_expr("----1") == 1
    assert eval_expr("+-+2") == -2
def test_vm_execution():
    import io
    from contextlib import redirect_stdout
    def eval_vm(expr: str) -> float:
        tokens = Lexer.tokenize(expr)
        ast = Parser.parse(tokens)
        bytecode = Compiler().compile(ast)
        vm = Vm()
        vm.load_program(bytecode)

        f = io.StringIO()
        with redirect_stdout(f):
            vm.execute()
        output = f.getvalue().strip()
        return float(output) if '.' in output else int(output)

    # Same tests as in test_evaluator, adapted for VM output
    assert eval_vm("1") == 1
    assert eval_vm("42") == 42
    assert eval_vm("3.5") == 3.5
    assert eval_vm("1 + 2") == 3
    assert eval_vm("5 - 2") == 3
    assert eval_vm("4 * 3") == 12
    assert eval_vm("8 / 2") == 4
    assert eval_vm("2 + 3 * 4") == 14 
    assert eval_vm("(2 + 3) * 4") == 20
    assert eval_vm("2 * (3 + 4)") == 14
    assert eval_vm("(1 + 2)(3 + 4)") == 21
    #assert eval_vm("2 ** 3") == 8
    #assert eval_vm("4 ** 0.5") == 2.0
    #assert eval_vm("2 ** 3 * 4") == 32
    #assert eval_vm("(2 + 1)(3 ** 2)") == 27
    assert eval_vm("-2 * 3") == -6
    assert eval_vm("2 * -3") == -6
    assert eval_vm("-2 * -3") == 6
    assert eval_vm("4 / -2") == -2
    assert eval_vm("-4 / -2") == 2
    assert eval_vm("--1") == 1
    assert eval_vm("+-+2") == -2
    #assert eval_vm("-2 ** 2") == -4
    #assert eval_vm("(-2) ** 2") == 4
    #assert eval_vm("-(2 ** 2)") == -4
    #assert eval_vm("+2 ** 3") == 8

def main():
    test_evaluator()
    test_vm_execution()
    print("All tests passed")
    while True:
        try:
            raw = input(">> ")
            if raw:
                iter = PeakIter(raw)
                tokens = Lexer.tokenize(raw)
            else:
                exit()
            ast = Parser.parse(tokens)
            bytecode = Compiler().compile(ast)
            vm = Vm()
            vm.load_program(bytecode)
            #vm.print_program()
            vm.execute()
            #output = Evaluator.evaluate(ast)
        except SyntaxError as err:
            print("SyntaxError: ", err)
            continue
        except ZeroDivisionError:
            print("Error: Can't divide by zero!")
            continue

if __name__ == "__main__":
    main()
