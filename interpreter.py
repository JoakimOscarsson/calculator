from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Any
# read input
# Tokenize
# syntax parsing
# Semantic-analysis
# Evaluation
# print

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
    @property
    def operation(self):
        return {
            TokenType.ADD: lambda l, r: l + r, 
            TokenType.SUBTRACT: lambda l,r: l - r,
            TokenType.MULTIPLY: lambda l,r: l * r,
            TokenType.DIVIDE: lambda l,r: l / r,
            TokenType.POWER: lambda l,r: l**r,
            TokenType.UNARY_PLUS: lambda o: o,
            TokenType.UNARY_MINUS: lambda o: -o
        }.get(self, None)

@dataclass
class Token:
    token_type: TokenType
    value: Any = None
    @property
    def operation(self):
        return self.token_type.operation

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
        

def tokenize(str):
    def build_number(number: str, iter: PeakIter):
        try:
            peaked = iter.peak()
        except StopIteration:
            return number
        if peaked in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]:
            return build_number(number+iter.next(), iter)
        else:
            return number
    
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
                            TokenType.NUMBER, float(build_number(char, iter))
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


def parse(tokens: [Token]) -> Expression:
    ast = parse_additive(tokens)
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
            
    operand = parse_power(tokens)
    return UnaryOperator(operator, operand)
    

def parse_parenthesised(tokens: [Token]) -> Expression:
    tokens.pop(0)
    expr = parse_additive(tokens)
    if tokens[0].token_type != TokenType.CLOSE_PARENTHESES:
        raise SyntaxError("Error, parentheses not closed")
    tokens.pop(0)
    return expr


def parse_additive(tokens: [Token]) -> Expression:
    left = parse_multiplicative(tokens) 
    while tokens[0].token_type == TokenType.ADD or tokens[0].token_type == TokenType.SUBTRACT:
        operator = tokens.pop(0)
        right = parse_multiplicative(tokens)
        left = BinaryOperator(left, operator, right)
    return left
    

def parse_multiplicative(tokens: [Token]) -> Expression:
    left = parse_power(tokens)
    while tokens[0].token_type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.OPEN_PARENTHESES]:
        if tokens[0].token_type ==  TokenType.OPEN_PARENTHESES:
            operator = Token(TokenType.MULTIPLY)
            right = parse_parenthesised(tokens)
        else:
            operator = tokens.pop(0)
            right = parse_power(tokens)
            if operator.token_type is TokenType.DIVIDE and right == 0:
                raise ZeroDivisionError()
        left = BinaryOperator(left, operator, right)
    return left


def parse_power(tokens: [Token]) -> Expression:
    left = parse_primitive(tokens) 
    while tokens[0].token_type == TokenType.POWER:
        operator = tokens.pop(0)
        right = parse_primitive(tokens)
        left = BinaryOperator(left, operator, right)
    return left


def parse_primitive(tokens: [Token]) -> Expression:
    match tokens[0].token_type:
        case TokenType.NUMBER:
            return Literal(tokens.pop(0))
        case TokenType.OPEN_PARENTHESES:
            return parse_parenthesised(tokens)
        case TokenType.ADD | TokenType.SUBTRACT:
            return parse_unary(tokens)
        case _:
            raise SyntaxError(f"Unexpected token when parsing primitive {tokens[0].token_type}")


def evaluator(ast: Expression):
    if type(ast) is Literal:
        return ast.token.value
    elif type(ast) is BinaryOperator:
        return ast.operator.operation(evaluator(ast.left), evaluator(ast.right))
    elif type(ast) is UnaryOperator:
        return ast.operator.operation(evaluator(ast.operand))
    else:
        raise SyntaxError("Unable to evaluate")


def test_evaluator():
    eval_expr = lambda s: evaluator(parse(tokenize(s)))

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
    assert eval_expr("0 ** 0") == 1
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

def main():
    test_evaluator()
    while True:
        try:
            raw = input(">> ")
            if raw:
                iter = PeakIter(raw)
                tokens = tokenize(raw)
            else:
                exit()
            ast = parse(tokens)
            print(evaluator(ast))
        except SyntaxError as err:
            print("SyntaxError: ", err)
            continue
        except ZeroDivisionError:
            print("Error: Can't divide by zero!")
            continue

if __name__ == "__main__":
    main()

