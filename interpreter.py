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
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/" 

    OPEN_PARENTHESES = "("
    CLOSE_PARENTHESES = ")" 

    NUMBER = "num" 
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
        return self.iterator.next()

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
    tokens = []
    iter = PeakIter(str)
    while True:
        try:
            char = iter.next()
            match char: 
                case "+":
                    tokens.append(Token(TokenType.ADD, lambda l, r: l + r)) 
                case "-":
                    tokens.append(Token(TokenType.SUBTRACT, lambda l,r: l - r))
                case "*":
                    tokens.append(Token(TokenType.MULTIPLY, lambda l,r: l * r))
                case "/":
                    tokens.append(Token(TokenType.DIVIDE, lambda l,r: l / r))
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
        
def build_number(number: str, iter: PeakIter):
    try:
        peaked = iter.peak()
    except StopIteration:
        return number
    if peaked in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]:
        return build_number(number+iter.next(), iter)
    else:
        return number


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


def parse(tokens: [Token]) -> Expression:
    ast = parse_additive(tokens)
    if tokens[0].token_type != TokenType.EOF:
        raise SyntaxError("Error, expected end of file!")
    return ast


def parse_parenthesised(tokens: [Token]) -> Expression:
    tokens.pop(0)
    expr = parse_additive(tokens)
    if tokens[0].token_type != TokenType.CLOSE_PARENTHESES:
        raise SyntaxError("Error, parentheses not closed")
    tokens.pop(0)
    return expr


def parse_primitive(tokens: [Token]) -> Expression:
    match tokens[0].token_type:
        case TokenType.NUMBER:
            return Literal(tokens.pop(0))
        case TokenType.OPEN_PARENTHESES:
            return parse_parenthesised(tokens)
        case _:
            raise SyntaxError(f"Unexpected token {tokens[0].token_type}")

def parse_additive(tokens: [Token]) -> Expression:
    left = parse_multiplicative(tokens) 
    while tokens[0].token_type == TokenType.ADD or tokens[0].token_type == TokenType.SUBTRACT:
        operator = tokens.pop(0)
        right = parse_multiplicative(tokens)
        left = BinaryOperator(left, operator, right)
    return left
    

def parse_multiplicative(tokens: [Token]) -> Expression:
    left = parse_primitive(tokens)
    while tokens[0].token_type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.OPEN_PARENTHESES]:
        if tokens[0].token_type ==  TokenType.OPEN_PARENTHESES:
            operator = Token(TokenType.MULTIPLY, lambda l,r: l*r)
            right = parse_parenthesised(tokens)
        else:
            operator = tokens.pop(0)
            right = parse_primitive(tokens)
        left = BinaryOperator(left, operator, right)
    return left


def evaluator(ast: Expression):
    if type(ast) is Literal:
        return ast.token.value
    elif type(ast) is BinaryOperator:
        return ast.operator.value(evaluator(ast.left), evaluator(ast.right))
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
        except (SyntaxError) as err:
            print("SyntaxError: ", err)
            continue

if __name__ == "__main__":
    main()

