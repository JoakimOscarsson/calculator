from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol
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
    value: int = None

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
                    tokens.append(Token(TokenType.ADD))
                case "-":
                    tokens.append(Token(TokenType.SUBTRACT))
                case "*":
                    tokens.append(Token(TokenType.MULTIPLY))
                case "/":
                    tokens.append(Token(TokenType.DIVIDE))
                case "(":
                    tokens.append(Token(TokenType.OPEN_PARENTHESES))
                case ")":
                    tokens.append(Token(TokenType.CLOSE_PARENTHESES))
                case " ":
                    pass
                case "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                    tokens.append(
                        Token(
                            TokenType.NUMBER,
                            value = float(build_number(char, iter))
                        )
                    )
                case _:
                    print(f"Invalid Expression {char}")
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
    def print(self) -> str:
        ...


@dataclass
class Literal():
    token: Token
    def print(self) -> str:
        return str(self.token.value)
 

@dataclass
class BinaryOperator():
    left: Expression
    operator: Token
    right: Expression 
    def print(self) -> str:
        return f"({self.left.print()} {self.operator.token_type.value} {self.right.print()})"


def parse(tokens: [Token]) -> Expression:
    ast = parse_additive(tokens)
    if tokens[0].token_type != TokenType.EOF:
        print("Error, expected end of file!")
    return ast


def parse_parenthesised(tokens: [Token]) -> Expression:
    tokens.pop(0)
    expr = parse_additive(tokens)
    if tokens[0].token_type != TokenType.CLOSE_PARENTHESES:
        print("Error, parentheses not closed")
        return None
    tokens.pop(0)
    return expr


def parse_primitive(tokens: [Token]) -> Expression:
    match tokens[0].token_type:
        case TokenType.NUMBER:
            return Literal(tokens.pop(0))
        case TokenType.OPEN_PARENTHESES:
            return parse_parenthesised(tokens)
        case _:
            print(f"Unexpected token {tokens[0].token_type}") #  TODO: Implement custom error which is thrown to the top
            return None


def parse_additive(tokens: [Token]) -> Expression:
    left = parse_multiplicative(tokens) 
    while tokens[0].token_type == TokenType.ADD or tokens[0].token_type == TokenType.SUBTRACT:
        operator = tokens.pop(0)
        right = parse_multiplicative(tokens)
        left = BinaryOperator(left, operator, right)
    while tokens[0].token_type == TokenType.OPEN_PARENTHESES:
        operator = Token(TokenType.MULTIPLY)
        right = parse_parenthesised(tokens)
        left = BinaryOperator(left, operator, right)
    return left
    

def parse_multiplicative(tokens: [Token]) -> Expression:
    left = parse_primitive(tokens)
    while tokens[0].token_type == TokenType.MULTIPLY or tokens[0].token_type == TokenType.DIVIDE:
        operator = tokens.pop(0)
        right = parse_primitive(tokens)
        left = BinaryOperator(left, operator, right)
    return left


def main():
    while True:
        raw = input(">> ")
        if raw:
            iter = PeakIter(raw)
            tokens = tokenize(raw)
        else:
            exit()

        for token in tokens:
            print(token.token_type.name)
        
        ast = parse(tokens)
        print(ast.print())

if __name__ == "__main__":
    main()
