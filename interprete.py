"""Interprete de un pseudolenguaje basado en etiquetas XML.

Este módulo realiza:
    * Análisis léxico mediante expresiones regulares.
    * Análisis sintáctico con un parser descendente recursivo.
    * Reporte de los elementos válidos e inválidos encontrados.

El pseudolenguaje soporta las etiquetas principales <funcion>, <parametros>,
<codigo>, <if>, <do> y <condicion>. Dentro del código se permiten asignaciones
con operadores aritméticos y expresiones booleanas con operadores lógicos.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. ANALIZADOR LÉXICO
# ---------------------------------------------------------------------------

TOKEN_SPECIFICATION = [
    ("TAG_CLOSE", r"</[a-zA-Z]+>"),
    ("TAG_OPEN", r"<[a-zA-Z]+>"),
    ("LOGICAL_AND", r"&&"),
    ("LOGICAL_OR", r"\|\|"),
    ("NE", r"!="),
    ("EQ", r"=="),
    ("GE", r">="),
    ("LE", r"<="),
    ("ASSIGN", r"="),
    ("GT", r">"),
    ("LT", r"<"),
    ("LOGICAL_NOT", r"!"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("NUMBER", r"\d+"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("MISMATCH", r"."),
]

TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPECIFICATION)
)


@dataclass
class Token:
    """Representa un token reconocido por el analizador léxico."""

    type: str
    value: str
    line: int
    column: int


class LexerResult:
    """Contiene la lista de tokens válidos y los errores léxicos detectados."""

    def __init__(self, tokens: List[Token], errors: List[str]):
        self.tokens = tokens
        self.errors = errors


def tokenize(code: str) -> LexerResult:
    """Tokeniza el código fuente usando las expresiones regulares definidas."""

    tokens: List[Token] = []
    errors: List[str] = []
    line = 1
    column = 1

    for match in TOKEN_REGEX.finditer(code):
        kind = match.lastgroup
        value = match.group()

        if kind == "NEWLINE":
            line += 1
            column = 1
            continue
        if kind == "SKIP":
            column += len(value)
            continue
        if kind == "MISMATCH":
            errors.append(
                f"Error léxico: caracter inesperado '{value}' en línea {line}, columna {column}."
            )
            column += len(value)
            continue

        tokens.append(Token(kind, value, line, column))
        column += len(value)

    tokens.append(Token("EOF", "", line, column))
    return LexerResult(tokens, errors)


# ---------------------------------------------------------------------------
# 2. ANALIZADOR SINTÁCTICO
# ---------------------------------------------------------------------------

class ParserError(Exception):
    """Errores producidos durante el análisis sintáctico."""


class Parser:
    """Parser descendente recursivo para validar el pseudolenguaje."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.stats: Dict[str, int] = {
            "Funciones válidas": 0,
            "Funciones inválidas": 0,
            "Parámetros válidos": 0,
            "Parámetros inválidos": 0,
            "Asignaciones válidas": 0,
            "Asignaciones inválidas": 0,
            "If válidos": 0,
            "If inválidos": 0,
            "Do válidos": 0,
            "Do inválidos": 0,
            "Condiciones válidas": 0,
            "Condiciones inválidas": 0,
        }
        self.errors: List[str] = []
        # Contaremos “Asignaciones válidas” como VARS asignadas distintas
        self._assigned_vars: set[str] = set()

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------

    def parse(self) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_OPEN" and self.tag_name(token) == "funcion":
                self.parse_function()
            else:
                self.errors.append(
                    f"Error sintáctico: se esperaba <funcion> y se encontró '{token.value}' (línea {token.line})."
                )
                self.advance()

        # al terminar, fija el total de asignaciones válidas por variables únicas
        self.stats["Asignaciones válidas"] = len(self._assigned_vars)

    # ------------------------------------------------------------------
    # Reglas principales
    # ------------------------------------------------------------------

    def parse_function(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("funcion", closing=False)
            params_valid = self.parse_parameters()
            self.parse_code_block()
            self.expect_tag("funcion", closing=True)
            if not params_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <funcion> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("funcion")

        if is_valid:
            self.stats["Funciones válidas"] += 1
        else:
            self.stats["Funciones inválidas"] += 1

    def parse_parameters(self) -> bool:
        valid_count = 0
        invalid_count = 0
        is_block_valid = True

        # <parametros> de apertura
        self.expect_tag("parametros", closing=False)
        expecting_value = True

        while True:
            token = self.current_token()

            # ¿llegamos al cierre </parametros>?
            if token.type == "TAG_CLOSE" and self.tag_name(token) == "parametros":
                break
            if token.type == "EOF":
                raise ParserError("fin de archivo inesperado en <parametros>.")

            if expecting_value:
                if token.type in {"IDENT", "NUMBER"}:
                    self.advance()
                    valid_count += 1
                    expecting_value = False
                else:
                    invalid_count += 1
                    is_block_valid = False
                    self.errors.append(
                        f"Error sintáctico: parámetro inválido '{token.value}' en línea {token.line}."
                    )
                    self.consume_until_closing_tag("parametros")
                    break
            else:
                # después de un valor esperamos coma o el cierre
                if token.type == "COMMA":
                    self.advance()
                    expecting_value = True
                elif token.type == "TAG_CLOSE" and self.tag_name(token) == "parametros":
                    break
                else:
                    is_block_valid = False
                    self.errors.append(
                        f"Error sintáctico: se esperaba ',' en línea {token.line}."
                    )
                    self.consume_until_closing_tag("parametros")
                    break

        # </parametros> de cierre
        self.expect_tag("parametros", closing=True)

        # actualizar métricas por elemento
        self.stats["Parámetros válidos"] += valid_count
        self.stats["Parámetros inválidos"] += invalid_count

        return is_block_valid and (invalid_count == 0)

    def parse_code_block(self) -> None:
        start_line = self.current_token().line
        self.expect_tag("codigo", closing=False)

        while True:
            token = self.current_token()

            if token.type == "TAG_CLOSE" and self.tag_name(token) == "codigo":
                break
            if token.type == "EOF":
                raise ParserError(
                    f"fin de archivo inesperado: falta </codigo> (línea {start_line})."
                )

            if token.type == "IDENT":
                self.parse_assignment()
                continue

            if token.type == "TAG_OPEN":
                tag = self.tag_name(token)
                if tag == "if":
                    self.parse_if()
                    continue
                if tag == "do":
                    self.parse_do()
                    continue

                self.errors.append(
                    f"Error sintáctico: etiqueta <{tag}> no permitida dentro de <codigo> (línea {token.line})."
                )
                self.skip_unknown_tag(tag)
                continue

            self.errors.append(
                f"Error sintáctico: elemento inesperado '{token.value}' dentro de <codigo> (línea {token.line})."
            )
            self.advance()

        self.expect_tag("codigo", closing=True)

    def parse_assignment(self) -> None:
        start_line = self.current_token().line
        is_valid = True
        lhs_name = None  # nombre del LHS (variable a la izquierda)

        try:
            # capturamos el IDENT antes de consumirlo
            tok_ident = self.current_token()
            self.expect_type("IDENT")
            lhs_name = tok_ident.value

            self.expect_type("ASSIGN")
            self.parse_expression()
            self.expect_type("SEMICOLON")
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: asignación inválida en línea {start_line}. Detalle: {exc}"
            )
            self.recover_after_assignment_error()

        if is_valid:
            # contar variables asignadas distintas (no sentencias)
            if lhs_name is not None:
                self._assigned_vars.add(lhs_name)
        else:
            self.stats["Asignaciones inválidas"] += 1

    def parse_if(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("if", closing=False)
            condition_valid = self.parse_condition()
            self.parse_code_block()
            self.expect_tag("if", closing=True)
            if not condition_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <if> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("if")

        if is_valid:
            self.stats["If válidos"] += 1
        else:
            self.stats["If inválidos"] += 1

    def parse_do(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("do", closing=False)
            self.parse_code_block()
            condition_valid = self.parse_condition()
            self.expect_tag("do", closing=True)
            if not condition_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <do> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("do")

        if is_valid:
            self.stats["Do válidos"] += 1
        else:
            self.stats["Do inválidos"] += 1

    def parse_condition(self) -> bool:
        start_line = self.current_token().line
        is_valid = True

        self.expect_tag("condicion", closing=False)
        try:
            self.parse_expression()
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: condición inválida en línea {start_line}. Detalle: {exc}"
            )
            self.consume_until_closing_tag("condicion")

        self.expect_tag("condicion", closing=True)

        if is_valid:
            self.stats["Condiciones válidas"] += 1
        else:
            self.stats["Condiciones inválidas"] += 1
        return is_valid

    # ------------------------------------------------------------------
    # Expresiones
    # ------------------------------------------------------------------

    def parse_expression(self) -> None:
        self.parse_logical_or()

    def parse_logical_or(self) -> None:
        self.parse_logical_and()
        while self.match("LOGICAL_OR"):
            self.parse_logical_and()

    def parse_logical_and(self) -> None:
        self.parse_equality()
        while self.match("LOGICAL_AND"):
            self.parse_equality()

    def parse_equality(self) -> None:
        self.parse_relational()
        while self.match("EQ", "NE"):
            self.parse_relational()

    def parse_relational(self) -> None:
        self.parse_additive()
        while self.match("GT", "LT", "GE", "LE"):
            self.parse_additive()

    def parse_additive(self) -> None:
        self.parse_multiplicative()
        while self.match("PLUS", "MINUS"):
            self.parse_multiplicative()

    def parse_multiplicative(self) -> None:
        self.parse_unary()
        while self.match("TIMES", "DIVIDE"):
            self.parse_unary()

    def parse_unary(self) -> None:
        if self.match("LOGICAL_NOT", "MINUS"):
            self.parse_unary()
        else:
            self.parse_primary()

    def parse_primary(self) -> None:
        token = self.current_token()
        if token.type in {"IDENT", "NUMBER"}:
            self.advance()
            return
        if token.type == "LPAREN":
            self.advance()
            self.parse_expression()
            self.expect_type("RPAREN")
            return
        raise ParserError(
            f"se esperaba identificador, número o '(' y se encontró '{token.value}' (línea {token.line})."
        )

    # ------------------------------------------------------------------
    # Utilidades del parser
    # ------------------------------------------------------------------

    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> None:
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def match(self, *types: str) -> bool:
        token = self.current_token()
        if token.type in types:
            self.advance()
            return True
        return False

    def expect_type(self, token_type: str) -> None:
        token = self.current_token()
        if token.type == token_type:
            self.advance()
            return
        raise ParserError(
            f"se esperaba un token de tipo {token_type} y se encontró '{token.value}' (línea {token.line})."
        )

    def expect_tag(self, name: str, *, closing: bool) -> None:
        token = self.current_token()
        expected_type = "TAG_CLOSE" if closing else "TAG_OPEN"
        if token.type == expected_type and self.tag_name(token) == name:
            self.advance()
            return
        closing_text = "/" if closing else ""
        raise ParserError(
            f"se esperaba <{closing_text}{name}> y se encontró '{token.value}' (línea {token.line})."
        )

    @staticmethod
    def tag_name(token: Token) -> str:
        if token.type == "TAG_OPEN":
            return token.value[1:-1]
        if token.type == "TAG_CLOSE":
            return token.value[2:-1]
        return ""

    def consume_until_closing_tag(self, name: str) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                break
            self.advance()

    def recover_until_closing_tag(self, name: str) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                self.advance()
                break
            self.advance()

    def skip_unknown_tag(self, name: str) -> None:
        depth = 0
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_OPEN" and self.tag_name(token) == name:
                depth += 1
            elif token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                depth -= 1
                self.advance()
                if depth <= 0:
                    break
                continue
            self.advance()

    def recover_after_assignment_error(self) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "SEMICOLON":
                self.advance()
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) in {"codigo", "if", "do"}:
                break
            self.advance()


# ---------------------------------------------------------------------------
# 3. EJECUCIÓN Y REPORTE
# ---------------------------------------------------------------------------

def generate_report(stats: Dict[str, int], lexer_errors: List[str], parser_errors: List[str]) -> None:
    """Imprime un reporte con el resultado del análisis."""

    print("--- REPORTE DE VALIDACIÓN ---")
    print(f"Funciones válidas: {stats['Funciones válidas']}")
    print(f"Funciones inválidas: {stats['Funciones inválidas']}")
    print(f"Parámetros válidos: {stats['Parámetros válidos']}")
    print(f"Parámetros inválidos: {stats['Parámetros inválidos']}")
    print(f"Asignaciones válidas: {stats['Asignaciones válidas']}")
    print(f"Asignaciones inválidas: {stats['Asignaciones inválidas']}")
    print(f"If válidos: {stats['If válidos']}")
    print(f"If inválidos: {stats['If inválidos']}")
    print(f"Do válidos: {stats['Do válidos']}")
    print(f"Do inválidos: {stats['Do inválidos']}")
    print(f"Condiciones válidas: {stats['Condiciones válidas']}")
    print(f"Condiciones inválidas: {stats['Condiciones inválidas']}")
    print(f"Errores léxicos: {len(lexer_errors)}")
    print(f"Errores sintácticos: {len(parser_errors)}")
    print("-----------------------------")

    if lexer_errors or parser_errors:
        print("\nDetalle de errores:")
        for error in lexer_errors:
            print(error)
        for error in parser_errors:
            print(error)


def main(filename: str) -> None:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            code = file.read()
    except FileNotFoundError:
        print(f"Error: el archivo de entrada '{filename}' no fue encontrado.")
        sys.exit(1)

    lexer_result = tokenize(code)
    parser = Parser(lexer_result.tokens)
    parser.parse()

    generate_report(parser.stats, lexer_result.errors, parser.errors)


if __name__ == "__main__":
    input_file = "entrada.txt"
    input_path = Path(input_file)

    if not input_path.exists():
        # ejemplo sin comillas extra
        input_path.write_text(
            "<funcion>\n"
            "<parametros>x, y, limite</parametros>\n"
            "<codigo>\n"
            "x = 5;\n"
            "y = (x + 3) * 2;\n"
            "<if>\n"
            "<condicion>(x + y) > 10 && limite != 0</condicion>\n"
            "<codigo>\n"
            "resultado = y / limite;\n"
            "</codigo>\n"
            "</if>\n"
            "<do>\n"
            "<codigo>\n"
            "x = x + 1;\n"
            "</codigo>\n"
            "<condicion>x < limite || limite == 0</condicion>\n"
            "</do>\n"
            "</codigo>\n"
            "</funcion>\n",
            encoding="utf-8",
        )
        print(
            f"Se creó un archivo de ejemplo '{input_file}'. Modifíquelo y ejecute nuevamente."
        )

    main(input_file)
