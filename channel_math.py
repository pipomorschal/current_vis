"""Restricted, elementwise channel arithmetic on a common measured time range."""
from __future__ import annotations

import ast
import keyword
import operator

import numpy as np


class ChannelMathError(ValueError):
    pass


def channel_aliases(labels) -> dict[str, str]:
    """Keep instrument names, and give non-identifier measurement names aliases."""
    labels = list(labels)
    reserved = {label for label in labels if label.isidentifier() and not keyword.iskeyword(label)}
    aliases = {}
    for index, label in enumerate(labels, 1):
        alias = label if label in reserved else f"S{index}"
        while alias in aliases or (label not in reserved and alias in reserved):
            alias = "_" + alias
        aliases[alias] = label
    return aliases


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow}
_REDUCERS = {"max": np.max, "min": np.min}


def evaluate_channel_math(expression: str, traces: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Use the first operand's timestamps; interpolate others within their overlap.

    Never execute Python code, extrapolate, or replace undefined results with zero.
    Nonfinite results remain NaN so the plot can show a gap.
    min/max reduce finite values on the aligned time range to scalar constants.
    """
    if not expression.strip():
        raise ChannelMathError("Enter an expression, for example CH1 + CH2 / CH3.")
    if len(expression) > 512:
        raise ChannelMathError("Keep each expression to 512 characters or fewer.")
    try:
        root = ast.parse(expression.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ChannelMathError("Invalid expression. Use channel names, numbers, min/max and + - * / ** ( ).") from exc
    if sum(1 for _ in ast.walk(root)) > 128:
        raise ChannelMathError("Expression is too complex; use fewer operations.")

    names = []

    def validate(node):
        if isinstance(node, ast.Name):
            if node.id not in traces:
                raise ChannelMathError(f"Unknown channel '{node.id}'. See the available names below.")
            if node.id not in names:
                names.append(node.id)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                finite = np.isfinite(float(node.value))
            except (ValueError, OverflowError):
                finite = False
            if not finite:
                raise ChannelMathError("Constants must be finite numbers.")
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _REDUCERS:
            if len(node.args) != 1 or node.keywords:
                raise ChannelMathError(f"{node.func.id}() requires exactly one expression, for example {node.func.id}(S1).")
            validate(node.args[0])
        else:
            raise ChannelMathError("Use channel names, numbers, min/max and + - * / ** ( ); other functions are not supported.")

    validate(root)
    if not names:
        raise ChannelMathError("Include at least one channel in the expression.")

    operands = {}
    for name in names:
        x, y = (np.asarray(values, dtype=float) for values in traces[name])
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ChannelMathError(f"{name}: timestamps and values must be matching vectors.")
        mask = np.isfinite(x)
        x, y = x[mask], y[mask]
        if not x.size:
            raise ChannelMathError(f"{name} has no samples in the selected time range.")
        if np.any(np.diff(x) <= 0):
            raise ChannelMathError(f"{name}: timestamps must be strictly increasing for channel math.")
        operands[name] = (x, np.where(np.isfinite(y), y, np.nan))
    start = max(x[0] for x, _ in operands.values())
    end = min(x[-1] for x, _ in operands.values())
    first = operands[names[0]][0]
    time = first[(first >= start) & (first <= end)]
    if not time.size:
        raise ChannelMathError("The referenced channels have no common sampled time range.")
    values = {name: y if np.array_equal(x, time) else np.interp(time, x, y)
              for name, (x, y) in operands.items()}

    def calculate(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant):
            # NumPy scalars keep divide-by-zero and overflow in array semantics.
            return np.float64(node.value)
        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call):
            argument = np.asarray(calculate(node.args[0]), dtype=float)
            finite = argument[np.isfinite(argument)]
            if not finite.size:
                raise ChannelMathError(f"{node.func.id}() has no finite values in the selected time range.")
            return _REDUCERS[node.func.id](finite)
        return _OPERATORS[type(node.op)](calculate(node.left), calculate(node.right))

    with np.errstate(all="ignore"):
        result = np.asarray(calculate(root), dtype=float)
    if result.ndim == 0:
        result = np.full(time.shape, result.item())
    result = np.where(np.isfinite(result), result, np.nan)
    return time, result
