"""
NexusData Studio Pro - Scientific & Data Science Calculator

Tab 1: Scientific Calculator (keypad, trig with RAD/DEG, memory)
Tab 2: Pandas & Data Analysis (describe, corr, cov, mean, std, skew)
Tab 3: Calculus & SciPy (integral, derivatives, roots, n!, nCr, nPr)
Tab 4: Statistics & Graphs (normal curve, histogram, box plot, scatter +
       regression, line, bar, correlation heatmap, summary table)"""


import math
import re
import statistics
import tkinter as tk
from collections import Counter
from tkinter import ttk, messagebox, filedialog

# ---------------------------------------------------------------------------
# Optional libraries (the app keeps working without them)
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.integrate import quad
    from scipy.optimize import root_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Colours (GitHub dark theme)
# ---------------------------------------------------------------------------
BG = "#0D1117"
CARD = "#161B22"
PANEL = "#21262D"
KEY = "#1F242C"
BORDER = "#30363D"
TEXT = "#F0F6FC"
MUTED = "#8B949E"
BLUE = "#58A6FF"
GREEN = "#3FB950"
RED = "#F85149"
AMBER = "#D29922"
PURPLE = "#BC8CFF"

GRAPH_TYPES = [
    "Normal Curve",
    "Histogram + Normal Fit",
    "Box Plot",
    "Scatter + Regression Line",
    "Line Plot",
    "Bar Chart (Frequency)",
    "Correlation Heatmap",
]


# ---------------------------------------------------------------------------
# Input validation: only calculator symbols are allowed
# ---------------------------------------------------------------------------
class InvalidInput(ValueError):
    """Raised when something is typed that the calculator does not support."""

    def __init__(self, message="Invalid Input"):
        super().__init__(message)


ALLOWED_FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan",
                 "sqrt", "log", "ln", "log10", "exp", "abs"}
ALLOWED_CONSTS = {"pi", "e"}

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*|\.\d+)|(?P<name>[A-Za-z_]\w*)|(?P<op>\*\*|[+\-*/%(),]))"
)


def sanitize_expression(expr, variables=()):
    """Check every token of `expr`. Anything the calculator does not have
    (letters, words, symbols such as ^ # ; [ ] etc.) raises InvalidInput.
    Returns a clean string that is safe to evaluate."""
    expr = expr.strip()
    if not expr or len(expr) > 300:
        raise InvalidInput()

    pos, out = 0, []
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if m is None or m.end() == pos:
            raise InvalidInput()
        pos = m.end()

        if m.group("num"):
            value = float(m.group("num"))
            if not math.isfinite(value):
                raise InvalidInput()
            out.append(repr(value))          # every number becomes a float
        elif m.group("name"):
            name = m.group("name")
            followed_by_bracket = expr[pos:].lstrip().startswith("(")
            if name in ALLOWED_FUNCS:
                if not followed_by_bracket:
                    raise InvalidInput()
            elif name in ALLOWED_CONSTS or name in variables:
                if followed_by_bracket:
                    raise InvalidInput()
            else:
                raise InvalidInput()
            out.append(name)
        else:
            out.append(m.group("op"))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (used by the Statistics tab)
# ---------------------------------------------------------------------------
def _try_float(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_dataset(raw):
    """Turn pasted text into {column name: [floats]}.
    Accepts CSV, tab, semicolon or space separated numbers. First row may be
    a header. A single line of numbers is treated as one column."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise InvalidInput("Invalid Input: no data entered")

    rows = [[t for t in re.split(r"\s*[,;\t]\s*|\s+", ln) if t] for ln in lines]

    header = None
    if any(_try_float(t) is None for t in rows[0]):
        header, rows = rows[0], rows[1:]
    if not rows:
        raise InvalidInput("Invalid Input: no numeric rows found")

    if header is None and len(rows) == 1 and len(rows[0]) > 1:
        rows = [[t] for t in rows[0]]            # one line of numbers = one column

    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise InvalidInput("Invalid Input: every row needs the same number of values")
        for token in row:
            if _try_float(token) is None:
                raise InvalidInput(f"Invalid Input: '{token}' is not a number")

    if header is not None and len(header) != width:
        raise InvalidInput("Invalid Input: header and data have different column counts")

    names = header if header else [f"Col{i + 1}" for i in range(width)]
    seen, unique = {}, []
    for name in names:                            # make duplicate names unique
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    return {unique[i]: [float(r[i]) for r in rows] for i in range(width)}


def percentile(sorted_values, p):
    k = (len(sorted_values) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def linear_regression(x, y):
    n = len(x)
    if n < 2:
        raise InvalidInput("Invalid Input: need at least 2 points for a regression line")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        raise InvalidInput("Invalid Input: all X values are identical - cannot fit a line")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    r = pearson(x, y)
    return slope, intercept, r, (r * r if r is not None else None)


def describe(values):
    """Full descriptive statistics as a list of (label, value) pairs."""
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    median = statistics.median(s)
    m2 = sum((v - mean) ** 2 for v in values) / n
    pstd = math.sqrt(m2)
    svar = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else None
    sstd = math.sqrt(svar) if svar is not None else None
    q1, q3 = percentile(s, 25), percentile(s, 75)
    iqr = q3 - q1
    outliers = sum(1 for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr)

    modes = statistics.multimode(s)
    mode = "none" if len(modes) == len(set(s)) else modes[0]

    skew = kurt = None
    if m2 > 0:
        m3 = sum((v - mean) ** 3 for v in values) / n
        m4 = sum((v - mean) ** 4 for v in values) / n
        if n > 2:
            skew = math.sqrt(n * (n - 1)) / (n - 2) * (m3 / m2 ** 1.5)
        if n > 3:
            g2 = m4 / m2 ** 2 - 3
            kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)

    return [
        ("Count (n)", n), ("Mean", mean), ("Median", median), ("Mode", mode),
        ("Std Dev (population)", pstd), ("Std Dev (sample)", sstd), ("Variance (sample)", svar),
        ("Minimum", s[0]), ("Maximum", s[-1]), ("Range", s[-1] - s[0]),
        ("Q1 (25%)", q1), ("Q3 (75%)", q3), ("IQR", iqr),
        ("Skewness", skew), ("Kurtosis (excess)", kurt), ("Outliers (1.5×IQR)", outliers),
    ]


def fmt_value(v):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if v != v:
        return "n/a"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------
class NexusDataScienceStudio(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("")
        self.geometry("1180x860")
        self.minsize(980, 740)
        self.configure(bg=BG)

        # State
        self.memory_register = 0.0
        self.current_angle_unit = tk.StringVar(value="RAD")
        self._display_state = "typing"      # "typing", "result" or "message"
        self._stats_drawn = False
        self.stats_fig = None
        self.stats_canvas = None

        self._configure_styles()
        self._build_top_app_bar()
        self._build_main_workbench()
        self._bind_hotkeys()
        self.entry_display.focus_set()
        self.update_idletasks()
        self._hide_title_icon()

    # ------------------------------------------------------- title bar icon
    def _hide_title_icon(self):
        """Windows only: remove the icon from the title bar completely."""
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_DLGMODALFRAME = 0x0001          # a window with this style shows no icon
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_DLGMODALFRAME)
            # SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0020)
        except Exception:
            pass

    # ------------------------------------------------------------------ styles
    def _configure_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure("TNotebook", background=BG, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                             font=("Segoe UI", 9, "bold"), padding=[10, 7], borderwidth=0)
        self.style.map("TNotebook.Tab",
                       background=[("selected", PANEL)],
                       foreground=[("selected", BLUE)])

        self.style.configure("TCombobox", fieldbackground=CARD, background=PANEL,
                             foreground=TEXT, arrowcolor=BLUE, bordercolor=BORDER)
        self.style.map("TCombobox",
                       fieldbackground=[("readonly", CARD)],
                       foreground=[("readonly", TEXT)],
                       selectbackground=[("readonly", CARD)],
                       selectforeground=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", PANEL)
        self.option_add("*TCombobox*Listbox.selectForeground", BLUE)

    # ----------------------------------------------------------------- hotkeys
    def _bind_hotkeys(self):
        self.bind("<Return>", self._on_enter)
        self.bind("<KP_Enter>", self._on_enter)
        self.bind("<Escape>", lambda e: self.clear_all())
        self.entry_display.bind("<KeyPress>", self._on_display_key)

    def _on_enter(self, event):
        """Enter = '='. In multi-line text boxes Enter still adds a new line."""
        if isinstance(event.widget, tk.Text):
            return None
        on_calculator_tab = self.notebook.index(self.notebook.select()) == 0
        typing_in_display = (event.widget is self.entry_display
                             and "," not in self.input_var.get())
        if on_calculator_tab or typing_in_display:
            self.evaluate_expression()
            return "break"
        return None

    def _on_display_key(self, event):
        """Keyboard typing in the display: start fresh after a message/result."""
        if event.char and event.char.isprintable():
            state = self._display_state
            current = self.input_var.get()
            continues_result = event.char in "+-*/%"
            if (state == "message" or current == "0"
                    or (state == "result" and not continues_result)):
                self.input_var.set("")
                self.entry_display.icursor(tk.END)
            self._display_state = "typing"

    def _on_tab_changed(self, event):
        # Tab index 3 = "4. Stats & Graphs" (draw the first graph when opened)
        if (self.notebook.index(self.notebook.select()) == 3
                and not self._stats_drawn):
            self._stats_drawn = True
            self._stats_plot()

    # ------------------------------------------------------------------ top bar
    def _build_top_app_bar(self):
        top_bar = tk.Frame(self, bg=CARD, height=50, padx=16)
        top_bar.pack(fill="x", side="top")

        title_box = tk.Frame(top_bar, bg=CARD)
        title_box.pack(side="left", pady=10)

        tk.Label(title_box, text="Scientific Calculator", font=("JetBrains Mono", 12, "bold"),
                 fg=BLUE, bg="#1F242C", padx=8, pady=2).pack(side="left")

        ctrl_box = tk.Frame(top_bar, bg=CARD)
        ctrl_box.pack(side="right", pady=10)
        for mode in ["RAD", "DEG"]:
            tk.Radiobutton(ctrl_box, text=mode, variable=self.current_angle_unit,
                           value=mode, font=("Segoe UI", 8, "bold"), fg=BLUE, bg=CARD,
                           selectcolor=PANEL, activebackground=CARD).pack(side="left", padx=2)

    # --------------------------------------------------------------- main layout
    def _build_main_workbench(self):
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=14, pady=10)

        left_pane = tk.Frame(container, bg=BG)
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_pane = tk.Frame(container, bg=CARD, width=290,
                              highlightthickness=1, highlightbackground=BORDER)
        right_pane.pack(side="right", fill="y")
        right_pane.pack_propagate(False)

        self._build_dual_display(left_pane)
        self._build_notebook_tabs(left_pane)
        self._build_tape_sidebar(right_pane)

    def _build_dual_display(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightthickness=1,
                        highlightbackground=BORDER, padx=16, pady=10)
        card.pack(fill="x", pady=(0, 10))

        self.formula_var = tk.StringVar(value="")
        tk.Label(card, textvariable=self.formula_var, font=("JetBrains Mono", 11),
                 fg=MUTED, bg=CARD, anchor="e").pack(fill="x", pady=(0, 2))

        self.input_var = tk.StringVar(value="0")
        self.entry_display = tk.Entry(card, textvariable=self.input_var,
                                      font=("JetBrains Mono", 24, "bold"), fg=TEXT, bg=CARD,
                                      bd=0, justify="right", insertbackground=BLUE)
        self.entry_display.pack(fill="x", pady=(0, 4))

        strip = tk.Frame(card, bg=CARD)
        strip.pack(fill="x")

        self.mem_badge = tk.Label(strip, text="MEM: 0", font=("JetBrains Mono", 9),
                                  fg=GREEN, bg=CARD)
        self.mem_badge.pack(side="left")

        quick = tk.Frame(strip, bg=CARD)
        quick.pack(side="right")
        for label, cmd in [("M+", self._mem_add), ("M-", self._mem_sub),
                           ("MR", self._mem_recall), ("MC", self._mem_clear)]:
            tk.Button(quick, text=label, font=("JetBrains Mono", 8, "bold"), fg=MUTED,
                      bg=PANEL, activebackground=BORDER, activeforeground="#FFFFFF",
                      bd=0, padx=6, pady=1, cursor="hand2",
                      command=cmd).pack(side="left", padx=2)

    def _build_notebook_tabs(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        tabs = [
            ("1. Calculator", self._populate_scientific_tab),
            ("2. Pandas", self._populate_pandas_tab),
            ("3. Calculus", self._populate_calculus_tab),
            ("4. Stats & Graphs", self._populate_stats_tab),
        ]
        for title, builder in tabs:
            frame = tk.Frame(self.notebook, bg=BG, padx=8, pady=8)
            self.notebook.add(frame, text=title)
            builder(frame)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # ------------------------------------------------- display helpers (state)
    def _show_message(self, text):
        self.input_var.set(text)
        self._display_state = "message"

    def _show_result(self, text):
        self.input_var.set(text)
        self._display_state = "result"

    def _report_error(self, label, exc):
        text = str(exc) or "Invalid Input"
        self._show_message(text if isinstance(exc, InvalidInput) else f"{label}: {text}")

    # ============================================================ EXPRESSIONS
    def _build_env(self, angle_aware):
        degrees = angle_aware and self.current_angle_unit.get() == "DEG"
        to_rad = math.radians if degrees else (lambda v: v)
        to_ang = math.degrees if degrees else (lambda v: v)

        def tidy(v):                       # sin(pi) -> 0 instead of 1.2e-16
            return 0.0 if abs(v) < 1e-12 else v

        return {
            "sin": lambda v: tidy(math.sin(to_rad(v))),
            "cos": lambda v: tidy(math.cos(to_rad(v))),
            "tan": lambda v: tidy(math.tan(to_rad(v))),
            "asin": lambda v: to_ang(math.asin(v)),
            "acos": lambda v: to_ang(math.acos(v)),
            "atan": lambda v: to_ang(math.atan(v)),
            "sqrt": math.sqrt, "log": math.log, "ln": math.log,
            "log10": math.log10, "exp": math.exp, "abs": abs,
            "pi": math.pi, "e": math.e,
        }

    def _compile_expr(self, expr, variables=()):
        clean = sanitize_expression(expr, variables)
        try:
            return compile(clean, "<calculator>", "eval")
        except SyntaxError:
            raise InvalidInput()

    def _run_code(self, code, variables=None, angle_aware=False):
        env = self._build_env(angle_aware)
        if variables:
            env.update(variables)
        try:
            res = eval(code, {"__builtins__": {}}, env)
        except (TypeError, NameError, SyntaxError):
            raise InvalidInput()
        if isinstance(res, bool) or not isinstance(res, (int, float)):
            raise InvalidInput()
        if not math.isfinite(res):
            raise OverflowError("Overflow")
        return res

    def _run_expression(self, expr, angle_aware=False):
        return self._run_code(self._compile_expr(expr), None, angle_aware)

    def _make_func(self, expr):
        """Turn text such as 'x**3 - 4*x + 2' into a safe Python function f(x)."""
        code = self._compile_expr(expr, ("x",))
        return lambda x: self._run_code(code, {"x": float(x)}, False)

    def _current_number(self):
        text = self.input_var.get().strip()
        value = _try_float(text)
        return value if value is not None else self._run_expression(text, True)

    def _entry_number(self, entry, name):
        try:
            return float(self._run_expression(entry.get()))
        except (InvalidInput, ArithmeticError, ValueError):
            raise InvalidInput(f"Invalid Input: {name} must be a number")

    def _read_args(self, count, hint):
        parts = [p.strip() for p in self.input_var.get().split(",")]
        if len(parts) != count or any(p == "" for p in parts):
            raise InvalidInput(f"Invalid Input. Format: {hint}")
        try:
            return [float(self._run_expression(p)) for p in parts]
        except (InvalidInput, ArithmeticError, ValueError):
            raise InvalidInput(f"Invalid Input. Format: {hint}")

    @staticmethod
    def _whole(value, name):
        if not float(value).is_integer():
            raise InvalidInput(f"Invalid Input: {name} must be a whole number")
        return int(value)

    # --------------------------------------------------------------- evaluate
    def evaluate_expression(self):
        expr = self.input_var.get().strip()
        self.formula_var.set(expr + " =")
        try:
            open_p, close_p = expr.count("("), expr.count(")")
            if open_p > close_p:
                expr += ")" * (open_p - close_p)

            res = self._run_expression(expr, angle_aware=True)
            formatted = f"{res:.10g}"
            if formatted == "-0":
                formatted = "0"

            self._show_result(formatted)
            self._record_tape(f"{expr} = {formatted}")
        except ZeroDivisionError:
            self._show_message("Error: Div by 0")
            self._record_tape(f"{expr} = Error: Div by 0")
        except OverflowError:
            self._show_message("Error: Overflow")
            self._record_tape(f"{expr} = Error: Overflow")
        except InvalidInput:
            self._show_message("Invalid Input")
            self._record_tape(f"{expr} = Invalid Input")
        except ValueError:                       # e.g. sqrt(-1), log(0)
            self._show_message("Math Error")
            self._record_tape(f"{expr} = Math Error")
        except Exception:
            self._show_message("Invalid Input")
            self._record_tape(f"{expr} = Invalid Input")

    # ============================================================== KEYPAD
    def _append_raw(self, val):
        state = self._display_state
        continues = val[0] in "+-*/%"
        if state == "message" or (state == "result" and not continues):
            self.input_var.set("0")
        self._display_state = "typing"

        current = self.input_var.get()
        if current == "0" and val not in [".", ",", "+", "-", "*", "/", "%", "**"]:
            self.input_var.set(val)
        else:
            self.input_var.set(current + val)
        self.entry_display.icursor(tk.END)

    def _insert_func(self, func_name):
        if self._display_state in ("message", "result"):
            self.input_var.set("0")
        self._display_state = "typing"
        current = self.input_var.get()
        prefix = f"{func_name}("
        self.input_var.set(prefix if current == "0" else current + prefix)
        self.entry_display.icursor(tk.END)

    def backspace(self):
        if self._display_state == "message":
            self.clear_all()
            return
        current = self.input_var.get()
        self.input_var.set(current[:-1] if len(current) > 1 else "0")
        self._display_state = "typing"

    def clear_all(self):
        self.input_var.set("0")
        self.formula_var.set("")
        self._display_state = "typing"

    def _populate_scientific_tab(self, parent):
        for r in range(7):
            parent.rowconfigure(r, weight=1)
        for c in range(6):
            parent.columnconfigure(c, weight=1)

        keys = [
            ("sin", 0, 0, lambda: self._insert_func("sin")),
            ("cos", 0, 1, lambda: self._insert_func("cos")),
            ("tan", 0, 2, lambda: self._insert_func("tan")),
            ("ln", 0, 3, lambda: self._insert_func("log")),
            ("log₁₀", 0, 4, lambda: self._insert_func("log10")),
            ("AC", 0, 5, self.clear_all),

            ("sin⁻¹", 1, 0, lambda: self._insert_func("asin")),
            ("cos⁻¹", 1, 1, lambda: self._insert_func("acos")),
            ("tan⁻¹", 1, 2, lambda: self._insert_func("atan")),
            ("√x", 1, 3, lambda: self._insert_func("sqrt")),
            ("x²", 1, 4, lambda: self._append_raw("**2")),
            ("⌫", 1, 5, self.backspace),

            ("xʸ", 2, 0, lambda: self._append_raw("**")),
            ("eˣ", 2, 1, lambda: self._insert_func("exp")),
            ("π", 2, 2, lambda: self._append_raw(str(math.pi))),
            ("e", 2, 3, lambda: self._append_raw(str(math.e))),
            ("(", 2, 4, lambda: self._append_raw("(")),
            (")", 2, 5, lambda: self._append_raw(")")),

            ("7", 3, 0, lambda: self._append_raw("7")),
            ("8", 3, 1, lambda: self._append_raw("8")),
            ("9", 3, 2, lambda: self._append_raw("9")),
            ("÷", 3, 3, lambda: self._append_raw("/")),
            ("|x|", 3, 4, lambda: self._insert_func("abs")),
            ("1/x", 3, 5, lambda: self._append_raw("**(-1)")),

            ("4", 4, 0, lambda: self._append_raw("4")),
            ("5", 4, 1, lambda: self._append_raw("5")),
            ("6", 4, 2, lambda: self._append_raw("6")),
            ("×", 4, 3, lambda: self._append_raw("*")),
            ("%", 4, 4, lambda: self._append_raw("%")),      # spans 2 columns

            ("1", 5, 0, lambda: self._append_raw("1")),
            ("2", 5, 1, lambda: self._append_raw("2")),
            ("3", 5, 2, lambda: self._append_raw("3")),
            ("-", 5, 3, lambda: self._append_raw("-")),
            ("+", 5, 4, lambda: self._append_raw("+")),
            ("=", 5, 5, self.evaluate_expression),

            ("0", 6, 0, lambda: self._append_raw("0")),
            ("00", 6, 1, lambda: self._append_raw("00")),
            (".", 6, 2, lambda: self._append_raw(".")),
            (",", 6, 3, lambda: self._append_raw(",")),
        ]

        functions = ["sin", "cos", "tan", "ln", "log₁₀", "sin⁻¹", "cos⁻¹", "tan⁻¹",
                     "√x", "x²", "xʸ", "eˣ", "|x|", "1/x"]
        for text, r, c, cmd in keys:
            if text == "=":
                bg_col, fg_col = "#238636", "#FFFFFF"
            elif text in ["AC", "⌫"]:
                bg_col, fg_col = "#DA3633", "#FFFFFF"
            elif text in ["+", "-", "×", "÷", "%"]:
                bg_col, fg_col = PANEL, AMBER
            elif text in functions:
                bg_col, fg_col = CARD, BLUE
            else:
                bg_col, fg_col = KEY, TEXT

            tk.Button(parent, text=text,
                      font=("JetBrains Mono", 10, "bold" if text in ["=", "AC"] else "normal"),
                      bg=bg_col, fg=fg_col, activebackground=BORDER,
                      activeforeground="#FFFFFF", bd=0, cursor="hand2",
                      command=cmd).grid(row=r, column=c, padx=2, pady=2, sticky="nsew",
                                        columnspan={"%": 2, ",": 3}.get(text, 1))

    # ============================================================== MEMORY
    def _update_badge(self):
        self.mem_badge.config(text=f"MEM: {self.memory_register:g}")

    def _mem_change(self, sign):
        try:
            self.memory_register += sign * self._current_number()
            self._update_badge()
        except Exception:
            self._show_message("Invalid Input")

    def _mem_add(self):
        self._mem_change(+1)

    def _mem_sub(self):
        self._mem_change(-1)

    def _mem_recall(self):
        self._show_result(f"{self.memory_register:g}")

    def _mem_clear(self):
        self.memory_register = 0.0
        self._update_badge()

    # ================================================== TAB 2: PANDAS ANALYSIS
    def _populate_pandas_tab(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", pady=(0, 6))

        tk.Label(top, text="DataFrame Input (paste CSV, or numbers separated by spaces):",
                 font=("Segoe UI", 9, "bold"), fg=MUTED, bg=BG).pack(anchor="w")

        self.df_input = tk.Text(top, height=5, font=("JetBrains Mono", 10), bg=CARD,
                                fg=TEXT, bd=0, insertbackground=BLUE)
        sample_csv = "age,salary,experience\n25,50000,1\n30,65000,3\n35,85000,7\n40,110000,12\n48,135000,18"
        self.df_input.insert("1.0", sample_csv)
        self.df_input.pack(fill="x", pady=4)

        presets = tk.Frame(top, bg=BG)
        presets.pack(fill="x", pady=2)
        for text, fg_col, cmd in [
            ("Load CSV...", GREEN, self._load_csv_file),
            ("Clear Data", RED, lambda: self.df_input.delete("1.0", tk.END)),
        ]:
            tk.Button(presets, text=text, bg=PANEL, fg=fg_col, bd=0, padx=6, pady=2,
                      cursor="hand2", command=cmd).pack(side="left", padx=2)

        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="both", expand=True, pady=4)

        actions = [
            ("df.describe() (Summary)", self._pd_describe),
            ("df.corr() (Correlation)", self._pd_corr),
            ("df.cov() (Covariance)", self._pd_cov),
            ("Mean & Median", self._pd_mean_median),
            ("Std Deviation & Variance", self._pd_std_var),
            ("Skewness & Kurtosis", self._pd_skew_kurt),
        ]
        for i, (name, fn) in enumerate(actions):
            tk.Button(grid, text=name, font=("Segoe UI", 9, "bold"), bg=CARD,
                      fg="#38BDF8",
                      activebackground=PANEL, activeforeground="#FFFFFF", bd=0,
                      cursor="hand2", command=fn).grid(row=i // 2, column=i % 2, padx=4,
                                                       pady=3, sticky="nsew")
        for r in range(3):
            grid.rowconfigure(r, weight=1)
        for c in range(2):
            grid.columnconfigure(c, weight=1)

    def _set_df_text(self, text):
        self.df_input.delete("1.0", tk.END)
        self.df_input.insert("1.0", text)

    def _get_dataframe(self):
        import io
        raw = self.df_input.get("1.0", tk.END).strip()
        if not raw:
            raise InvalidInput("Invalid Input: the data box is empty")
        if not HAS_PANDAS:
            raise RuntimeError("Pandas is not installed (pip install pandas)")
        try:
            df = pd.read_csv(io.StringIO(raw))
            if all(_try_float(str(c)) is not None for c in df.columns):
                df = pd.read_csv(io.StringIO(raw), header=None)   # no header row
            if df.empty:
                raise ValueError("empty")
            return df
        except Exception:
            try:
                lines = [ln.split() for ln in raw.split("\n") if ln.strip()]
                return pd.DataFrame(lines, dtype=float)
            except Exception:
                raise InvalidInput("Invalid Input: could not read the data as numbers")

    def _numeric_df(self):
        num = self._get_dataframe().select_dtypes(include="number")
        if num.empty:
            raise InvalidInput("Invalid Input: no numeric columns found")
        return num

    def _log_block(self, title, text):
        self._record_tape(f"=== {title} ===")
        self._record_tape(text)

    def _pd_describe(self):
        try:
            self._log_block("df.describe()", self._numeric_df().describe().round(4).to_string())
            self._show_message("Summary logged to History ↗")
        except Exception as e:
            self._report_error("Pandas Error", e)

    def _pd_corr(self):
        try:
            self._log_block("df.corr()", self._numeric_df().corr().round(4).to_string())
            self._show_message("Correlation matrix in History ↗")
        except Exception as e:
            self._report_error("Corr Error", e)

    def _pd_cov(self):
        try:
            self._log_block("df.cov()", self._numeric_df().cov().round(4).to_string())
            self._show_message("Covariance matrix in History ↗")
        except Exception as e:
            self._report_error("Cov Error", e)

    def _pd_mean_median(self):
        try:
            num = self._numeric_df()
            self._record_tape(f"df.mean(): {num.mean().round(3).to_dict()}")
            self._record_tape(f"df.median(): {num.median().round(3).to_dict()}")
            self._show_message("Mean & Median logged ↗")
        except Exception as e:
            self._report_error("Error", e)

    def _pd_std_var(self):
        try:
            num = self._numeric_df()
            self._record_tape(f"df.std(): {num.std().round(3).to_dict()}")
            self._record_tape(f"df.var(): {num.var().round(3).to_dict()}")
            self._show_message("Std & Variance logged ↗")
        except Exception as e:
            self._report_error("Error", e)

    def _pd_skew_kurt(self):
        try:
            num = self._numeric_df()
            self._record_tape(f"df.skew(): {num.skew().round(3).to_dict()}")
            self._record_tape(f"df.kurt(): {num.kurt().round(3).to_dict()}")
            self._show_message("Skew & Kurtosis logged ↗")
        except Exception as e:
            self._report_error("Error", e)

    def _load_csv_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV / text files", "*.csv *.txt")])
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._set_df_text(f.read())
            except Exception as e:
                messagebox.showerror("Could not open file", str(e))

    # =================================================== TAB 3: CALCULUS & SCIPY
    def _populate_calculus_tab(self, parent):
        tk.Label(parent, text="Calculus Engine (write the function in terms of x, e.g. x**3 - 4*x + 2)",
                 font=("Segoe UI", 9, "bold"), fg=MUTED, bg=BG).pack(anchor="w")

        self.calc_expr_entry = tk.Entry(parent, font=("JetBrains Mono", 11), bg=CARD, fg=TEXT, bd=0)
        self.calc_expr_entry.insert(0, "x**3 - 4*x + 2")
        self.calc_expr_entry.pack(fill="x", pady=4, ipady=3)

        ctrls = tk.Frame(parent, bg=BG)
        ctrls.pack(fill="x", pady=2)

        tk.Label(ctrls, text="Point x₀:", fg=MUTED, bg=BG).pack(side="left")
        self.calc_x0 = tk.Entry(ctrls, width=8, font=("JetBrains Mono", 9), bg=CARD, fg=TEXT, bd=0)
        self.calc_x0.insert(0, "2.0")
        self.calc_x0.pack(side="left", padx=4)

        tk.Label(ctrls, text="Range [a, b]:", fg=MUTED, bg=BG).pack(side="left", padx=(10, 0))
        self.calc_a = tk.Entry(ctrls, width=6, font=("JetBrains Mono", 9), bg=CARD, fg=TEXT, bd=0)
        self.calc_a.insert(0, "0")
        self.calc_a.pack(side="left", padx=4)
        self.calc_b = tk.Entry(ctrls, width=6, font=("JetBrains Mono", 9), bg=CARD, fg=TEXT, bd=0)
        self.calc_b.insert(0, "3")
        self.calc_b.pack(side="left", padx=4)

        box = tk.Frame(parent, bg=BG)
        box.pack(fill="both", expand=True, pady=6)

        actions = [
            ("Integral ∫ f(x) dx over [a, b]", self._scipy_integral),
            ("1st Derivative f'(x₀)", self._calc_derivative),
            ("2nd Derivative f''(x₀)", self._calc_2nd_derivative),
            ("Root Finder (f(x) = 0 in [a, b])", self._scipy_root),
            ("n! (type n in the main display)", self._calc_factorial_ui),
            ("nCr (type 'n, r' in the display)", self._calc_ncr_ui),
            ("nPr (type 'n, r' in the display)", self._calc_npr_ui),
        ]
        for i, (name, fn) in enumerate(actions):
            tk.Button(box, text=name, font=("Segoe UI", 9, "bold"), bg=CARD, fg="#FBBF24",
                      activebackground=PANEL, activeforeground="#FFFFFF", bd=0,
                      cursor="hand2", command=fn).grid(row=i // 2, column=i % 2, padx=3,
                                                       pady=3, sticky="nsew")
        for r in range(4):
            box.rowconfigure(r, weight=1)
        for c in range(2):
            box.columnconfigure(c, weight=1)

    def _scipy_integral(self):
        try:
            f = self._make_func(self.calc_expr_entry.get())
            a = self._entry_number(self.calc_a, "a")
            b = self._entry_number(self.calc_b, "b")
            if HAS_SCIPY:
                res, err = quad(f, a, b)
                out = f"∫[{a:g}, {b:g}] = {res:.6g} (err {err:.1e})"
            else:                                         # Simpson's rule fallback
                n = 200
                h = (b - a) / n
                s = f(a) + f(b)
                for i in range(1, n):
                    s += (4 if i % 2 else 2) * f(a + i * h)
                out = f"Simpson ∫[{a:g}, {b:g}] = {s * h / 3:.6g}"
            self._show_message(out)
            self._record_tape(out)
        except Exception as e:
            self._report_error("Integral Error", e)

    def _calc_derivative(self):
        try:
            f = self._make_func(self.calc_expr_entry.get())
            x0 = self._entry_number(self.calc_x0, "x₀")
            h = 1e-6
            out = f"f'({x0:g}) = {(f(x0 + h) - f(x0 - h)) / (2 * h):.6g}"
            self._show_message(out)
            self._record_tape(out)
        except Exception as e:
            self._report_error("Deriv Error", e)

    def _calc_2nd_derivative(self):
        try:
            f = self._make_func(self.calc_expr_entry.get())
            x0 = self._entry_number(self.calc_x0, "x₀")
            h = 1e-4
            out = f"f''({x0:g}) = {(f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h ** 2:.6g}"
            self._show_message(out)
            self._record_tape(out)
        except Exception as e:
            self._report_error("Deriv2 Error", e)

    @staticmethod
    def _find_bracket(f, a, b, steps=300):
        """Scan [a, b] for a place where f changes sign."""
        prev_x, prev_f = a, f(a)
        if prev_f == 0:
            return a, a
        for i in range(1, steps + 1):
            x = a + (b - a) * i / steps
            fx = f(x)
            if fx == 0:
                return x, x
            if prev_f * fx < 0:
                return prev_x, x
            prev_x, prev_f = x, fx
        raise InvalidInput("Root Error: no sign change in [a, b] - try a wider range")

    def _scipy_root(self):
        try:
            f = self._make_func(self.calc_expr_entry.get())
            a = self._entry_number(self.calc_a, "a")
            b = self._entry_number(self.calc_b, "b")
            if a >= b:
                raise InvalidInput("Invalid Input: a must be smaller than b")
            lo, hi = self._find_bracket(f, a, b)
            if lo == hi:
                root = lo
            elif HAS_SCIPY:
                root = root_scalar(f, bracket=[lo, hi], method="brentq").root
            else:                                          # bisection fallback
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if f(lo) * f(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                root = (lo + hi) / 2
            out = f"Root: x = {root:.6g}"
            self._show_message(out)
            self._record_tape(out)
        except Exception as e:
            self._report_error("Root Error", e)

    def _calc_factorial_ui(self):
        try:
            n = self._whole(self._current_number(), "n")
            if not 0 <= n <= 500:
                raise InvalidInput("Invalid Input: n must be between 0 and 500")
            res = math.factorial(n)
            text = str(res) if len(str(res)) <= 20 else f"{float(res):.10g}"
            self._show_result(text)
            self._record_tape(f"{n}! = {text}")
        except Exception as e:
            self._report_error("Factorial Error", e)

    def _read_n_r(self):
        parts = [p.strip() for p in self.input_var.get().split(",")]
        if len(parts) != 2 or "" in parts:
            raise InvalidInput("Invalid Input. Format: 'n, r' (e.g. 10, 3)")
        n, r = self._read_args(2, "'n, r' (e.g. 10, 3)")
        n, r = self._whole(n, "n"), self._whole(r, "r")
        if n < 0 or r < 0:
            raise InvalidInput("Invalid Input: n and r cannot be negative")
        return n, r

    def _calc_ncr_ui(self):
        try:
            n, r = self._read_n_r()
            res = math.comb(n, r)
            self._show_result(str(res))
            self._record_tape(f"C({n},{r}) = {res}")
        except Exception as e:
            self._report_error("nCr Error", e)

    def _calc_npr_ui(self):
        try:
            n, r = self._read_n_r()
            res = math.perm(n, r)
            self._show_result(str(res))
            self._record_tape(f"P({n},{r}) = {res}")
        except Exception as e:
            self._report_error("nPr Error", e)

    # =========================================== TAB 4: STATISTICS & GRAPHS
    def _populate_stats_tab(self, parent):
        self.stats_graph_var = tk.StringVar(value=GRAPH_TYPES[0])
        self.stats_x_var = tk.StringVar()
        self.stats_y_var = tk.StringVar()

        # --- row 1: graph choice + columns + plot button
        row1 = tk.Frame(parent, bg=BG)
        row1.pack(fill="x")
        tk.Label(row1, text="Graph:", fg=MUTED, bg=BG).pack(side="left")
        ttk.Combobox(row1, textvariable=self.stats_graph_var, values=GRAPH_TYPES,
                     state="readonly", width=26).pack(side="left", padx=(4, 10))
        tk.Label(row1, text="X:", fg=MUTED, bg=BG).pack(side="left")
        self.stats_x_box = ttk.Combobox(row1, textvariable=self.stats_x_var, state="readonly", width=9)
        self.stats_x_box.pack(side="left", padx=(4, 8))
        tk.Label(row1, text="Y:", fg=MUTED, bg=BG).pack(side="left")
        self.stats_y_box = ttk.Combobox(row1, textvariable=self.stats_y_var, state="readonly", width=9)
        self.stats_y_box.pack(side="left", padx=4)
        tk.Button(row1, text="▶ Plot", font=("Segoe UI", 9, "bold"), bg="#238636", fg="#FFFFFF",
                  bd=0, padx=14, pady=2, cursor="hand2",
                  command=self._stats_plot).pack(side="right")

        # --- data box
        self.stats_input = tk.Text(parent, height=4, font=("JetBrains Mono", 10), bg=CARD,
                                   fg=TEXT, bd=0, insertbackground=BLUE)
        self.stats_input.pack(fill="x", pady=(6, 4))
        self.stats_input.bind("<Control-Return>", lambda e: (self._stats_plot(), "break")[1])
        self._set_stats_text(self._sample_study_text())

        # --- row 2: helper buttons
        row2 = tk.Frame(parent, bg=BG)
        row2.pack(fill="x")
        for text, fg_col, cmd in [
            ("Load CSV...", GREEN, self._load_stats_csv),
            ("Clear", RED, lambda: self.stats_input.delete("1.0", tk.END)),
            ("Reset View", AMBER, self._stats_plot),
            ("Save Graph", PURPLE, self._save_graph),
        ]:
            tk.Button(row2, text=text, bg=PANEL, fg=fg_col, bd=0, padx=6, pady=2,
                      cursor="hand2", command=cmd).pack(side="left", padx=2)

        self.stats_status = tk.Label(parent, text="Paste numbers (one column) or CSV with a header, then press Plot.  Ctrl+Enter also plots.  Mouse wheel zooms the graph.",
                                     font=("Segoe UI", 8), fg=MUTED, bg=BG, anchor="w")
        self.stats_status.pack(fill="x", pady=(4, 2))

        # --- plot area
        plot_frame = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        plot_frame.pack(fill="both", expand=True)

        if HAS_MATPLOTLIB:
            self.stats_fig = Figure(figsize=(6.4, 3.2), dpi=100, facecolor=BG)
            self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=plot_frame)
            self.stats_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.stats_canvas.mpl_connect("scroll_event", self._on_stats_scroll)
            try:
                toolbar = NavigationToolbar2Tk(self.stats_canvas, plot_frame, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side="bottom", fill="x")
            except Exception:
                pass
        else:
            tk.Label(plot_frame, text="Graphs need Matplotlib.\nInstall it with:  pip install matplotlib",
                     font=("Segoe UI", 11), fg=AMBER, bg=CARD).pack(expand=True)

        # --- summary table
        self.stats_summary = tk.Text(parent, height=9, font=("JetBrains Mono", 9), bg=CARD,
                                     fg=TEXT, bd=0, state="disabled")
        self.stats_summary.tag_configure("head", foreground=BLUE, font=("JetBrains Mono", 9, "bold"))
        self.stats_summary.pack(fill="x", pady=(6, 0))

    @staticmethod
    def _sample_study_text():
        return ("hours,score\n1,38\n2,45\n2.5,50\n3,55\n4,60\n4.5,64\n"
                "5,70\n6,74\n6.5,79\n7,83\n8,88\n9,93")

    def _set_stats_text(self, text):
        self.stats_input.delete("1.0", tk.END)
        self.stats_input.insert("1.0", text)

    def _load_stats_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV / text files", "*.csv *.txt")])
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._set_stats_text(f.read())
                self._stats_plot()
            except Exception as e:
                messagebox.showerror("Could not open file", str(e))

    def _save_graph(self):
        if not HAS_MATPLOTLIB or self.stats_fig is None:
            messagebox.showinfo("Save Graph", "Install Matplotlib first: pip install matplotlib")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.stats_fig.savefig(path, dpi=200, facecolor=self.stats_fig.get_facecolor())

    def _stats_message(self, text, ok=True):
        self.stats_status.config(text=text, fg=GREEN if ok else RED)

    def _set_summary(self, title, pairs):
        half = (len(pairs) + 1) // 2
        left, right = pairs[:half], pairs[half:]
        box = self.stats_summary
        box.config(state="normal")
        box.delete("1.0", tk.END)
        box.insert(tk.END, f"Summary • {title}\n", "head")
        for i in range(half):
            line = f"{left[i][0]:<22}{fmt_value(left[i][1]):>12}"
            if i < len(right):
                line += f"    {right[i][0]:<22}{fmt_value(right[i][1]):>12}"
            box.insert(tk.END, line + "\n")
        box.config(state="disabled")

    def _stats_plot(self):
        try:
            cols = parse_dataset(self.stats_input.get("1.0", tk.END))
        except InvalidInput as e:
            self._stats_message(str(e), ok=False)
            return

        names = list(cols)
        self.stats_x_box["values"] = names
        self.stats_y_box["values"] = names
        if self.stats_x_var.get() not in names:
            self.stats_x_var.set(names[0])
        if self.stats_y_var.get() not in names:
            self.stats_y_var.set(names[1] if len(names) > 1 else names[0])

        gtype = self.stats_graph_var.get()
        xname, yname = self.stats_x_var.get(), self.stats_y_var.get()
        try:
            title, pairs = self._build_summary(gtype, cols, xname, yname)
            if HAS_MATPLOTLIB:
                self._draw_graph(gtype, cols, xname, yname)
            self._set_summary(title, pairs)
            n = len(cols[xname])
            self._stats_message(f"Plotted '{gtype}' with {n} rows and {len(names)} column(s).")
        except InvalidInput as e:
            self._stats_message(str(e), ok=False)
        except Exception as e:
            self._stats_message(f"Invalid Input: could not draw this graph ({e})", ok=False)

    def _build_summary(self, gtype, cols, xname, yname):
        x, y = cols[xname], cols[yname]
        if gtype == GRAPH_TYPES[3]:                      # scatter + regression
            slope, intercept, r, r2 = linear_regression(x, y)
            n = len(x)
            mx, my = sum(x) / n, sum(y) / n
            cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n - 1)
            return f"{yname} vs {xname}", [
                ("Count (n)", n), ("Mean X", mx), ("Mean Y", my),
                ("Std Dev X (s)", statistics.stdev(x)), ("Std Dev Y (s)", statistics.stdev(y)),
                ("Correlation (r)", r), ("R² (fit quality)", r2),
                ("Slope (m)", slope), ("Intercept (b)", intercept), ("Covariance", cov),
            ]
        if gtype == GRAPH_TYPES[6]:                      # heatmap
            if len(cols) < 2:
                raise InvalidInput("Invalid Input: the heatmap needs at least 2 columns")
            best, best_r = None, 0.0
            names = list(cols)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    r = pearson(cols[names[i]], cols[names[j]])
                    if r is not None and abs(r) >= abs(best_r):
                        best, best_r = f"{names[i]} & {names[j]}", r
            return "all columns", [("Columns", len(names)), ("Rows", len(x)),
                                   ("Strongest pair", best or "n/a"), ("Its correlation (r)", best_r)]
        return xname, describe(x)

    # ------------------------------------------------------------- drawing
    def _style_axes(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=PANEL, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)

    def _legend(self, ax):
        leg = ax.legend(facecolor=CARD, edgecolor=BORDER, fontsize=8)
        for text in leg.get_texts():
            text.set_color(TEXT)

    def _draw_graph(self, gtype, cols, xname, yname):
        fig = self.stats_fig
        fig.clear()
        ax = fig.add_subplot(111)
        x, y = cols[xname], cols[yname]
        n = len(x)

        if gtype == GRAPH_TYPES[0]:                                   # normal curve
            mu, sigma = sum(x) / n, statistics.pstdev(x)
            if sigma == 0:
                raise InvalidInput("Invalid Input: all values are identical - cannot draw a curve")
            dist = statistics.NormalDist(mu, sigma)
            lo, hi = mu - 4 * sigma, mu + 4 * sigma
            xs = [lo + i * (hi - lo) / 400 for i in range(401)]
            ys = [dist.pdf(v) for v in xs]
            self._style_axes(ax, "Normal curve", xname, "Probability density")
            ax.plot(xs, ys, color="#E5534B", linewidth=2)
            ax.fill_between(xs, ys, color="#E5534B", alpha=0.20)
            ax.axvline(mu, color=GREEN, linewidth=1.2, label="mean")
            self._legend(ax)

        elif gtype == GRAPH_TYPES[1]:                                 # histogram + fit
            bins = max(5, min(30, int(round(math.sqrt(n)))))
            self._style_axes(ax, f"Histogram of {xname}", xname, "Density")
            ax.hist(x, bins=bins, density=True, color=BLUE, edgecolor=BG, alpha=0.85)
            mu, sigma = sum(x) / n, statistics.pstdev(x)
            if sigma > 0:
                dist = statistics.NormalDist(mu, sigma)
                lo, hi = min(x) - sigma, max(x) + sigma
                xs = [lo + i * (hi - lo) / 300 for i in range(301)]
                ax.plot(xs, [dist.pdf(v) for v in xs], color="#E5534B", linewidth=2,
                        label="Normal fit")
                self._legend(ax)

        elif gtype == GRAPH_TYPES[2]:                                 # box plot
            names = list(cols)
            self._style_axes(ax, "Box plot (dots = outliers)", "", "Value")
            bp = ax.boxplot([cols[nm] for nm in names], patch_artist=True)
            for box in bp["boxes"]:
                box.set(facecolor=PANEL, edgecolor=BLUE)
            for key in ("whiskers", "caps"):
                for item in bp[key]:
                    item.set(color=MUTED)
            for item in bp["medians"]:
                item.set(color=AMBER, linewidth=2)
            for item in bp["fliers"]:
                item.set(markerfacecolor=RED, markeredgecolor=RED, markersize=4)
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names)

        elif gtype == GRAPH_TYPES[3]:                                 # scatter + regression
            slope, intercept, r, r2 = linear_regression(x, y)
            self._style_axes(ax, f"{yname} vs {xname}", xname, yname)
            ax.scatter(x, y, color=BLUE, s=28, alpha=0.9, label="data")
            x0, x1 = min(x), max(x)
            ax.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
                    color="#E5534B", linewidth=2, label="best fit line")
            self._legend(ax)

        elif gtype == GRAPH_TYPES[4]:                                 # line plot
            self._style_axes(ax, "Line plot (value by row number)", "Row", "Value")
            idx = list(range(1, n + 1))
            ax.plot(idx, x, color=BLUE, marker="o", markersize=4, label=xname)
            if yname != xname:
                ax.plot(idx, y, color=AMBER, marker="o", markersize=4, label=yname)
            self._legend(ax)

        elif gtype == GRAPH_TYPES[5]:                                 # bar chart
            counts = Counter(round(v, 2) for v in x)
            if len(counts) > 25:
                raise InvalidInput("Invalid Input: too many different values for a bar chart - use Histogram")
            keys = sorted(counts)
            self._style_axes(ax, f"Frequency of {xname}", xname, "Count")
            ax.bar([f"{k:g}" for k in keys], [counts[k] for k in keys], color=GREEN, edgecolor=BG)
            ax.tick_params(axis="x", labelrotation=45)

        elif gtype == GRAPH_TYPES[6]:                                 # correlation heatmap
            names = list(cols)
            if len(names) < 2:
                raise InvalidInput("Invalid Input: the heatmap needs at least 2 columns")
            matrix = []
            for a in names:
                row = []
                for b in names:
                    r = pearson(cols[a], cols[b])
                    row.append(float("nan") if r is None else r)
                matrix.append(row)
            self._style_axes(ax, "Correlation heatmap (Pearson r)")
            ax.grid(False)
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names)
            ax.set_yticklabels(names)
            for i in range(len(names)):
                for j in range(len(names)):
                    value = matrix[i][j]
                    ax.text(j, i, "n/a" if value != value else f"{value:.2f}",
                            ha="center", va="center", color="#111111", fontsize=9)
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(colors=MUTED, labelsize=8)
            cbar.outline.set_edgecolor(BORDER)

        try:
            fig.tight_layout()
        except Exception:
            pass
        self.stats_canvas.draw_idle()

    def _on_stats_scroll(self, event):
        """Mouse wheel zoom, centred on the cursor."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata - (event.xdata - x0) * scale, event.xdata + (x1 - event.xdata) * scale)
        ax.set_ylim(event.ydata - (event.ydata - y0) * scale, event.ydata + (y1 - event.ydata) * scale)
        self.stats_canvas.draw_idle()

    # =========================================================== TAPE SIDEBAR
    def _build_tape_sidebar(self, parent):
        tk.Label(parent, text="History", font=("Segoe UI", 9, "bold"),
                 fg=MUTED, bg=CARD).pack(anchor="w", padx=10, pady=(10, 4))

        frame = tk.Frame(parent, bg=CARD)
        frame.pack(fill="both", expand=True, padx=10, pady=4)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.history_listbox = tk.Listbox(frame, font=("JetBrains Mono", 8), bg=BG, fg=MUTED,
                                          selectbackground=PANEL, selectforeground=BLUE, bd=0,
                                          yscrollcommand=scrollbar.set)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        self.history_listbox.bind("<Double-Button-1>", self._on_history_double_click)

        tk.Button(parent, text="Clear All", font=("Segoe UI", 8, "bold"), bg=PANEL, fg=RED,
                  bd=0, cursor="hand2", command=self._clear_tape).pack(fill="x", padx=10, pady=8)

    def _record_tape(self, entry):
        for line in str(entry).split("\n"):
            self.history_listbox.insert(tk.END, line)
        self.history_listbox.see(tk.END)

    def _on_history_double_click(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            item = self.history_listbox.get(selection[0])
            if "=" in item:
                value = item.split("=")[-1].strip()
                if _try_float(value) is not None:
                    self._show_result(value)

    def _clear_tape(self):
        self.history_listbox.delete(0, tk.END)


if __name__ == "__main__":
    app = NexusDataScienceStudio()
    app.mainloop()