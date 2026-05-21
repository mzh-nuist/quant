# -*- coding: utf-8 -*-
"""
多因子选股回测 —— 图形界面
---------------------------
- 股票池管理（内置默认池 / 搜索添加 / 从文件加载 / 直接编辑）
- 五因子权重配置（动量 / 低波 / 量能 / RSI / 规模）
- 每期持股数量、回测区间、初始资金设置
- 一键回测，展示 Sharpe / Sortino / 最大回撤 / 月度胜率等完整指标

启动：
    python -m multifactor.gui
    或
    python multifactor/gui.py
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from multifactor.engine import run_backtest, MultifactorResult, _default_universe
from multifactor.selector import DEFAULT_WEIGHTS

# ============================================================================
#  样式常量（与 dual_ma/gui.py 保持一致）
# ============================================================================
FONT_TITLE = ("Microsoft YaHei", 14, "bold")
FONT_HEADING = ("Microsoft YaHei", 11, "bold")
FONT_BODY = ("Microsoft YaHei", 10)
FONT_SMALL = ("Microsoft YaHei", 9)
FONT_RESULT = ("Consolas", 11, "bold")
FONT_MONO = ("Consolas", 10)

COLOR_BG = "#f5f6fa"
COLOR_CARD = "#ffffff"
COLOR_PRIMARY = "#2d3436"
COLOR_ACCENT = "#0984e3"
COLOR_ACCENT2 = "#6c5ce7"
COLOR_GREEN = "#00b894"
COLOR_RED = "#d63031"
COLOR_TEXT = "#2d3436"
COLOR_SUBTEXT = "#636e72"
COLOR_BORDER = "#ddd"

# 因子定义（与 factors.py 同步）
FACTOR_KEYS = ["momentum", "volatility", "volume_ratio", "rsi", "size"]
FACTOR_LABELS = {
    "momentum":     "动量 (20日)",
    "volatility":   "低波动 (60日)",
    "volume_ratio": "低换手 (5/20)",
    "rsi":          "RSI 反转 (14日)",
    "size":         "小盘规模",
}
FACTOR_DEFAULTS = {
    "momentum":     0.25,
    "volatility":   0.20,
    "volume_ratio": 0.15,
    "rsi":          0.15,
    "size":         0.25,
}

# 结果面板指标定义
RESULT_SECTIONS = [
    ("Capital & Benchmark", [
        ("initial_cash",        "Initial Capital",     False),
        ("final_value",         "Final Value",         False),
        ("total_return",        "Total Return",        True),
        ("benchmark_return",    "Benchmark (EqualWt)", True),
        ("annual_return",       "Annual Return",       True),
        ("annual_volatility",   "Annual Volatility",   False),
    ]),
    ("Risk-Adjusted Ratios", [
        ("sharpe_ratio",        "Sharpe Ratio",        True),
        ("sortino_ratio",       "Sortino Ratio",       True),
        ("calmar_ratio",        "Calmar Ratio",        True),
        ("max_drawdown",        "Max Drawdown",        True),
        ("max_drawdown_days",   "Max DD Duration",     False),
    ]),
    ("Monthly & Rebalance", [
        ("monthly_win_rate",    "Monthly Win Rate",    True),
        ("best_month",          "Best Month",          True),
        ("worst_month",         "Worst Month",         True),
        ("selected_count",      "Rebalance Count",     False),
    ]),
]


class StockLookup:
    """股票代码查询器（与 dual_ma/gui.py 共用逻辑）。"""

    def __init__(self):
        self._stocks = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            self._stocks = list(zip(df["代码"].astype(str), df["名称"].astype(str)))
        except Exception:
            # 离线回退：使用内置列表
            from multifactor.engine import _default_universe
            codes = _default_universe()
            self._stocks = [(c, c) for c in codes]

    def search(self, query: str, limit: int = 30) -> list:
        self._ensure_loaded()
        q = query.strip().lower()
        if not q:
            return []
        results = []
        for code, name in self._stocks:
            if q in code or q in name.lower():
                results.append((code, name))
                if len(results) >= limit:
                    break
        return results


class MultifactorGUI:
    """多因子选股回测图形界面。"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Multi-Factor Stock Selection — Backtest")
        self.root.geometry("1120x920")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(True, True)
        self.root.minsize(950, 750)

        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        self._lookup = StockLookup()
        self._last_result = None
        self._alive = True

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ------------------------------------------------------------------
    #  UI 布局
    # ------------------------------------------------------------------
    def _build_ui(self):
        # 标题
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=46)
        header.pack(fill=tk.X)
        tk.Label(header, text="Multi-Factor Stock Selection Backtest",
                 font=FONT_TITLE, fg="white", bg=COLOR_PRIMARY, pady=8).pack()

        # 主区域
        main = tk.Frame(self.root, bg=COLOR_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # 左侧：参数面板
        left = tk.Frame(main, bg=COLOR_BG, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_input_panel(left)

        # 右侧：结果面板（可滚动）
        right = tk.Frame(main, bg=COLOR_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self._build_result_panel(right)

        # 状态栏
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Frame(self.root, bg="#dfe6e9", height=24)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_bar, textvariable=self.status_var,
                 font=FONT_SMALL, fg=COLOR_SUBTEXT, bg="#dfe6e9",
                 anchor=tk.W, padx=10).pack(fill=tk.X)

    # ------------------------------------------------------------------
    #  左侧输入面板
    # ------------------------------------------------------------------
    def _build_input_panel(self, parent: tk.Frame):
        # 用 Canvas + Scrollbar 让整个左面板可滚动
        canvas = tk.Canvas(parent, bg=COLOR_BG, highlightthickness=0, width=370)
        scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        card_frame = tk.Frame(canvas, bg=COLOR_BG)

        card_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=card_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ---- 股票池管理 ----
        pool_card = tk.Frame(card_frame, bg=COLOR_CARD,
                           highlightbackground=COLOR_BORDER,
                           highlightthickness=1, padx=10, pady=8)
        pool_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(pool_card, text="Stock Universe", font=FONT_HEADING,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(anchor=tk.W)

        # 搜索添加
        search_row = tk.Frame(pool_card, bg=COLOR_CARD)
        search_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(search_row, text="Search:", font=FONT_SMALL, bg=COLOR_CARD,
                 fg=COLOR_SUBTEXT).pack(side=tk.LEFT)
        self.entry_search = tk.Entry(search_row, font=FONT_BODY, width=12,
                                     relief=tk.SOLID, borderwidth=1)
        self.entry_search.pack(side=tk.LEFT, padx=(4, 2))
        self.entry_search.bind("<KeyRelease>", self._on_search_typing)
        tk.Button(search_row, text="Add", font=FONT_SMALL, bg=COLOR_ACCENT, fg="white",
                  relief=tk.FLAT, cursor="hand2", padx=8,
                  command=self._add_searched_stock).pack(side=tk.LEFT, padx=(2, 0))

        # 搜索结果列表
        sb = tk.Scrollbar(search_row)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.search_listbox = tk.Listbox(
            search_row, font=("Consolas", 9), height=3, yscrollcommand=sb.set,
            relief=tk.SOLID, borderwidth=1)
        self.search_listbox.pack(fill=tk.X, pady=(4, 2))
        self.search_listbox.bind("<Double-Button-1>", self._on_search_select)
        self.search_listbox.bind("<Return>", self._on_search_select)
        sb.config(command=self.search_listbox.yview)

        # 股票池文本编辑区
        self.universe_text = tk.Text(
            pool_card, font=FONT_MONO, height=6, width=34,
            relief=tk.SOLID, borderwidth=1, wrap=tk.WORD)
        self.universe_text.pack(fill=tk.X, pady=(4, 2))
        self.universe_text.insert("1.0", "\n".join(_default_universe()))

        # 股票池按钮行
        btn_row = tk.Frame(pool_card, bg=COLOR_CARD)
        btn_row.pack(fill=tk.X, pady=(2, 0))
        tk.Button(btn_row, text="Load Default", font=FONT_SMALL, bg="#636e72", fg="white",
                  relief=tk.FLAT, cursor="hand2", padx=6,
                  command=self._load_default_universe).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text="Load File...", font=FONT_SMALL, bg="#636e72", fg="white",
                  relief=tk.FLAT, cursor="hand2", padx=6,
                  command=self._load_universe_file).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text="Clear", font=FONT_SMALL, bg="#b2bec3", fg="white",
                  relief=tk.FLAT, cursor="hand2", padx=6,
                  command=self._clear_universe).pack(side=tk.LEFT)

        stock_count_var = tk.StringVar(value=f"{len(_default_universe())} stocks")
        tk.Label(btn_row, textvariable=stock_count_var, font=FONT_SMALL,
                 bg=COLOR_CARD, fg=COLOR_SUBTEXT).pack(side=tk.RIGHT)
        self._stock_count_var = stock_count_var

        # 绑定文本变更更新计数
        self.universe_text.bind("<KeyRelease>", lambda e: self._update_stock_count())

        # ---- 回测参数 ----
        param_card = tk.Frame(card_frame, bg=COLOR_CARD,
                            highlightbackground=COLOR_BORDER,
                            highlightthickness=1, padx=10, pady=8)
        param_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(param_card, text="Backtest Parameters", font=FONT_HEADING,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(anchor=tk.W)

        self._add_field(param_card, "Start Date", "2020-01-01", "start")
        self._add_field(param_card, "End Date", "2025-12-31", "end")
        self._add_field(param_card, "Top N Stocks", "20", "top_n")
        self._add_field(param_card, "Initial Cash", "1000000", "cash")
        self._add_field(param_card, "Commission", "0.001", "commission")

        # ---- 因子权重 ----
        factor_card = tk.Frame(card_frame, bg=COLOR_CARD,
                             highlightbackground=COLOR_BORDER,
                             highlightthickness=1, padx=10, pady=8)
        factor_card.pack(fill=tk.X)

        tk.Label(factor_card, text="Factor Weights", font=FONT_HEADING,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(anchor=tk.W)
        tk.Label(factor_card, text="(自动归一化，总和不必恰好为 1)",
                 font=FONT_SMALL, bg=COLOR_CARD, fg=COLOR_SUBTEXT).pack(anchor=tk.W, pady=(0, 4))

        self.weight_entries = {}
        for key in FACTOR_KEYS:
            row = tk.Frame(factor_card, bg=COLOR_CARD)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=FACTOR_LABELS[key], font=FONT_BODY, bg=COLOR_CARD,
                     fg=COLOR_TEXT, width=16, anchor=tk.W).pack(side=tk.LEFT)
            e = tk.Entry(row, font=FONT_BODY, width=8, justify=tk.CENTER,
                        relief=tk.SOLID, borderwidth=1)
            e.insert(0, str(FACTOR_DEFAULTS[key]))
            e.pack(side=tk.LEFT)
            self.weight_entries[key] = e

        # ---- 运行按钮 ----
        btn_frame = tk.Frame(card_frame, bg=COLOR_BG)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = tk.Button(btn_frame, text="▶  Run Backtest",
                                 font=FONT_HEADING, bg=COLOR_ACCENT, fg="white",
                                 activebackground="#0773c5", activeforeground="white",
                                 relief=tk.FLAT, cursor="hand2", padx=14, pady=8,
                                 command=self._on_run)
        self.run_btn.pack(fill=tk.X)

        self.progress_var = tk.StringVar(value="")
        tk.Label(btn_frame, textvariable=self.progress_var, font=FONT_SMALL,
                 fg=COLOR_SUBTEXT, bg=COLOR_BG).pack(pady=(4, 0))

        # 强制刷新
        self.force_refresh_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_frame, text="Force Refresh (skip cache)",
                       variable=self.force_refresh_var,
                       font=FONT_SMALL, bg=COLOR_BG, fg=COLOR_SUBTEXT,
                       activebackground=COLOR_BG, selectcolor=COLOR_CARD,
                       ).pack(pady=(6, 0))

        # 代理设置
        px_frame = tk.Frame(card_frame, bg=COLOR_BG)
        px_frame.pack(fill=tk.X, pady=(6, 0))
        tk.Label(px_frame, text="Proxy:", font=FONT_SMALL, bg=COLOR_BG,
                 fg=COLOR_TEXT).pack(side=tk.LEFT, padx=(0, 6))
        self.entry_proxy = tk.Entry(px_frame, font=FONT_SMALL, width=26,
                                    relief=tk.SOLID, borderwidth=1)
        self.entry_proxy.insert(0, "http://127.0.0.1:10808")
        self.entry_proxy.pack(side=tk.LEFT)

    def _add_field(self, parent, label, default, attr):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_TEXT, width=14, anchor=tk.W).pack(side=tk.LEFT)
        e = tk.Entry(row, font=FONT_BODY, width=18, justify=tk.CENTER,
                     relief=tk.SOLID, borderwidth=1)
        e.insert(0, default)
        e.pack(side=tk.LEFT)
        setattr(self, f"entry_{attr}", e)

    # ------------------------------------------------------------------
    #  股票池操作
    # ------------------------------------------------------------------
    def _get_universe(self) -> list[str]:
        """从文本框中解析股票池。"""
        text = self.universe_text.get("1.0", tk.END).strip()
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _update_stock_count(self):
        n = len(self._get_universe())
        self._stock_count_var.set(f"{n} stocks")

    def _load_default_universe(self):
        self.universe_text.delete("1.0", tk.END)
        self.universe_text.insert("1.0", "\n".join(_default_universe()))
        self._update_stock_count()

    def _load_universe_file(self):
        path = filedialog.askopenfilename(
            title="选择股票池文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                codes = [line.strip() for line in f if line.strip()]
            self.universe_text.delete("1.0", tk.END)
            self.universe_text.insert("1.0", "\n".join(codes))
            self._update_stock_count()
            self.status_var.set(f"Loaded {len(codes)} stocks from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("File Error", str(e))

    def _clear_universe(self):
        self.universe_text.delete("1.0", tk.END)
        self._update_stock_count()

    # ------------------------------------------------------------------
    #  股票搜索
    # ------------------------------------------------------------------
    def _on_search_typing(self, event=None):
        if hasattr(self, "_search_timer") and self._search_timer:
            self.root.after_cancel(self._search_timer)
        self._search_timer = self.root.after(250, self._do_search)

    def _do_search(self):
        q = self.entry_search.get().strip()
        self.search_listbox.delete(0, tk.END)
        if not q:
            return
        for code, name in self._lookup.search(q):
            self.search_listbox.insert(tk.END, f"{code}  {name}")

    def _on_search_select(self, event=None):
        """双击搜索结果 → 添加到股票池。"""
        sel = self.search_listbox.curselection()
        if not sel:
            return
        text = self.search_listbox.get(sel[0])
        code = text.split()[0] if text else ""
        self._add_to_universe(code)

    def _add_searched_stock(self):
        """Add 按钮：添加搜索框中输入的代码。"""
        q = self.entry_search.get().strip()
        if not q:
            return
        # 尝试匹配搜索结果第一条
        results = self._lookup.search(q, limit=1)
        if results:
            self._add_to_universe(results[0][0])
        else:
            # 直接当代码添加
            self._add_to_universe(q)

    def _add_to_universe(self, code: str):
        """添加一只股票到股票池（去重）。"""
        current = self._get_universe()
        if code in current:
            self.status_var.set(f"{code} already in universe.")
            return
        text = self.universe_text.get("1.0", tk.END).strip()
        if text:
            self.universe_text.insert(tk.END, f"\n{code}")
        else:
            self.universe_text.insert("1.0", code)
        self._update_stock_count()
        self.status_var.set(f"Added: {code}")

    # ------------------------------------------------------------------
    #  右侧结果面板（可滚动）
    # ------------------------------------------------------------------
    def _build_result_panel(self, parent: tk.Frame):
        canvas = tk.Canvas(parent, bg=COLOR_CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.result_frame = tk.Frame(canvas, bg=COLOR_CARD)

        self.result_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.result_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        tk.Label(self.result_frame, text="Results", font=FONT_HEADING,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(anchor=tk.W, padx=16, pady=(10, 6))
        self.result_labels = {}

        for section_title, rows in RESULT_SECTIONS:
            tk.Label(self.result_frame, text=section_title,
                     font=("Microsoft YaHei", 10, "bold"),
                     bg=COLOR_CARD, fg=COLOR_ACCENT,
                     ).pack(anchor=tk.W, padx=16, pady=(8, 2))
            for key, label, _ in rows:
                self._add_result_row(self.result_frame, label, key)

        self.placeholder = tk.Label(
            self.result_frame,
            text="Configure parameters and click 'Run Backtest'.",
            font=FONT_BODY, fg="#b2bec3", bg=COLOR_CARD, justify=tk.CENTER)
        self.placeholder.pack(expand=True, pady=40)

    def _add_result_row(self, parent, label, key):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill=tk.X, padx=16, pady=1)
        tk.Label(row, text=label, font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_SUBTEXT, width=24, anchor=tk.W).pack(side=tk.LEFT)
        vl = tk.Label(row, text="—", font=FONT_RESULT, bg=COLOR_CARD,
                      fg=COLOR_TEXT, anchor=tk.E, width=18)
        vl.pack(side=tk.RIGHT)
        self.result_labels[key] = vl

    # ------------------------------------------------------------------
    #  安全更新
    # ------------------------------------------------------------------
    def _safe_update(self, func, *args, **kwargs):
        if not self._alive:
            return
        try:
            func(*args, **kwargs)
        except tk.TclError:
            pass

    def _on_close(self):
        self._alive = False
        self.root.destroy()

    # ------------------------------------------------------------------
    #  参数读取
    # ------------------------------------------------------------------
    def _read_params(self):
        """读取所有输入参数并返回 dict，校验失败抛出 ValueError。"""
        universe = self._get_universe()
        if len(universe) < 3:
            raise ValueError("Stock universe must contain at least 3 stocks.")

        start_date = self.entry_start.get().strip().replace("-", "")
        end_date = self.entry_end.get().strip().replace("-", "")
        top_n = int(self.entry_top_n.get().strip())
        cash = float(self.entry_cash.get().strip())
        commission = float(self.entry_commission.get().strip())

        if top_n < 1:
            raise ValueError("Top N must be at least 1.")
        if top_n >= len(universe):
            raise ValueError(f"Top N ({top_n}) must be less than universe size ({len(universe)}).")

        # 读取因子权重并自动归一化
        raw = {}
        for key in FACTOR_KEYS:
            raw[key] = float(self.weight_entries[key].get().strip())
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Factor weights sum must be positive.")
        weights = {k: v / total for k, v in raw.items()}

        proxy = self.entry_proxy.get().strip() or None

        return {
            "universe": universe,
            "start_date": start_date,
            "end_date": end_date,
            "top_n": top_n,
            "cash": cash,
            "commission": commission,
            "weights": weights,
            "force_refresh": self.force_refresh_var.get(),
            "proxy": proxy,
        }

    # ------------------------------------------------------------------
    #  运行回测
    # ------------------------------------------------------------------
    def _on_run(self):
        try:
            p = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self._safe_update(self.progress_var.set, "Fetching data & running backtest...")
        self._safe_update(self.status_var.set, "Running multi-factor backtest...")
        try:
            self.placeholder.pack_forget()
        except (tk.TclError, AttributeError):
            pass

        def worker():
            try:
                result = run_backtest(
                    universe=p["universe"],
                    start_date=p["start_date"],
                    end_date=p["end_date"],
                    top_n=p["top_n"],
                    cash=p["cash"],
                    commission=p["commission"],
                    weights=p["weights"],
                    force_refresh=p.get("force_refresh", False),
                    proxy=p.get("proxy"),
                    verbose=False,
                )
                self._last_result = result
                if self._alive:
                    self.root.after(0, lambda: self._on_backtest_done(p))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if self._alive:
                    self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    #  回测完成处理
    # ------------------------------------------------------------------
    def _on_backtest_done(self, params: dict):
        if not self._alive or self._last_result is None:
            return

        self._display_result(self._last_result)

        self._safe_update(self.run_btn.config, state=tk.NORMAL, text="▶  Run Backtest")
        self._safe_update(self.progress_var.set, "")
        top_n = params.get("top_n", "?")
        self._safe_update(self.status_var.set,
                          f"Backtest done. Top {top_n}, "
                          f"Final: {self._last_result.final_value:,.0f}")

    def _display_result(self, result: MultifactorResult):
        green, red = COLOR_GREEN, COLOR_RED

        def fmt(val, suffix="", decimals=2, sign_color=True):
            text = f"{val:,.{decimals}f}{suffix}"
            if sign_color and isinstance(val, (int, float)):
                c = green if val > 0 else (red if val < 0 else COLOR_TEXT)
            else:
                c = COLOR_TEXT
            return text, c

        updates = {
            "initial_cash":         fmt(result.initial_cash, sign_color=False),
            "final_value":          fmt(result.final_value, sign_color=False),
            "total_return":         fmt(result.total_return, suffix="%"),
            "benchmark_return":     fmt(result.benchmark_return, suffix="%"),
            "annual_return":        fmt(result.annual_return, suffix="%"),
            "annual_volatility":    fmt(result.annual_volatility, suffix="%", sign_color=False),
            "sharpe_ratio":         fmt(result.sharpe_ratio),
            "sortino_ratio":        fmt(result.sortino_ratio),
            "calmar_ratio":         fmt(result.calmar_ratio),
            "max_drawdown":         fmt(result.max_drawdown, suffix="%", sign_color=False),
            "max_drawdown_days":    (f"{result.max_drawdown_days}", COLOR_TEXT),
            "monthly_win_rate":     fmt(result.monthly_win_rate, suffix="%"),
            "best_month":           fmt(result.best_month, suffix="%"),
            "worst_month":          fmt(result.worst_month, suffix="%"),
            "selected_count":       (f"{result.selected_count}", COLOR_TEXT),
        }

        for key, (text, color) in updates.items():
            if key in self.result_labels:
                self.result_labels[key].config(text=text, fg=color)

    def _on_error(self, msg: str):
        self._safe_update(self.run_btn.config, state=tk.NORMAL, text="▶  Run Backtest")
        self._safe_update(self.progress_var.set, "")
        self._safe_update(self.status_var.set, "Error occurred.")
        messagebox.showerror("Backtest Error", msg)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MultifactorGUI()
    app.run()
