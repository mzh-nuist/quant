# -*- coding: utf-8 -*-
"""
SMA 双均线交叉回测 —— 图形界面
-------------------------------
- 公司名称模糊搜索 → 股票代码
- 固定/移动止损参数配置
- 一键回测，展示 Sharpe / Sortino / Buy&Hold 等完整指标
- 参数网格搜索优化（新窗口展示结果表格）
- 可选显示 / 保存图表

启动：
    python gui.py
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from dual_ma.engine import run_backtest, run_optimization, BacktestResult, OptResult

# ============================================================================
#  内置热门 A 股列表
# ============================================================================
FALLBACK_STOCKS = [
    ("600519", "贵州茅台"), ("000001", "平安银行"), ("000002", "万科A"),
    ("000858", "五粮液"), ("002594", "比亚迪"), ("300750", "宁德时代"),
    ("601318", "中国平安"), ("600036", "招商银行"), ("000333", "美的集团"),
    ("600276", "恒瑞医药"), ("601888", "中国中免"), ("002415", "海康威视"),
    ("600900", "长江电力"), ("600030", "中信证券"), ("000651", "格力电器"),
    ("601398", "工商银行"), ("601857", "中国石油"), ("600028", "中国石化"),
    ("002475", "立讯精密"), ("300059", "东方财富"), ("688981", "中芯国际"),
    ("600809", "山西汾酒"), ("000568", "泸州老窖"), ("002714", "牧原股份"),
    ("601012", "隆基绿能"), ("300274", "阳光电源"), ("600585", "海螺水泥"),
    ("000725", "京东方A"), ("002230", "科大讯飞"), ("300124", "汇川技术"),
    ("601166", "兴业银行"), ("600000", "浦发银行"), ("601899", "紫金矿业"),
    ("600050", "中国联通"), ("601728", "中国电信"), ("600941", "中国移动"),
    ("002352", "顺丰控股"), ("000338", "潍柴动力"), ("600031", "三一重工"),
    ("002304", "洋河股份"), ("000625", "长安汽车"), ("601238", "广汽集团"),
    ("600104", "上汽集团"), ("002466", "天齐锂业"), ("002460", "赣锋锂业"),
    ("601615", "明阳智能"), ("300014", "亿纬锂能"), ("002129", "中环股份"),
    ("600438", "通威股份"), ("600690", "海尔智家"), ("000100", "TCL科技"),
    ("002371", "北方华创"), ("603259", "药明康德"), ("300015", "爱尔眼科"),
    ("300122", "智飞生物"), ("000661", "长春高新"), ("600196", "复星医药"),
    ("600887", "伊利股份"), ("002142", "宁波银行"), ("600048", "保利发展"),
    ("001979", "招商蛇口"), ("601668", "中国建筑"), ("600309", "万华化学"),
    ("002049", "紫光国微"), ("603986", "兆易创新"), ("300782", "卓胜微"),
    ("688111", "金山办公"), ("002236", "大华股份"), ("000063", "中兴通讯"),
    ("600570", "恒生电子"), ("300033", "同花顺"), ("002410", "广联达"),
    ("600588", "用友网络"), ("000977", "浪潮信息"), ("002841", "视源股份"),
    ("600845", "宝信软件"), ("300454", "深信服"), ("688012", "中微公司"),
    ("600406", "国电南瑞"), ("601088", "中国神华"), ("600019", "宝钢股份"),
    ("000786", "北新建材"), ("002271", "东方雨虹"), ("600176", "中国巨石"),
    ("601939", "建设银行"), ("601288", "农业银行"), ("600016", "民生银行"),
]

# ============================================================================
#  样式常量
# ============================================================================
FONT_TITLE = ("Microsoft YaHei", 14, "bold")
FONT_HEADING = ("Microsoft YaHei", 11, "bold")
FONT_BODY = ("Microsoft YaHei", 10)
FONT_SMALL = ("Microsoft YaHei", 9)
FONT_RESULT = ("Consolas", 11, "bold")
FONT_TABLE = ("Consolas", 9)

COLOR_BG = "#f5f6fa"
COLOR_CARD = "#ffffff"
COLOR_PRIMARY = "#2d3436"
COLOR_ACCENT = "#0984e3"
COLOR_ACCENT2 = "#6c5ce7"  # 优化按钮颜色
COLOR_GREEN = "#00b894"
COLOR_RED = "#d63031"
COLOR_TEXT = "#2d3436"
COLOR_SUBTEXT = "#636e72"
COLOR_BORDER = "#ddd"

# 结果面板全量指标定义
RESULT_SECTIONS = [
    ("Capital & Benchmark", [
        ("initial_cash",     "Initial Capital",    False),
        ("final_value",      "Final Value",        False),
        ("total_return",     "Total Return",       True),
        ("buy_hold_return",  "Buy & Hold Return",  True),
        ("annual_return",    "Annual Return",      True),
        ("annual_volatility","Annual Volatility",  False),
    ]),
    ("Risk-Adjusted Ratios", [
        ("sharpe_ratio",     "Sharpe Ratio",        True),
        ("sortino_ratio",    "Sortino Ratio",       True),
        ("calmar_ratio",     "Calmar Ratio",        True),
        ("max_drawdown",     "Max Drawdown",        True),
        ("max_drawdown_days","Max DD Duration",     False),
    ]),
    ("Trade Statistics", [
        ("trade_count",      "Total Trades",        False),
        ("win_count",        "Wins",                True),
        ("loss_count",       "Losses",              True),
        ("win_rate",         "Win Rate",            True),
        ("avg_win",          "Avg Win",             True),
        ("avg_loss",         "Avg Loss",            True),
        ("profit_factor",    "Profit Factor",       True),
    ]),
]


class StockLookup:
    """股票代码查询器。"""

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
            self._stocks = list(FALLBACK_STOCKS)

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


class BacktestGUI:
    """SMA 双均线交叉回测图形界面。"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SMA Crossover — Quant Backtest")
        self.root.geometry("1050x900")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(True, True)
        self.root.minsize(900, 700)

        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        self._lookup = StockLookup()
        self._last_cerebro = None
        self._last_result = None
        self._alive = True

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_plot_style()
        self._build_ui()

    @staticmethod
    def _configure_plot_style():
        try:
            mplstyle.use("seaborn-v0_8-darkgrid")
        except Exception:
            try:
                mplstyle.use("seaborn-v0_8")
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  UI 布局
    # ------------------------------------------------------------------
    def _build_ui(self):
        # 标题
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=46)
        header.pack(fill=tk.X)
        tk.Label(header, text="SMA Crossover Backtest",
                 font=FONT_TITLE, fg="white", bg=COLOR_PRIMARY, pady=8).pack()

        # 主区域
        main = tk.Frame(self.root, bg=COLOR_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # 左侧：参数
        left = tk.Frame(main, bg=COLOR_BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self._build_input_panel(left)

        # 右侧：结果（可滚动）
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
        card = tk.Frame(parent, bg=COLOR_CARD,
                        highlightbackground=COLOR_BORDER,
                        highlightthickness=1, padx=14, pady=10)
        card.pack()

        tk.Label(card, text="Parameters", font=FONT_HEADING,
                 bg=COLOR_CARD, fg=COLOR_PRIMARY).pack(anchor=tk.W, pady=(0, 6))

        # ---- 股票搜索 ----
        search_section = tk.Frame(card, bg=COLOR_CARD, bd=1, relief=tk.GROOVE)
        search_section.pack(fill=tk.X, pady=(0, 6))

        tk.Label(search_section, text="Stock Lookup", font=("Microsoft YaHei", 9, "bold"),
                 bg=COLOR_CARD, fg=COLOR_ACCENT).pack(anchor=tk.W, padx=6, pady=(4, 0))

        sr = tk.Frame(search_section, bg=COLOR_CARD)
        sr.pack(fill=tk.X, padx=6, pady=(3, 1))
        tk.Label(sr, text="Company:", font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_TEXT, width=9, anchor=tk.W).pack(side=tk.LEFT)
        self.entry_search = tk.Entry(sr, font=FONT_BODY, width=13,
                                     relief=tk.SOLID, borderwidth=1)
        self.entry_search.pack(side=tk.LEFT, padx=(0, 3))
        self.entry_search.bind("<KeyRelease>", self._on_search_typing)
        tk.Button(sr, text="Search", font=FONT_SMALL, bg=COLOR_ACCENT, fg="white",
                  relief=tk.FLAT, cursor="hand2", padx=6,
                  command=self._do_search).pack(side=tk.LEFT)

        lf = tk.Frame(search_section, bg=COLOR_CARD)
        lf.pack(fill=tk.X, padx=6, pady=(1, 4))
        sb = tk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.search_listbox = tk.Listbox(
            lf, font=("Consolas", 9), height=5, yscrollcommand=sb.set,
            relief=tk.SOLID, borderwidth=1)
        self.search_listbox.pack(fill=tk.X)
        self.search_listbox.bind("<Double-Button-1>", self._on_search_select)
        self.search_listbox.bind("<Return>", self._on_search_select)
        sb.config(command=self.search_listbox.yview)

        cr = tk.Frame(search_section, bg=COLOR_CARD)
        cr.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(cr, text="Code:", font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_TEXT, width=9, anchor=tk.W).pack(side=tk.LEFT)
        self.entry_code = tk.Entry(cr, font=("Consolas", 11, "bold"), width=13,
                                   justify=tk.CENTER, relief=tk.SOLID, borderwidth=1)
        self.entry_code.insert(0, "600519")
        self.entry_code.pack(side=tk.LEFT)

        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ---- 回测参数 ----
        self._add_field(card, "Start Date", "2020-01-01", "start")
        self._add_field(card, "End Date", "2025-12-31", "end")
        self._add_field(card, "Short SMA", "10", "short")
        self._add_field(card, "Long SMA", "30", "long")
        self._add_field(card, "Capital", "100000", "cash")

        # ---- 止损控制 ----
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        tk.Label(card, text="Stop-Loss", font=("Microsoft YaHei", 9, "bold"),
                 bg=COLOR_CARD, fg=COLOR_ACCENT2).pack(anchor=tk.W, pady=(2, 2))

        sl_row1 = tk.Frame(card, bg=COLOR_CARD)
        sl_row1.pack(fill=tk.X)
        self.use_sl_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sl_row1, text="Enable Stop-Loss", variable=self.use_sl_var,
                       font=FONT_BODY, bg=COLOR_CARD, fg=COLOR_TEXT,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(side=tk.LEFT)

        sl_row2 = tk.Frame(card, bg=COLOR_CARD)
        sl_row2.pack(fill=tk.X, pady=2)
        tk.Label(sl_row2, text="SL %:", font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_TEXT, width=9, anchor=tk.W).pack(side=tk.LEFT)
        self.entry_sl_pct = tk.Entry(sl_row2, font=FONT_BODY, width=6,
                                     justify=tk.CENTER, relief=tk.SOLID, borderwidth=1)
        self.entry_sl_pct.insert(0, "5")
        self.entry_sl_pct.pack(side=tk.LEFT)
        tk.Label(sl_row2, text="%", font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_SUBTEXT).pack(side=tk.LEFT, padx=(2, 8))

        self.sl_mode_var = tk.StringVar(value="trailing")
        tk.Radiobutton(sl_row2, text="Trailing", variable=self.sl_mode_var,
                       value="trailing", font=FONT_SMALL, bg=COLOR_CARD,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(sl_row2, text="Fixed", variable=self.sl_mode_var,
                       value="fixed", font=FONT_SMALL, bg=COLOR_CARD,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(side=tk.LEFT)

        # ---- 图表选项 ----
        chk_frame = tk.Frame(card, bg=COLOR_CARD)
        chk_frame.pack(fill=tk.X, pady=(6, 0))
        self.show_chart_var = tk.BooleanVar(value=True)
        self.save_chart_var = tk.BooleanVar(value=True)
        tk.Checkbutton(chk_frame, text="Show Chart", variable=self.show_chart_var,
                       font=FONT_BODY, bg=COLOR_CARD, fg=COLOR_TEXT,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(anchor=tk.W)
        tk.Checkbutton(chk_frame, text="Save Chart to results/",
                       variable=self.save_chart_var,
                       font=FONT_BODY, bg=COLOR_CARD, fg=COLOR_TEXT,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(anchor=tk.W)
        self.force_refresh_var = tk.BooleanVar(value=False)
        tk.Checkbutton(chk_frame, text="Force Refresh Data",
                       variable=self.force_refresh_var,
                       font=FONT_BODY, bg=COLOR_CARD, fg=COLOR_TEXT,
                       activebackground=COLOR_CARD, selectcolor=COLOR_CARD,
                       ).pack(anchor=tk.W)

        # ---- 按钮 ----
        btn_frame = tk.Frame(card, bg=COLOR_CARD)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = tk.Button(btn_frame, text="▶  Run Backtest",
                                 font=FONT_HEADING, bg=COLOR_ACCENT, fg="white",
                                 activebackground="#0773c5", activeforeground="white",
                                 relief=tk.FLAT, cursor="hand2", padx=14, pady=6,
                                 command=self._on_run)
        self.run_btn.pack(fill=tk.X)

        self.opt_btn = tk.Button(btn_frame, text="⚡ Optimize (Grid Search)",
                                 font=FONT_HEADING, bg=COLOR_ACCENT2, fg="white",
                                 activebackground="#5541d1", activeforeground="white",
                                 relief=tk.FLAT, cursor="hand2", padx=14, pady=6,
                                 command=self._on_optimize)
        self.opt_btn.pack(fill=tk.X, pady=(6, 0))

        self.progress_var = tk.StringVar(value="")
        tk.Label(card, textvariable=self.progress_var, font=FONT_SMALL,
                 fg=COLOR_SUBTEXT, bg=COLOR_CARD).pack(pady=(6, 0))

    def _add_field(self, parent, label, default, attr):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_TEXT, width=14, anchor=tk.W).pack(side=tk.LEFT)
        e = tk.Entry(row, font=FONT_BODY, width=14, justify=tk.CENTER,
                     relief=tk.SOLID, borderwidth=1)
        e.insert(0, default)
        e.pack(side=tk.LEFT)
        setattr(self, f"entry_{attr}", e)

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
        sel = self.search_listbox.curselection()
        if not sel:
            return
        text = self.search_listbox.get(sel[0])
        code = text.split()[0] if text else ""
        self.entry_code.delete(0, tk.END)
        self.entry_code.insert(0, code)
        self.status_var.set(f"Selected: {text}")

    # ------------------------------------------------------------------
    #  右侧结果面板（可滚动）
    # ------------------------------------------------------------------
    def _build_result_panel(self, parent: tk.Frame):
        # Canvas + Scrollbar 实现滚动区域
        canvas = tk.Canvas(parent, bg=COLOR_CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.result_frame = tk.Frame(canvas, bg=COLOR_CARD)

        self.result_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.result_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 用鼠标滚轮滚动
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 内容
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
            text="Enter parameters and click 'Run Backtest'\nor 'Optimize' for grid search.",
            font=FONT_BODY, fg="#b2bec3", bg=COLOR_CARD, justify=tk.CENTER)
        self.placeholder.pack(expand=True, pady=40)

    def _add_result_row(self, parent, label, key):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill=tk.X, padx=16, pady=1)
        tk.Label(row, text=label, font=FONT_BODY, bg=COLOR_CARD,
                 fg=COLOR_SUBTEXT, width=22, anchor=tk.W).pack(side=tk.LEFT)
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
        symbol = self.entry_code.get().strip()
        if not symbol:
            raise ValueError("Please enter a stock code.")

        start_date = self.entry_start.get().strip().replace("-", "")
        end_date = self.entry_end.get().strip().replace("-", "")
        short_period = int(self.entry_short.get().strip())
        long_period = int(self.entry_long.get().strip())
        cash = float(self.entry_cash.get().strip())

        if short_period >= long_period:
            raise ValueError("Short SMA must be less than Long SMA.")

        use_sl = self.use_sl_var.get()
        sl_pct = float(self.entry_sl_pct.get().strip()) / 100.0
        trailing = self.sl_mode_var.get() == "trailing"
        force_refresh = self.force_refresh_var.get()

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "short_period": short_period,
            "long_period": long_period,
            "cash": cash,
            "use_stop_loss": use_sl,
            "stop_loss_pct": sl_pct,
            "use_trailing": trailing,
            "force_refresh": force_refresh,
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

        show_chart = self.show_chart_var.get()
        save_chart = self.save_chart_var.get()

        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.opt_btn.config(state=tk.DISABLED)
        self._safe_update(self.progress_var.set, "Running backtest...")
        self._safe_update(self.status_var.set, "Running backtest...")
        try:
            self.placeholder.pack_forget()
        except (tk.TclError, AttributeError):
            pass

        def worker():
            try:
                result, cerebro = run_backtest(
                    symbol=p["symbol"],
                    start_date=p["start_date"],
                    end_date=p["end_date"],
                    short_period=p["short_period"],
                    long_period=p["long_period"],
                    cash=p["cash"],
                    use_stop_loss=p["use_stop_loss"],
                    stop_loss_pct=p["stop_loss_pct"],
                    use_trailing=p["use_trailing"],
                    verbose=False,
                    return_cerebro=True,
                    force_refresh=p["force_refresh"],
                )
                self._last_result = result
                self._last_cerebro = cerebro
                if self._alive:
                    self.root.after(0, lambda: self._on_backtest_done(
                        show_chart, save_chart,
                        p["symbol"], p["short_period"], p["long_period"],
                        p["start_date"], p["end_date"]))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if self._alive:
                    self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    #  参数优化
    # ------------------------------------------------------------------
    def _on_optimize(self):
        try:
            p = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self.run_btn.config(state=tk.DISABLED)
        self.opt_btn.config(state=tk.DISABLED, text="Optimizing...")
        self._safe_update(self.progress_var.set, "Grid search running...")
        self._safe_update(self.status_var.set, "Optimizing SMA parameters...")

        def worker():
            try:
                results = run_optimization(
                    symbol=p["symbol"],
                    start_date=p["start_date"],
                    end_date=p["end_date"],
                    cash=p["cash"],
                    use_stop_loss=p["use_stop_loss"],
                    stop_loss_pct=p["stop_loss_pct"],
                    use_trailing=p["use_trailing"],
                    top_n=30,
                    force_refresh=p["force_refresh"],
                )
                if self._alive:
                    self.root.after(0, lambda: self._show_optimization(results))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if self._alive:
                    self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_optimization(self, results: list):
        """弹出新窗口展示网格搜索结果表格。"""
        self.run_btn.config(state=tk.NORMAL, text="▶  Run Backtest")
        self.opt_btn.config(state=tk.NORMAL, text="⚡ Optimize (Grid Search)")
        self._safe_update(self.progress_var.set, "")
        self._safe_update(self.status_var.set,
                          f"Optimization done — {len(results)} results shown.")

        if not results:
            messagebox.showinfo("Optimization", "No valid parameter combinations found.")
            return

        # 弹出窗口
        win = tk.Toplevel(self.root)
        win.title("Grid Search Results — SMA Crossover")
        win.geometry("850x550")
        win.configure(bg=COLOR_CARD)

        tk.Label(win, text="Top Parameter Combinations (by Sharpe Ratio)",
                 font=FONT_HEADING, bg=COLOR_CARD, fg=COLOR_PRIMARY,
                 ).pack(pady=(12, 6))

        # 表格
        columns = ("#", "Short", "Long", "Return%", "Sharpe", "MaxDD%",
                   "WinRate%", "Trades", "Final$")
        tree = ttk.Treeview(win, columns=columns, show="headings",
                            height=min(len(results), 20))
        widths = [40, 60, 60, 80, 70, 80, 80, 60, 100]
        for col, w in zip(columns, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor=tk.CENTER)

        for i, r in enumerate(results, 1):
            tree.insert("", tk.END, values=(
                i, r.short_period, r.long_period,
                f"{r.total_return:.2f}", f"{r.sharpe_ratio:.2f}",
                f"{r.max_drawdown:.2f}", f"{r.win_rate:.1f}",
                r.trade_count, f"{r.final_value:,.0f}",
            ))

        # 高亮第一行（最优）
        tree.tag_configure("best", background="#dff9fb")
        if tree.get_children():
            tree.item(tree.get_children()[0], tags=("best",))

        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # 双击某行 → 自动填入参数
        def _on_double_click(event):
            sel = tree.selection()
            if not sel:
                return
            vals = tree.item(sel[0], "values")
            self.entry_short.delete(0, tk.END)
            self.entry_short.insert(0, vals[1])
            self.entry_long.delete(0, tk.END)
            self.entry_long.insert(0, vals[2])
            self._safe_update(self.status_var.set,
                              f"Applied: Short={vals[1]} Long={vals[2]}")
            win.destroy()

        tree.bind("<Double-1>", _on_double_click)

        tk.Label(win, text="Double-click a row to apply parameters to the main window.",
                 font=FONT_SMALL, fg=COLOR_SUBTEXT, bg=COLOR_CARD,
                 ).pack(pady=(0, 10))

    # ------------------------------------------------------------------
    #  回测完成处理
    # ------------------------------------------------------------------
    def _on_backtest_done(self, show_chart, save_chart,
                           symbol, short, long, start, end):
        if not self._alive or self._last_result is None:
            return

        self._display_result(self._last_result)

        if show_chart or save_chart:
            self._handle_chart(self._last_cerebro, show_chart, save_chart,
                               symbol, short, long, start, end)

        self._safe_update(self.run_btn.config, state=tk.NORMAL, text="▶  Run Backtest")
        self._safe_update(self.opt_btn.config, state=tk.NORMAL,
                          text="⚡ Optimize (Grid Search)")
        self._safe_update(self.progress_var.set, "")

    def _handle_chart(self, cerebro, show, save, symbol, short, long, start, end):
        if not self._alive:
            return

        self._safe_update(self.status_var.set, "Generating chart...")
        self.root.update_idletasks()

        figure = cerebro.plot(
            style="candlestick",
            barup="red", bardown="green",
            volup="red", voldown="green",
            grid=True, plotdist=0.06, dpi=120,
        )

        if save:
            os.makedirs("results", exist_ok=True)
            fname = os.path.join("results",
                                 f"{symbol}_SMA{short}-{long}_{start}-{end}.png")
            for fig_group in figure:
                for i, fig in enumerate(fig_group):
                    if hasattr(fig, "savefig"):
                        base, ext = os.path.splitext(fname)
                        path = f"{base}_{i}{ext}" if len(fig_group) > 1 else fname
                        parent = os.path.dirname(os.path.abspath(path))
                        if parent:
                            os.makedirs(parent, exist_ok=True)
                        fig.savefig(path, bbox_inches="tight", facecolor="white")
                        self._safe_update(self.status_var.set,
                                          f"Chart saved: {os.path.abspath(path)}")

        if show:
            plt.tight_layout()
            self._safe_update(self.status_var.set,
                              self.status_var.get() + "  |  Chart window opened.")
            plt.show()

    def _display_result(self, result: BacktestResult):
        green, red = COLOR_GREEN, COLOR_RED

        def fmt(val, suffix="", decimals=2, sign_color=True):
            text = f"{val:,.{decimals}f}{suffix}"
            if sign_color and isinstance(val, (int, float)):
                c = green if val > 0 else (red if val < 0 else COLOR_TEXT)
            else:
                c = COLOR_TEXT
            return text, c

        updates = {
            "initial_cash":     fmt(result.initial_cash, sign_color=False),
            "final_value":      fmt(result.final_value, sign_color=False),
            "total_return":     fmt(result.total_return, suffix="%"),
            "buy_hold_return":  fmt(result.buy_hold_return, suffix="%"),
            "annual_return":    fmt(result.annual_return, suffix="%"),
            "annual_volatility":fmt(result.annual_volatility, suffix="%", sign_color=False),
            "sharpe_ratio":     fmt(result.sharpe_ratio),
            "sortino_ratio":    fmt(result.sortino_ratio),
            "calmar_ratio":     fmt(result.calmar_ratio),
            "max_drawdown":     fmt(result.max_drawdown, suffix="%", sign_color=False),
            "max_drawdown_days":(f"{result.max_drawdown_days}", COLOR_TEXT),
            "trade_count":      (f"{result.trade_count}", COLOR_TEXT),
            "win_count":        (f"{result.win_count}", green),
            "loss_count":       (f"{result.loss_count}", red),
            "win_rate":         fmt(result.win_rate, suffix="%"),
            "avg_win":          fmt(result.avg_win, sign_color=False),
            "avg_loss":         fmt(result.avg_loss, sign_color=False),
            "profit_factor":    fmt(result.profit_factor, sign_color=False),
        }

        for key, (text, color) in updates.items():
            if key in self.result_labels:
                self.result_labels[key].config(text=text, fg=color)

    def _on_error(self, msg: str):
        self._safe_update(self.run_btn.config, state=tk.NORMAL, text="▶  Run Backtest")
        self._safe_update(self.opt_btn.config, state=tk.NORMAL,
                          text="⚡ Optimize (Grid Search)")
        self._safe_update(self.progress_var.set, "")
        self._safe_update(self.status_var.set, "Error occurred.")
        messagebox.showerror("Backtest Error", msg)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BacktestGUI()
    app.run()
