import gradio as gr
from ui_utils import css, js, Color
import pandas as pd
from trading_floor import names, lastnames, short_model_names
import plotly.express as px
from accounts import Account
from database import read_log
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime
from traders_strategy import reset_traders
import plotly.express as px
import pandas as pd

mapper = {
    "trace": Color.WHITE,
    "agent": Color.CYAN,
    "function": Color.GREEN,
    "generation": Color.YELLOW,
    "response": Color.MAGENTA,
    "account": Color.RED,
}


class Trader:
    def __init__(self, name: str, lastname: str, model_name: str):
        self.name = name
        self.lastname = lastname
        self.model_name = model_name
        self.account = Account.get(name)

    def reload(self):
        self.account = Account.get(self.name)

    def get_title(self) -> str:
        return f"<div style='text-align: center;font-size:34px;'>{self.name}<span style='color:#ccc;font-size:24px;'> ({self.model_name}) - {self.lastname}</span></div>"

    def get_name(self) -> str:
        return self.name
    
    def get_strategy(self) -> str:
        return self.account.get_strategy()

    def get_portfolio_value_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.account.portfolio_value_time_series, columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def get_portfolio_value_chart(self):
        df = self.get_portfolio_value_df()
        fig = px.line(df, x="datetime", y="value")
        margin = dict(l=40, r=20, t=20, b=40)
        fig.update_layout(
            height=300,
            margin=margin,
            xaxis_title=None,
            yaxis_title=None,
            paper_bgcolor="#bbb",
            plot_bgcolor="#dde",
        )
        fig.update_xaxes(tickformat="%m/%d", tickangle=45, tickfont=dict(size=8))
        fig.update_yaxes(tickfont=dict(size=8), tickformat=",.0f")
        return fig

    def get_holdings_df(self) -> pd.DataFrame:
        """Convert holdings to DataFrame for display"""
        holdings = self.account.get_holdings()
        if not holdings:
            return pd.DataFrame(columns=["Symbol", "Quantity"])

        df = pd.DataFrame(
            [{"Symbol": symbol, "Quantity": quantity} for symbol, quantity in holdings.items()]
        )
        return df

    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions to DataFrame for display"""
        transactions = self.account.list_transactions()
        if not transactions:
            return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])

        return pd.DataFrame(transactions)

    def get_portfolio_value(self) -> str:
        """Calculate total portfolio value based on current prices"""
        portfolio_value = self.account.calculate_portfolio_value() or 0.0
        pnl = self.account.calculate_profit_loss(portfolio_value) or 0.0
        color = "green" if pnl >= 0 else "red"
        emoji = "⬆" if pnl >= 0 else "⬇"
        return f"<div style='text-align: center;background-color:{color};'><span style='font-size:32px'>${portfolio_value:,.0f}</span><span style='font-size:24px'>&nbsp;&nbsp;&nbsp;{emoji}&nbsp;${pnl:,.0f}</span></div>"

    def get_logs(self, previous=None) -> str:
        logs = read_log(self.name, last_n=13)
        response = ""
        for log in logs:
            timestamp, type, message = log
            color = mapper.get(type, Color.WHITE).value
            response += f"<span style='color:{color}'>{timestamp} : [{type}] {message}</span><br/>"
        response = f"<div style='height:250px; overflow-y:auto;'>{response}</div>"
        if response != previous:
            return response
        return gr.update()


class TraderView:
    def __init__(self, trader: Trader):
        self.trader = trader
        self.portfolio_value = None
        self.chart = None
        self.holdings_table = None
        self.transactions_table = None

    def make_ui(self):
        with gr.Column():
            gr.HTML(self.trader.get_title())
            with gr.Row():
                self.portfolio_value = gr.HTML(self.trader.get_portfolio_value)
            with gr.Row():
                self.chart = gr.Plot(
                    self.trader.get_portfolio_value_chart, container=True, show_label=False
                )
            with gr.Row(variant="panel"):
                self.log = gr.HTML(self.trader.get_logs)
            with gr.Row():
                self.holdings_table = gr.Dataframe(
                    value=self.trader.get_holdings_df,
                    label="Holdings",
                    headers=["Symbol", "Quantity"],
                    row_count=(10, "dynamic"),
                    col_count=2,
                    max_height=300,
                    elem_classes=["dataframe-fix-small"],
                )
            with gr.Row():
                self.transactions_table = gr.Dataframe(
                    value=self.trader.get_transactions_df,
                    label="Recent Transactions",
                    headers=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"],
                    row_count=(10, "dynamic"),
                    col_count=5,
                    max_height=300,
                    elem_classes=["dataframe-fix"],
                )

        timer = gr.Timer(value=120)
        timer.tick(
            fn=self.refresh,
            inputs=[],
            outputs=[
                self.portfolio_value,
                self.chart,
                self.holdings_table,
                self.transactions_table,
            ],
            show_progress="hidden",
            queue=False,
        )
        log_timer = gr.Timer(value=0.5)
        log_timer.tick(
            fn=self.trader.get_logs,
            inputs=[self.log],
            outputs=[self.log],
            show_progress="hidden",
            queue=False,
        )

    def refresh(self):
        self.trader.reload()
        return (
            self.trader.get_portfolio_value(),
            self.trader.get_portfolio_value_chart(),
            self.trader.get_holdings_df(),
            self.trader.get_transactions_df(),
        )



def _account_plot2(portfolio_value_time_series: list[tuple[str, float]]):
    """Return a matplotlib Figure of portfolio value over time."""
    fig, ax = plt.subplots()
    if portfolio_value_time_series:
        xs = [t for t, _ in portfolio_value_time_series]
        ys = [v for _, v in portfolio_value_time_series]
        ax.plot(xs, ys, marker="o")
        ax.set_xticks(xs[:: max(1, len(xs)//8) ])  # avoid overcrowded ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    fig.tight_layout()
    plt.close(fig)
    return fig

def _account_plot(portfolio_value_time_series: list[tuple[str, float]]):
    df = pd.DataFrame(portfolio_value_time_series, columns=["datetime", "value"])
    if df.empty:
        # minimal empty figure
        return px.line(pd.DataFrame({"datetime": [], "value": []}), x="datetime", y="value", title="Portfolio Value Over Time")
    df["datetime"] = pd.to_datetime(df["datetime"])
    fig = px.line(df, x="datetime", y="value", title="Portfolio Value Over Time")
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), height=300)
    fig.update_xaxes(tickformat="%m/%d", tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8), tickformat=",.0f")
    return fig

def _account_holdings_df(holdings: Dict[str, int]) -> pd.DataFrame:
    if not holdings:
        return pd.DataFrame(columns=["Symbol", "Quantity"])
    return pd.DataFrame(
        [{"Symbol": sym, "Quantity": qty} for sym, qty in holdings.items()],
        columns=["Symbol", "Quantity"],
    )

def _account_transactions_df(txs: list[Dict[str, Any]]) -> pd.DataFrame:
    if not txs:
        return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])
    # Your Transaction has keys: symbol, quantity, price, timestamp, rationale
    rows = [
        {
            "Timestamp": tx.get("timestamp"),
            "Symbol": tx.get("symbol"),
            "Quantity": tx.get("quantity"),
            "Price": tx.get("price"),
            "Rationale": tx.get("rationale"),
        }
        for tx in txs
    ]
    return pd.DataFrame(rows, columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])

def _account_summary_html(acct: Account) -> str:
    pv = acct.calculate_portfolio_value()
    pnl = acct.get_profit_loss(pv)
    return f"""
    <div>
        <h3>Account: {acct.name}</h3>
        <p><b>Balance:</b> {acct.balance:,.2f}</p>
        <p><b>Total Portfolio Value:</b> {pv:,.2f}</p>
        <p><b>Profit and Loss (P&amp;L):</b> {pnl:,.2f}</p>
    </div>
    """


def _refresh_account_view(name: str, show_toast: bool = False):
    """
    Pull latest account from storage and return all UI-bound pieces:
    portfolio value HTML, chart, holdings df, transactions df, strategy, JSON report, status.
    """
    try:
        acct = Account.get(name)
        # report() appends the current pv point & persists it
        report_json_str = acct.report()  # also writes a new time-series point
        holdings_df = _account_holdings_df(acct.get_holdings())
        tx_df = _account_transactions_df(acct.list_transactions())
        chart = _account_plot(acct.portfolio_value_time_series)
        m = f"Loaded {name} (last update on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})."
        if show_toast:
            gr.Info(m)
        return (
            _account_summary_html(acct),
            chart,
            holdings_df,
            tx_df,
            acct.get_strategy(),
            report_json_str,
            m,
        )
    except Exception as e:
        _notify("error", f"Failed to load {name}: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Failed to load {name}: {e}"


def _act_and_refresh(
    name: str,
    action: str,
    amount: float = 0.0,
    symbol: str = "",
    quantity: int = 0,
    rationale: str = "",
    strategy: str = "",
):
    """
    Execute an Account action, then return fresh UI payloads.
    """
    try:
        acct = Account.get(name)

        # Validation
        if action in ("buy", "sell"):
            if not symbol or not symbol.strip():
                msg = "Please enter a symbol before placing an order."
                gr.Warning(msg)  # ← reliable toast without raising
                pv, chart, hold, tx, strat, rep, _ = _refresh_account_view(name, show_toast=False)
                return pv, chart, hold, tx, strat, rep, msg
            if not quantity or int(quantity) <= 0:
                msg = "Quantity must be a positive integer."
                gr.Warning(msg)
                pv, chart, hold, tx, strat, rep, _ = _refresh_account_view(name, show_toast=False)
                return pv, chart, hold, tx, strat, rep, msg

        if action in ("deposit", "withdraw"):
            if amount is None or float(amount) <= 0:
                msg = "Amount must be a positive number."
                gr.Warning(msg)
                pv, chart, hold, tx, strat, rep, _ = _refresh_account_view(name, show_toast=False)
                return pv, chart, hold, tx, strat, rep, msg

        # Perform action
        if action == "deposit":
            acct.deposit(float(amount))
            _notify("info", f"Deposited {amount:.2f} to {name}.")
            status = f"Deposited {amount:.2f} to {name}."
        elif action == "withdraw":
            acct.withdraw(float(amount))
            _notify("info", f"Withdrew {amount:.2f} from {name}.")
            status = f"Withdrew {amount:.2f} from {name}."
        elif action == "buy":
            if not symbol or not quantity:
                raise ValueError("Symbol and positive quantity required for Buy.")
            acct.buy_shares(symbol.strip().upper(), int(quantity), rationale or "Manual buy")
            _notify("info", f"Bought {quantity} {symbol.upper()} for {name}.")
            status = f"Bought {quantity} {symbol.upper()} for {name}."
        elif action == "sell":
            if not symbol or not quantity:
                raise ValueError("Symbol and positive quantity required for Sell.")
            acct.sell_shares(symbol.strip().upper(), int(quantity), rationale or "Manual sell")
            _notify("info", f"Sold {quantity} {symbol.upper()} for {name}.")
            status = f"Sold {quantity} {symbol.upper()} for {name}."
        elif action == "change_strategy":
            acct.change_strategy(strategy or "")
            _notify("info", f"Changed strategy for {name}.")
            status = f"Changed strategy for {name}."
        elif action == "reset":
            acct.reset(strategy or "")
            _notify("warn", f"Reset {name} (strategy set).")
            status = f"Reset {name} (strategy set)."
        elif action == "full_reset":
            reset_traders()
            _notify("warn", "All traders and accounts were reset to factory defaults.")
            status = "All traders and accounts were reset to factory defaults."
        elif action == "refresh":
            m = f"Refreshed {name} (last update on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})."
            _notify("info", m)
            status = m
        else:
            _notify("error", f"Unknown action: {action}")
            raise ValueError(f"Unknown action: {action}")

        # Return updated views (silent toast here to avoid duplicates)
        pv_html, chart, holdings_df, tx_df, strat, report_json, _ = _refresh_account_view(name, show_toast=False)
        return pv_html, chart, holdings_df, tx_df, strat, report_json, status

    except Exception as e:
        # Attempt to still show latest state even on error
        _notify("error", f"Action failed: {e}")
        try:
            pv_html, chart, holdings_df, tx_df, strat, report_json, _ = _refresh_account_view(name, show_toast=False)
            return pv_html, chart, holdings_df, tx_df, strat, report_json, f"Action failed: {e}"
        except Exception as e2:
            _notify("error", f"Action failed: {e}; also failed to refresh: {e2}")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Action failed: {e}; also failed to refresh: {e2}"

def _notify(kind: str, msg: str):
    if kind == "error":
        gr.Error(msg)     # red popup
    elif kind == "warn":
        gr.Warning(msg)   # yellow popup
    else:
        gr.Info(msg)      # blue popup

def _full_reset_then_refresh(selected_name: str):
    """
    Run a global factory reset and refresh the Accounts panel for the selected (or default) trader.
    """
    try:
        reset_traders()
        gr.Warning("All traders and accounts were reset to factory defaults.")
    except Exception as e:
        gr.Error(f"Full reset failed: {e}")

    # After reset, show updated state for the selected or default trader
    name = selected_name
    if not name:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""
    return _refresh_account_view(name, show_toast=True)


# Main UI construction
def create_ui():
    """
    Create the main Gradio UI for the trading simulation
    """

    traders = [
        Trader(trader_name, lastname, model_name)
        for trader_name, lastname, model_name in zip(names, lastnames, short_model_names)
    ]
    trader_views = [TraderView(trader) for trader in traders]

    trader_names = [t.name for t in traders]
    traders_by_name: Dict[str, "Trader"] = {t.name: t for t in traders}
    default_name = trader_names[0] if trader_names else None

    with gr.Blocks(
        title="Autonomous Traders", 
        css=css, js=js, 
        theme=gr.themes.Default(primary_hue="sky"), 
        fill_width=True
    ) as ui:
        
        with gr.Tabs():

            with gr.Tab("### Dashboard") as tab_dashboard:
                with gr.Row():
                    for trader_view in trader_views:
                        trader_view.make_ui()

            with gr.Tab("### Accounts") as tab_accounts:
                gr.Markdown("### Manual Account Management")

                status_bar = gr.Markdown("")

                with gr.Row():
                    trader_select = gr.Dropdown(
                        choices=trader_names,
                        value=default_name,
                        label="Select Trader (by .name)",
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**Cash**")
                        cash_amount = gr.Number(label="Amount", value=1000.0, precision=2)
                        with gr.Row():
                            btn_deposit = gr.Button("💰 Deposit", variant="primary")
                            btn_withdraw = gr.Button("🏧 Withdraw")

                    with gr.Column(scale=3):
                        gr.Markdown("**Trading**")
                        symbol = gr.Textbox(label="Symbol (e.g., AAPL)")
                        quantity = gr.Number(label="Quantity (int)", value=1, precision=0)
                        rationale = gr.Textbox(label="Rationale (optional)", value="Manual action")
                        with gr.Row():
                            btn_buy = gr.Button("🟢 Buy", variant="primary")
                            btn_sell = gr.Button("🔴 Sell")

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("**Strategy**")
                        strategy_box = gr.Textbox(label="Current / New Strategy", lines=3, placeholder="Describe or paste your strategy…")
                        with gr.Row():
                            btn_change_strategy = gr.Button("✏️ Change Strategy")
                            btn_reset = gr.Button("♻️ Reset Account (sets new strategy & clears state)")
                            btn_full_reset = gr.Button(f"⚠️ Reset Account (factory defaults)")
                    with gr.Column(scale=2):
                        btn_refresh = gr.Button("🔄 Refresh", variant="secondary")

                gr.Markdown("---")
                with gr.Row():
                    acct_value = gr.HTML(label="Account Summary")
                with gr.Row():
                    acct_chart = gr.Plot(show_label=False, container=True)
                with gr.Row():
                    acct_holdings = gr.Dataframe(
                        label="Holdings",
                        headers=["Symbol", "Quantity"],
                        row_count=(10, "dynamic"),
                        col_count=2,
                        max_height=300,
                        elem_classes=["dataframe-fix-small"],
                    )
                with gr.Row():
                    acct_tx = gr.Dataframe(
                        label="Recent Transactions",
                        headers=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"],
                        row_count=(10, "dynamic"),
                        col_count=5,
                        max_height=300,
                        elem_classes=["dataframe-fix"],
                    )
                with gr.Row():
                    acct_report = gr.JSON(label="Account Report (JSON)")

                # ------ wiring ------
                def _wrap(action):
                    def runner(selected_name, _amount, _symbol, _qty, _rationale, _strategy):
                        return _act_and_refresh(
                            name=selected_name,
                            action=action,
                            amount=_amount or 0.0,
                            symbol=_symbol or "",
                            quantity=int(_qty or 0),
                            rationale=_rationale or "",
                            strategy=_strategy or "",
                        )
                    return runner

                outputs_all = [acct_value, acct_chart, acct_holdings, acct_tx, strategy_box, acct_report, status_bar]

                # cash
                btn_deposit.click(_wrap("deposit"),
                                  inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                  outputs=outputs_all, show_progress="hidden")
                btn_withdraw.click(_wrap("withdraw"),
                                   inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                   outputs=outputs_all, show_progress="hidden")
                # trade
                btn_buy.click(_wrap("buy"),
                              inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                              outputs=outputs_all, show_progress="hidden")
                btn_sell.click(_wrap("sell"),
                               inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                               outputs=outputs_all, show_progress="hidden")
                # strategy
                btn_change_strategy.click(_wrap("change_strategy"),
                                          inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                          outputs=outputs_all, show_progress="hidden")
                btn_reset.click(_wrap("reset"),
                                inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                outputs=outputs_all, show_progress="hidden")
                btn_full_reset.click(_wrap("full_reset"),
                                inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                outputs=outputs_all, show_progress="hidden")
                # load/refresh
                trader_select.change(lambda n: _refresh_account_view(n), inputs=[trader_select], outputs=outputs_all, show_progress="hidden")
                btn_refresh.click(_wrap("refresh"),
                                  inputs=[trader_select, cash_amount, symbol, quantity, rationale, strategy_box],
                                  outputs=outputs_all, show_progress="hidden")

                # periodic updater to capture price changes in chart/value
                acct_timer = gr.Timer(value=60)
                acct_timer.tick(fn=lambda n: _refresh_account_view(n),
                                inputs=[trader_select],
                                outputs=outputs_all,
                                show_progress="hidden",
                                queue=False)
                
                # collect all dashboard outputs in order: [pv, chart, holdings, tx] per trader
                dashboard_outputs = []
                for tv in trader_views:
                    dashboard_outputs.extend([tv.portfolio_value, tv.chart, tv.holdings_table, tv.transactions_table])

                # function to return concatenated refresh tuples for all traders
                def _refresh_dashboard_all():
                    out = []
                    for tv in trader_views:
                        # tv.refresh() returns (pv, chart, holdings_df, transactions_df)
                        out.extend(list(tv.refresh()))
                    return tuple(out)

                # when user switches back to Dashboard tab, refresh all cards at once
                tab_dashboard.select(
                    fn=_refresh_dashboard_all,
                    inputs=[],
                    outputs=dashboard_outputs,
                    show_progress="hidden",
                    queue=False,
                )

                # Bundle all outputs from the Accounts panel
                outputs_all = [
                    acct_value, acct_chart, acct_holdings, acct_tx, strategy_box, acct_report, status_bar
                ]

                # When entering the Accounts tab, refresh using the current dropdown value,
                # falling back to the default_name if the dropdown is empty.
                def _on_accounts_tab_open(selected_name):
                    name = selected_name or default_name
                    if not name:
                        _notify("warn", "No traders available.")
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "No traders available."
                    return _refresh_account_view(name)

                tab_accounts.select(
                    fn=_on_accounts_tab_open,
                    inputs=[trader_select],   # current dropdown value (may be None on first open)
                    outputs=outputs_all,
                    show_progress="hidden",
                    queue=False,
                )

    return ui


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(inbrowser=True)
