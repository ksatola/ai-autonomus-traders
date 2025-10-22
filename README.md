# Autonomous Traders
The aim of this prototype - an equity trading simulation - is to illustrate autonomous agents powered by tools and resources from MCP servers. Autonomous Traders simulates a trading floor with 4 trader agents (able to make autonomous trading decisions) using helper market researcher agents (converted to tools). Each trading agent starts with a given investment strategy to be subject to changed if they wish to do so. The simulation allows also manual trader accounts management (deposit cash, withdraw cash, place buy or sell orders).

Selected components in use:
- OpenAI SDK extended with API use of tracers for logging into trader's sqlite database
- Trader agents use:
    - Financial data - MCP client and server from Polygon.io official Github repository using Polygon.io API (free plan) to get market data (share price)
    - Account details - Homegrown MCP client and server implementation to work with a an shareholder account settings and a local sqlite database (resource). The Account module logic was created by AI-based development team (as part of another project).
    - Push notifications - Homegrown MCP client and server to push notifications using Pushover API (free plan). Trader agents send push notifications as part of their operations.
- Research agent (converted to a tool), on behalf of a trader agent, uses:
    - Fetch MCP server to download entire web pages
    - Brave MCP server for web searches
    - LibSQL MCP server to manage SQL-based memory (separate for every trader agent)
- Gradio UI

## Setup
```
$ uv sync
$ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
# Open a new terminal session
$ nvm install --lts
$ node -v
$ npm -v
$ npx -v
# Make sure the "memory" folder exists

# Export Jupyter notebook (Trader_Agent.ipynb) to HTML
$ uvx jupyter nbconvert --to html Trader_Agent.ipynb

# Modify environment variables in .env to adjust application settings:
RUN_EVERY_N_MINUTES=60
RUN_EVEN_WHEN_MARKET_IS_CLOSED=True
USE_MANY_MODELS=False
```

## Run
```
# First run only (resetting traders strategy)
$ uv run traders_strategy.py

# Run the app in Terminal 1
uv run app.py

# Run the trading floor in Terminal 2
uv run trading_floor.py
```