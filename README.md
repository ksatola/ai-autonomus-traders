# Autonomous Traders
The aim of this prototype - an equity trading simulation - is to illustrate autonomous agents powered by tools and resources from MCP servers. It uses 4 trader agents (able to make trading decisions) using market researcher agents each (converted to a tool).

Selected components in use:
- OpenAI SDK
- Trader agents use:
    - Financial data - MCP client and server from Polygon.io official Github repository using Polygon.io API (free plan) to get market data (share price)
    - Account details - Homegrown MCP client and server implementation to work with a an shareholder account settings and a local sqlite database (resource). The Account module logic was created by AI-based development team (as part of another project).
    - Push notifications - Homegrown MCP client and server to push notifications using Pushover API (free plan). Trader agents send push notifications as part of their operations.
- Research agent (converted to a tool) uses:
    - Fetch MCP server to download entire web pages
    - Brave MCP server for web searches
    - LibSQL MCP server to manage SQL-based memory (separate for every trader agent)

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
```

## Run
```

```