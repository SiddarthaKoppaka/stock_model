"""
DiffSTOCK AI Investment Advisor

Claude-powered assessment layer that combines:
    1. DiffSTOCK model predictions (Tool 1)
    2. Real-time market data via yfinance (Tool 2)
    3. Real-time news & sentiment (Tool 3)

Produces three types of assessments:
    - Weekly Portfolio Brief
    - Single Stock Deep Dive
    - Signal Conflict Report

Usage:
    python -m src.advisor.advisor --mode weekly --date 2024-10-14
    python -m src.advisor.advisor --mode stock --symbol RELIANCE
    python -m src.advisor.advisor --mode conflicts --date 2024-10-14
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── lazy imports for tools ──────────────────────────────────────────────────
from .tools.model_tool import ModelPredictionsTool, MODEL_TOOL_SCHEMA
from .tools.market_tool import get_market_data, MARKET_TOOL_SCHEMA
from .tools.news_tool import get_stock_news, NEWS_TOOL_SCHEMA

# ── Anthropic SDK ───────────────────────────────────────────────────────────
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed. Install with: pip install anthropic")


class DiffSTOCKAdvisor:
    """
    AI-powered investment assessment tool using Claude as the reasoning layer.

    Combines DiffSTOCK model predictions, live market data, and news sentiment
    to produce actionable investment assessments.
    """

    def __init__(
        self,
        simulator=None,
        config: Optional[Dict] = None,
        model_name: str = 'claude-sonnet-4-20250514',
    ):
        """
        Args:
            simulator: DiffSTOCKSimulator instance (model already loaded).
                       If None, tools that need the model will return stubs.
            config: optional config dict
            model_name: Claude model to use
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The anthropic package is required. Install with: pip install anthropic"
            )

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Export it before running the advisor."
            )

        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.simulator = simulator
        self.config = config or {}

        # ── load system prompt ───────────────────────────────────────────────
        prompt_path = Path(__file__).parent / 'prompts' / 'system_prompt.txt'
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()

        # ── model predictions tool ───────────────────────────────────────────
        if simulator is not None:
            self._model_tool = ModelPredictionsTool(simulator)
        else:
            self._model_tool = None

        # ── tool definitions for Claude ──────────────────────────────────────
        self.tools = [MODEL_TOOL_SCHEMA, MARKET_TOOL_SCHEMA, NEWS_TOOL_SCHEMA]

        # ── output directory ─────────────────────────────────────────────────
        self.briefs_dir = Path(__file__).parent / 'outputs' / 'briefs'
        self.briefs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DiffSTOCKAdvisor initialized")

    # ─────────────────────────────────────────────────────────────────────────
    #  Tool execution dispatcher
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool call and return JSON result string."""
        try:
            if tool_name == 'get_model_predictions':
                if self._model_tool is None:
                    return json.dumps({
                        'error': 'Model not loaded — simulator was not provided',
                        'status': 'unavailable',
                    })
                result = self._model_tool.get_predictions(
                    as_of_date=tool_input.get('as_of_date', '2024-10-14'),
                    top_k=tool_input.get('top_k', 20),
                    include_uncertainty=tool_input.get('include_uncertainty', True),
                )

            elif tool_name == 'get_market_data':
                result = get_market_data(
                    symbols=tool_input.get('symbols', []),
                    lookback_days=tool_input.get('lookback_days', 5),
                )

            elif tool_name == 'get_stock_news':
                result = get_stock_news(
                    symbols=tool_input.get('symbols', []),
                    lookback_hours=tool_input.get('lookback_hours', 48),
                )

            else:
                result = {'error': f'Unknown tool: {tool_name}'}

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({'error': str(e), 'tool': tool_name})

    # ─────────────────────────────────────────────────────────────────────────
    #  Claude conversation loop with tool use
    # ─────────────────────────────────────────────────────────────────────────

    def _run_conversation(self, user_message: str, max_rounds: int = 5) -> str:
        """
        Run a conversation with Claude that may involve multiple tool calls.

        Args:
            user_message: initial user prompt
            max_rounds: max tool-use rounds to prevent infinite loops

        Returns:
            Final text response from Claude
        """
        messages = [{"role": "user", "content": user_message}]

        for round_i in range(max_rounds):
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            # Check if Claude wants to use tools
            if response.stop_reason == 'tool_use':
                # Build assistant message with all content blocks
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == 'text':
                        assistant_content.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == 'tool_use':
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        # Execute tool
                        logger.info(f"  Tool call: {block.name}({json.dumps(block.input)[:100]}…)")
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude finished — extract text
                text_parts = [
                    block.text for block in response.content
                    if block.type == 'text'
                ]
                return '\n'.join(text_parts)

        return "(Max tool-use rounds reached. Partial response above.)"

    # ═══════════════════════════════════════════════════════════════════════════
    #  Public Assessment Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_weekly_brief(self, as_of_date: str) -> str:
        """
        Generate a full weekly investment brief.

        Covers:
            - Top-10 model picks with news context
            - 3 stocks where model and news conflict
            - Market regime assessment
            - Plain-English summary

        Args:
            as_of_date: date string (YYYY-MM-DD), typically a Monday

        Returns:
            Markdown-formatted brief
        """
        prompt = (
            f"Generate a comprehensive weekly investment brief for the week of {as_of_date}.\n\n"
            f"Please:\n"
            f"1. Call get_model_predictions for {as_of_date} with top_k=20\n"
            f"2. Call get_market_data for the top-10 predicted stocks\n"
            f"3. Call get_stock_news for those same stocks\n\n"
            f"Then produce a brief covering:\n"
            f"- Top-10 model picks with supporting news context\n"
            f"- 3 stocks where model and news signals conflict (risk flags)\n"
            f"- Overall market regime assessment (trending, volatile, or range-bound)\n"
            f"- A one-paragraph summary for a non-technical reader\n"
            f"- Risk section\n\n"
            f"Format as clean Markdown with headers."
        )

        logger.info(f"Generating weekly brief for {as_of_date} …")
        brief = self._run_conversation(prompt)

        # Save
        filename = f"{as_of_date}_weekly_brief.md"
        filepath = self.briefs_dir / filename
        with open(filepath, 'w') as f:
            f.write(brief)
        logger.info(f"Brief saved to {filepath}")

        return brief

    def assess_single_stock(self, symbol: str, as_of_date: Optional[str] = None) -> str:
        """
        Deep dive assessment on a single NIFTY500 stock.

        Args:
            symbol: NSE stock symbol (e.g. 'RELIANCE')
            as_of_date: optional date for model predictions

        Returns:
            Markdown-formatted assessment
        """
        date_str = as_of_date or datetime.date.today().isoformat()

        prompt = (
            f"Provide a deep-dive assessment of {symbol} as of {date_str}.\n\n"
            f"1. Call get_model_predictions for {date_str} to see the model's view\n"
            f"2. Call get_market_data for ['{symbol}'] with lookback_days=10\n"
            f"3. Call get_stock_news for ['{symbol}'] with lookback_hours=72\n\n"
            f"Cover:\n"
            f"- Model's view: bullish/bearish/neutral, confidence level, predicted return\n"
            f"- Recent price action vs model prediction\n"
            f"- News sentiment over last 48-72 hours\n"
            f"- Key risks specific to this stock this week\n"
            f"- How this stock ranks relative to the rest of the universe\n\n"
            f"Be specific with numbers."
        )

        logger.info(f"Assessing {symbol} as of {date_str} …")
        return self._run_conversation(prompt)

    def flag_conflicts(self, as_of_date: Optional[str] = None) -> str:
        """
        Weekly scan for signal conflicts between model and news.

        Identifies:
            - Stocks where model predicts top-20 but news is negative
            - Stocks where model predicts bottom-20 but news is positive
            - High-impact events on predicted stocks

        Returns:
            Markdown-formatted conflict report
        """
        date_str = as_of_date or datetime.date.today().isoformat()

        prompt = (
            f"Generate a signal conflict report as of {date_str}.\n\n"
            f"1. Call get_model_predictions for {date_str} with top_k=30\n"
            f"2. Call get_stock_news for the top-20 AND bottom-10 stocks\n"
            f"3. Call get_market_data for any stocks with sentiment conflicts\n\n"
            f"Identify and explain:\n"
            f"- Stocks where model is bullish (top-20) but news sentiment is negative\n"
            f"- Stocks where model is bearish (bottom-ranked) but news is positive\n"
            f"- Any stocks with high-impact events (earnings, regulatory, promoter)\n\n"
            f"These are the positions requiring human judgment before trading.\n"
            f"Rank conflicts by severity and explain why each is concerning."
        )

        logger.info(f"Scanning for conflicts as of {date_str} …")
        return self._run_conversation(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='DiffSTOCK AI Investment Advisor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.advisor.advisor --mode weekly --date 2024-10-14
  python -m src.advisor.advisor --mode stock --symbol RELIANCE
  python -m src.advisor.advisor --mode conflicts --date 2024-10-14
        """,
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['weekly', 'stock', 'conflicts'],
                        help='Assessment mode')
    parser.add_argument('--date', type=str, default=None,
                        help='As-of date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Stock symbol for single-stock mode')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--no-model', action='store_true',
                        help='Run without loading the DiffSTOCK model (uses stubs)')
    parser.add_argument('--claude-model', type=str, default='claude-sonnet-4-20250514',
                        help='Claude model to use')

    args = parser.parse_args()

    # Resolve date
    as_of_date = args.date or datetime.date.today().isoformat()

    # Load simulator if requested
    simulator = None
    if not args.no_model:
        try:
            from src.simulation.simulator import DiffSTOCKSimulator
            simulator = DiffSTOCKSimulator(
                checkpoint=args.checkpoint,
                config=args.config,
            )
        except Exception as e:
            logger.warning(f"Could not load simulator: {e}")
            logger.warning("Running without model predictions (--no-model)")

    # Create advisor
    advisor = DiffSTOCKAdvisor(
        simulator=simulator,
        model_name=args.claude_model,
    )

    # Run requested mode
    if args.mode == 'weekly':
        result = advisor.generate_weekly_brief(as_of_date)
        print(result)

    elif args.mode == 'stock':
        if not args.symbol:
            print("Error: --symbol is required for stock mode")
            sys.exit(1)
        result = advisor.assess_single_stock(args.symbol, as_of_date)
        print(result)

    elif args.mode == 'conflicts':
        result = advisor.flag_conflicts(as_of_date)
        print(result)


if __name__ == '__main__':
    main()
