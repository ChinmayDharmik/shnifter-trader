"""
AutoGen Multi-Agent Trading System for Shnifter Trader
Creates multiple AI agents that collaborate on trading decisions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logging.warning("AutoGen not available, using mock implementation")

logger = logging.getLogger(__name__)

class TradingAgentSystem:
    """
    Multi-agent trading system using AutoGen framework.
    Creates specialized agents for different aspects of trading analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents = {}
        self.group_chat = None
        self.model_client = None
        self.active_discussions = {}
        self.trading_consensus = {}
        
        # Agent configurations
        self.agent_configs = {
            "technical_analyst": {
                "name": "TechnicalAnalyst",
                "system_message": """You are a Technical Analysis Expert specializing in chart patterns, 
                indicators, and market sentiment. You analyze price movements, volume, RSI, MACD, moving averages, 
                and other technical indicators to make informed trading recommendations. You are data-driven and 
                focus on quantitative analysis."""
            },
            "fundamental_analyst": {
                "name": "FundamentalAnalyst", 
                "system_message": """You are a Fundamental Analysis Expert who evaluates companies based on 
                financial statements, earnings, revenue, market conditions, and economic factors. You analyze 
                business fundamentals, competitive advantages, and long-term value propositions. You think 
                strategically about intrinsic value."""
            },
            "risk_manager": {
                "name": "RiskManager",
                "system_message": """You are a Risk Management Specialist focused on protecting capital and 
                managing portfolio risk. You evaluate position sizing, stop-losses, diversification, correlation, 
                and maximum drawdown. Your primary concern is capital preservation and risk-adjusted returns. 
                You are conservative and methodical."""
            },
            "market_sentiment": {
                "name": "SentimentAnalyst",
                "system_message": """You are a Market Sentiment and News Analysis Expert. You analyze market 
                sentiment, news events, social media trends, institutional flows, and market psychology. You 
                understand how emotions and narratives drive market movements in the short term."""
            },
            "trading_coordinator": {
                "name": "TradingCoordinator",
                "system_message": """You are the Trading Coordinator who synthesizes analysis from all team 
                members to make final trading decisions. You weigh different perspectives, resolve conflicts, 
                and create actionable trading plans with specific entry/exit points, position sizes, and risk 
                parameters. You are decisive and practical."""
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the multi-agent trading system."""
        try:
            if not AUTOGEN_AVAILABLE:
                logger.warning("AutoGen not available, using mock mode")
                return await self._initialize_mock_mode()
            
            # Initialize OpenAI model client
            api_key = self.config.get("openai_api_key", "")
            if not api_key:
                logger.error("OpenAI API key required for AutoGen agents")
                return False
            
            self.model_client = OpenAIChatCompletionClient(
                model=self.config.get("model", "gpt-4o"),
                api_key=api_key
            )
            
            # Create specialized trading agents
            await self._create_trading_agents()
            
            # Set up group chat for agent collaboration
            await self._setup_group_chat()
            
            logger.info("AutoGen Trading Agent System initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen system: {e}")
            return False
    
    async def _initialize_mock_mode(self) -> bool:
        """Initialize mock mode when AutoGen is not available."""
        for agent_id, config in self.agent_configs.items():
            self.agents[agent_id] = MockTradingAgent(config["name"], config["system_message"])
        
        logger.info("Mock Trading Agent System initialized")
        return True
    
    async def _create_trading_agents(self):
        """Create the specialized trading agents."""
        for agent_id, config in self.agent_configs.items():
            agent = AssistantAgent(
                name=config["name"],
                model_client=self.model_client,
                system_message=config["system_message"]
            )
            self.agents[agent_id] = agent
            logger.info(f"Created agent: {config['name']}")
    
    async def _setup_group_chat(self):
        """Set up group chat for agent collaboration."""
        if not AUTOGEN_AVAILABLE:
            return
        
        agent_list = list(self.agents.values())
        
        # Add termination condition
        termination = TextMentionTermination("TRADING_DECISION_COMPLETE")
        
        # Create round-robin group chat
        self.group_chat = RoundRobinGroupChat(
            participants=agent_list,
            termination_condition=termination,
            max_turns=10  # Limit discussion length
        )
        
        logger.info("Group chat setup complete")
    
    async def analyze_trading_opportunity(self, 
                                        ticker: str, 
                                        market_data: Dict[str, Any],
                                        news_data: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a trading opportunity using multi-agent collaboration.
        
        Args:
            ticker: Stock/crypto ticker symbol
            market_data: Current market data (price, volume, indicators, etc.)
            news_data: Recent news articles or sentiment data
            
        Returns:
            Dict containing the collaborative analysis and recommendation
        """
        try:
            # Prepare analysis context
            context = self._prepare_analysis_context(ticker, market_data, news_data)
            
            if AUTOGEN_AVAILABLE:
                return await self._run_autogen_analysis(context)
            else:
                return await self._run_mock_analysis(context)
                
        except Exception as e:
            logger.error(f"Error in trading analysis: {e}")
            return {"error": str(e), "recommendation": "HOLD", "confidence": 0.0}
    
    def _prepare_analysis_context(self, ticker: str, market_data: Dict, news_data: List = None) -> str:
        """Prepare the analysis context for the agents."""
        context = f"""
TRADING ANALYSIS REQUEST for {ticker}

MARKET DATA:
- Current Price: ${market_data.get('current_price', 'N/A')}
- 24h Change: {market_data.get('change_24h', 'N/A')}%
- Volume: {market_data.get('volume', 'N/A')}
- RSI: {market_data.get('rsi', 'N/A')}
- MACD: {market_data.get('macd', 'N/A')}
- Moving Averages: {market_data.get('moving_averages', {})}
- Support/Resistance: {market_data.get('support_resistance', {})}

NEWS & SENTIMENT:
{chr(10).join(news_data) if news_data else "No recent news available"}

PLEASE ANALYZE THIS TRADING OPPORTUNITY:
1. Technical Analyst: Provide technical analysis based on charts and indicators
2. Fundamental Analyst: Evaluate the fundamental factors and long-term outlook  
3. Sentiment Analyst: Assess market sentiment and news impact
4. Risk Manager: Evaluate risks and suggest position sizing/stop-losses
5. Trading Coordinator: Synthesize all analysis into a final recommendation

Please provide:
- BUY/SELL/HOLD recommendation
- Confidence level (0-100%)
- Entry price target
- Stop-loss level
- Take-profit target
- Position size recommendation
- Key risks and considerations

End your final recommendation with: TRADING_DECISION_COMPLETE
"""
        return context
    
    async def _run_autogen_analysis(self, context: str) -> Dict[str, Any]:
        """Run analysis using AutoGen agents."""
        try:
            # Start the group chat discussion
            chat_result = await self.group_chat.run(
                task=context,
                cancellation_token=None
            )
            
            # Extract the trading decision from the chat
            decision = self._extract_trading_decision(chat_result.messages)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in AutoGen analysis: {e}")
            return {"error": str(e), "recommendation": "HOLD", "confidence": 0.0}
    
    async def _run_mock_analysis(self, context: str) -> Dict[str, Any]:
        """Run mock analysis when AutoGen is not available."""
        try:
            analyses = {}
            
            # Get analysis from each mock agent
            for agent_id, agent in self.agents.items():
                analysis = await agent.analyze(context)
                analyses[agent_id] = analysis
            
            # Synthesize the analyses
            decision = self._synthesize_mock_analyses(analyses)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in mock analysis: {e}")
            return {"error": str(e), "recommendation": "HOLD", "confidence": 0.0}
    
    def _extract_trading_decision(self, messages: List[Any]) -> Dict[str, Any]:
        """Extract trading decision from agent chat messages."""
        try:
            # Find the final coordinator message
            decision_text = ""
            for message in reversed(messages):
                if hasattr(message, 'content') and 'TRADING_DECISION_COMPLETE' in message.content:
                    decision_text = message.content
                    break
            
            # Parse the decision (simplified - would use more sophisticated parsing in production)
            decision = {
                "recommendation": "HOLD",
                "confidence": 50.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 1.0,
                "reasoning": decision_text,
                "timestamp": datetime.now().isoformat(),
                "agent_consensus": True
            }
            
            # Extract recommendation
            if "BUY" in decision_text.upper():
                decision["recommendation"] = "BUY"
            elif "SELL" in decision_text.upper():
                decision["recommendation"] = "SELL"
            
            # Extract confidence (look for percentage)
            import re
            confidence_match = re.search(r'confidence[:\s]*(\d+)%?', decision_text, re.IGNORECASE)
            if confidence_match:
                decision["confidence"] = float(confidence_match.group(1))
            
            return decision
            
        except Exception as e:
            logger.error(f"Error extracting decision: {e}")
            return {"error": str(e), "recommendation": "HOLD", "confidence": 0.0}
    
    def _synthesize_mock_analyses(self, analyses: Dict[str, str]) -> Dict[str, Any]:
        """Synthesize mock analyses into a trading decision."""
        # Simple mock synthesis logic
        buy_votes = sum(1 for analysis in analyses.values() if "BUY" in analysis.upper())
        sell_votes = sum(1 for analysis in analyses.values() if "SELL" in analysis.upper())
        
        if buy_votes > sell_votes:
            recommendation = "BUY"
            confidence = 60.0 + (buy_votes * 10)
        elif sell_votes > buy_votes:
            recommendation = "SELL"
            confidence = 60.0 + (sell_votes * 10)
        else:
            recommendation = "HOLD"
            confidence = 50.0
        
        return {
            "recommendation": recommendation,
            "confidence": min(confidence, 95.0),
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 1.0,
            "reasoning": f"Consensus from {len(analyses)} agents: {buy_votes} BUY, {sell_votes} SELL",
            "timestamp": datetime.now().isoformat(),
            "agent_consensus": True,
            "individual_analyses": analyses
        }
    
    async def get_risk_assessment(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk assessment from the risk manager agent."""
        try:
            risk_context = f"""
PORTFOLIO RISK ASSESSMENT REQUEST

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2)}

Please provide a comprehensive risk assessment including:
- Overall portfolio risk level (Low/Medium/High)
- Diversification analysis
- Correlation risks
- Maximum drawdown potential
- Recommended position adjustments
- Risk mitigation strategies
"""
            
            if AUTOGEN_AVAILABLE and "risk_manager" in self.agents:
                # Use the actual risk manager agent
                response = await self.agents["risk_manager"].run(task=risk_context)
                return {"assessment": response.messages[-1].content}
            else:
                # Use mock risk assessment
                return await self.agents["risk_manager"].analyze(risk_context)
                
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all trading agents."""
        return {
            "agents_active": len(self.agents),
            "autogen_available": AUTOGEN_AVAILABLE,
            "model_client": str(type(self.model_client).__name__) if self.model_client else "None",
            "agent_list": list(self.agents.keys())
        }


class MockTradingAgent:
    """Mock trading agent for when AutoGen is not available."""
    
    def __init__(self, name: str, system_message: str):
        self.name = name
        self.system_message = system_message
    
    async def analyze(self, context: str) -> str:
        """Provide mock analysis based on agent specialty."""
        if "TechnicalAnalyst" in self.name:
            return "Based on technical indicators, I see bullish momentum with RSI showing oversold conditions. Recommend BUY with targets above current resistance."
        elif "FundamentalAnalyst" in self.name:
            return "Fundamentals look strong with solid earnings growth and expanding market share. Long-term outlook positive. Recommend BUY."
        elif "RiskManager" in self.name:
            return "Current risk levels are manageable. Recommend position size of 2% with stop-loss at -5%. Monitor correlation risks."
        elif "SentimentAnalyst" in self.name:
            return "Market sentiment is cautiously optimistic. Social media chatter trending positive. Recommend HOLD pending clearer signals."
        elif "TradingCoordinator" in self.name:
            return "TRADING_DECISION_COMPLETE: Consensus recommendation is BUY with 75% confidence. Entry at current levels, stop-loss -5%, target +15%."
        else:
            return "Analysis complete. Recommend HOLD pending further evaluation."


# Example usage
async def main():
    """Example of how to use the AutoGen Trading System."""
    
    # Initialize the system
    trading_system = TradingAgentSystem({
        "openai_api_key": "your-api-key-here",  # Set this to use real AutoGen
        "model": "gpt-4o"
    })
    
    await trading_system.initialize()
    
    # Prepare mock market data
    market_data = {
        "current_price": 150.25,
        "change_24h": 2.5,
        "volume": 1000000,
        "rsi": 45.0,
        "macd": {"signal": "bullish"},
        "moving_averages": {"sma_20": 148.0, "sma_50": 145.0},
        "support_resistance": {"support": 145.0, "resistance": 155.0}
    }
    
    news_data = [
        "Company reports strong Q4 earnings, beating estimates",
        "New partnership announced with major tech firm",
        "Analyst upgrades rating to Buy with $160 target"
    ]
    
    # Get trading analysis
    result = await trading_system.analyze_trading_opportunity("AAPL", market_data, news_data)
    
    print("Trading Analysis Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
