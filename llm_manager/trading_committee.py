"""
Enhanced AutoGen Multi-Agent Trading Committee
Upgrade from dual-LLM to sophisticated trading committee
"""

import asyncio
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

from core.events import EventLog
from llm_manager.llm_providers import OllamaProvider

class TradingCommittee:
    """
    Multi-agent trading committee using AutoGen
    Replaces dual-LLM with sophisticated agent collaboration
    """
    
    def __init__(self):
        self.agents = {}
        self.group_chat = None
        self.model_client = None
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize trading committee agents"""
        if not AUTOGEN_AVAILABLE:
            EventLog.emit("WARNING", "AutoGen not available, using fallback")
            return
            
        try:
            # Market Analyst Agent
            self.agents['market_analyst'] = AssistantAgent(
                name="MarketAnalyst",
                model_client=self._get_model_client(),
                system_message="""You are a seasoned market analyst. Analyze price trends, 
                volume patterns, and market conditions. Focus on data-driven insights 
                and provide clear BUY/SELL/HOLD recommendations with confidence levels."""
            )
            
            # Risk Manager Agent
            self.agents['risk_manager'] = AssistantAgent(
                name="RiskManager", 
                model_client=self._get_model_client(),
                system_message="""You are a conservative risk manager. Evaluate position 
                sizing, stop-loss levels, and portfolio exposure. Challenge aggressive 
                positions and ensure proper risk management protocols."""
            )
            
            # Technical Analyst Agent
            self.agents['technical_analyst'] = AssistantAgent(
                name="TechnicalAnalyst",
                model_client=self._get_model_client(), 
                system_message="""You are a technical analysis expert. Focus on chart 
                patterns, indicators (RSI, MACD, SMA), support/resistance levels. 
                Provide entry/exit points based on technical signals."""
            )
            
            # Sentiment Analyst Agent  
            self.agents['sentiment_analyst'] = AssistantAgent(
                name="SentimentAnalyst",
                model_client=self._get_model_client(),
                system_message="""You are a market sentiment specialist. Analyze news, 
                social media, and market psychology. Assess how sentiment impacts 
                price movements and trading decisions."""
            )
            
            EventLog.emit("INFO", f"Initialized {len(self.agents)} trading committee agents")
            
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to initialize trading committee: {e}")
            
    def _get_model_client(self):
        """Get model client for agents"""
        # Use Ollama as primary model client
        ollama = OllamaProvider()
        ollama.initialize()
        return ollama
        
    async def analyze_trade_decision(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-agent collaborative trading decision
        """
        if not AUTOGEN_AVAILABLE or not self.agents:
            return self._fallback_analysis(ticker, market_data)
            
        try:
            # Create trading discussion prompt
            analysis_prompt = f"""
            Trading Analysis Request for {ticker}
            
            Market Data:
            - Current Price: ${market_data.get('price', 'N/A')}
            - Volume: {market_data.get('volume', 'N/A')}
            - SMA_20: {market_data.get('sma_20', 'N/A')}
            - SMA_50: {market_data.get('sma_50', 'N/A')}
            - News Sentiment: {market_data.get('sentiment', 'N/A')}
            
            Committee members, please provide your analysis and recommendation:
            1. Market context and overall outlook
            2. Technical signals and chart patterns
            3. Risk assessment and position sizing
            4. Sentiment and external factors
            
            Reach consensus on: BUY/SELL/HOLD with confidence level (1-10)
            """
            
            # Create group chat for committee discussion
            group_chat = RoundRobinGroupChat(
                participants=list(self.agents.values()),
                termination_condition=TextMentionTermination("CONSENSUS_REACHED")
            )
            
            # Run committee discussion
            result = await group_chat.run(analysis_prompt)
            
            # Parse committee decision
            decision = self._parse_committee_decision(result)
            
            EventLog.emit("INFO", f"Trading committee reached decision for {ticker}: {decision['recommendation']}")
            
            return {
                'ticker': ticker,
                'recommendation': decision['recommendation'],
                'confidence': decision['confidence'],
                'committee_analysis': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            EventLog.emit("ERROR", f"Committee analysis failed: {e}")
            return self._fallback_analysis(ticker, market_data)
            
    def _parse_committee_decision(self, committee_discussion: str) -> Dict[str, Any]:
        """Parse committee discussion for final decision"""
        # Simple parsing logic - can be enhanced with NLP
        discussion_lower = committee_discussion.lower()
        
        # Count recommendations
        buy_count = discussion_lower.count('buy')
        sell_count = discussion_lower.count('sell') 
        hold_count = discussion_lower.count('hold')
        
        # Determine consensus
        if buy_count > sell_count and buy_count > hold_count:
            recommendation = "BUY"
            confidence = min(8, 5 + buy_count)
        elif sell_count > buy_count and sell_count > hold_count:
            recommendation = "SELL" 
            confidence = min(8, 5 + sell_count)
        else:
            recommendation = "HOLD"
            confidence = 5
            
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'vote_counts': {'buy': buy_count, 'sell': sell_count, 'hold': hold_count}
        }
        
    def _fallback_analysis(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when AutoGen is not available"""
        EventLog.emit("INFO", f"Using fallback analysis for {ticker}")
        
        # Simple fallback logic
        sma_20 = market_data.get('sma_20', 0)
        sma_50 = market_data.get('sma_50', 0)
        sentiment = market_data.get('sentiment', 0)
        
        if sma_20 > sma_50 and sentiment > 0:
            recommendation = "BUY"
            confidence = 6
        elif sma_20 < sma_50 and sentiment < 0:
            recommendation = "SELL"
            confidence = 6
        else:
            recommendation = "HOLD"
            confidence = 5
            
        return {
            'ticker': ticker,
            'recommendation': recommendation,
            'confidence': confidence,
            'committee_analysis': "Fallback analysis used",
            'timestamp': datetime.now().isoformat()
        }
        
    def get_committee_status(self) -> Dict[str, Any]:
        """Get status of trading committee"""
        return {
            'autogen_available': AUTOGEN_AVAILABLE,
            'agents_initialized': len(self.agents),
            'agent_names': list(self.agents.keys()) if self.agents else [],
            'status': 'active' if self.agents else 'fallback'
        }

# Global trading committee instance
trading_committee = TradingCommittee()
