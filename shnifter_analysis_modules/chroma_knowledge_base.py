"""
Chroma Vector Database Integration for Shnifter Trader
Stores and retrieves trading patterns, strategies, and market insights using semantic search.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available, using mock implementation")

logger = logging.getLogger(__name__)

class TradingKnowledgeBase:
    """
    Vector database for storing and retrieving trading knowledge using ChromaDB.
    Enables semantic search of trading patterns, strategies, and market insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = None
        self.collections = {}
        
        # Collection names for different types of trading data
        self.collection_names = {
            "patterns": "trading_patterns",
            "strategies": "trading_strategies", 
            "insights": "market_insights",
            "news": "market_news",
            "analysis": "technical_analysis"
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the ChromaDB client and collections."""
        try:
            if not CHROMA_AVAILABLE:
                logger.warning("ChromaDB not available, using mock mode")
                return await self._initialize_mock_mode()
            
            # Initialize ChromaDB client
            persist_directory = self.config.get("persist_directory", "./chroma_trading_db")
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collections
            await self._setup_collections()
            
            self.initialized = True
            logger.info("ChromaDB Trading Knowledge Base initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    async def _initialize_mock_mode(self) -> bool:
        """Initialize mock mode when ChromaDB is not available."""
        self.mock_data = {
            "patterns": [],
            "strategies": [],
            "insights": [],
            "news": [],
            "analysis": []
        }
        self.initialized = True
        logger.info("Mock Trading Knowledge Base initialized")
        return True
    
    async def _setup_collections(self):
        """Set up ChromaDB collections for different data types."""
        for collection_type, collection_name in self.collection_names.items():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Found existing collection: {collection_name}")
            except:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Trading {collection_type} data"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self.collections[collection_type] = collection
    
    async def store_trading_pattern(self,
                                  pattern_name: str,
                                  description: str,
                                  indicators: Dict[str, Any],
                                  success_rate: float,
                                  examples: List[Dict] = None) -> str:
        """Store a trading pattern in the knowledge base."""
        try:
            pattern_data = {
                "name": pattern_name,
                "description": description,
                "indicators": indicators,
                "success_rate": success_rate,
                "examples": examples or [],
                "timestamp": datetime.now().isoformat(),
                "type": "pattern"
            }
            
            # Create unique ID
            pattern_id = hashlib.md5(f"{pattern_name}_{datetime.now()}".encode()).hexdigest()
            
            if CHROMA_AVAILABLE and self.initialized:
                # Store in ChromaDB
                self.collections["patterns"].add(
                    documents=[json.dumps(pattern_data)],
                    metadatas=[{
                        "name": pattern_name,
                        "success_rate": success_rate,
                        "type": "pattern",
                        "timestamp": pattern_data["timestamp"]
                    }],
                    ids=[pattern_id]
                )
            else:
                # Store in mock data
                pattern_data["id"] = pattern_id
                self.mock_data["patterns"].append(pattern_data)
            
            logger.info(f"Stored trading pattern: {pattern_name}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error storing trading pattern: {e}")
            return ""
    
    async def store_trading_strategy(self,
                                   strategy_name: str,
                                   description: str,
                                   entry_conditions: List[str],
                                   exit_conditions: List[str],
                                   risk_management: Dict[str, Any],
                                   backtest_results: Dict[str, Any] = None) -> str:
        """Store a trading strategy in the knowledge base."""
        try:
            strategy_data = {
                "name": strategy_name,
                "description": description,
                "entry_conditions": entry_conditions,
                "exit_conditions": exit_conditions,
                "risk_management": risk_management,
                "backtest_results": backtest_results or {},
                "timestamp": datetime.now().isoformat(),
                "type": "strategy"
            }
            
            strategy_id = hashlib.md5(f"{strategy_name}_{datetime.now()}".encode()).hexdigest()
            
            if CHROMA_AVAILABLE and self.initialized:
                self.collections["strategies"].add(
                    documents=[json.dumps(strategy_data)],
                    metadatas=[{
                        "name": strategy_name,
                        "type": "strategy",
                        "timestamp": strategy_data["timestamp"],
                        "win_rate": backtest_results.get("win_rate", 0) if backtest_results else 0
                    }],
                    ids=[strategy_id]
                )
            else:
                strategy_data["id"] = strategy_id
                self.mock_data["strategies"].append(strategy_data)
            
            logger.info(f"Stored trading strategy: {strategy_name}")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Error storing trading strategy: {e}")
            return ""
    
    async def store_market_insight(self,
                                 title: str,
                                 content: str,
                                 tickers: List[str],
                                 sentiment: str,
                                 confidence: float,
                                 source: str = "analysis") -> str:
        """Store a market insight in the knowledge base."""
        try:
            insight_data = {
                "title": title,
                "content": content,
                "tickers": tickers,
                "sentiment": sentiment,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "type": "insight"
            }
            
            insight_id = hashlib.md5(f"{title}_{datetime.now()}".encode()).hexdigest()
            
            if CHROMA_AVAILABLE and self.initialized:
                self.collections["insights"].add(
                    documents=[content],
                    metadatas=[{
                        "title": title,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "source": source,
                        "tickers": ",".join(tickers),
                        "timestamp": insight_data["timestamp"],
                        "type": "insight"
                    }],
                    ids=[insight_id]
                )
            else:
                insight_data["id"] = insight_id
                self.mock_data["insights"].append(insight_data)
            
            logger.info(f"Stored market insight: {title}")
            return insight_id
            
        except Exception as e:
            logger.error(f"Error storing market insight: {e}")
            return ""
    
    async def store_autogen_decision(self, 
                                   ticker: str,
                                   autogen_result: Dict[str, Any]) -> str:
        """Store AutoGen multi-agent trading decision in knowledge base."""
        try:
            # Extract key information from AutoGen result
            consensus_decision = autogen_result.get("consensus_decision", "HOLD")
            consensus_confidence = autogen_result.get("consensus_confidence", 0.0)
            individual_decisions = autogen_result.get("individual_decisions", {})
            
            # Create comprehensive content for storage
            decision_content = f"""
            AutoGen Multi-Agent Trading Decision for {ticker}
            
            Consensus: {consensus_decision} (Confidence: {consensus_confidence:.2%})
            
            Individual Agent Opinions:
            """
            
            for agent, decision_data in individual_decisions.items():
                decision_content += f"""
            - {agent}: {decision_data.get('decision', 'N/A')} 
              (Confidence: {decision_data.get('confidence', 0):.2%})
              Reasoning: {decision_data.get('reasoning', 'N/A')[:100]}...
            """
            
            decision_content += f"""
            
            Discussion Summary: {autogen_result.get('discussion_summary', 'N/A')}
            Consensus Reasoning: {autogen_result.get('consensus_reasoning', 'N/A')}
            """
            
            # Store as market insight
            insight_id = await self.store_market_insight(
                title=f"AutoGen Decision: {ticker} - {consensus_decision}",
                content=decision_content,
                tickers=[ticker],
                sentiment="bullish" if consensus_decision == "BUY" else "bearish" if consensus_decision == "SELL" else "neutral",
                confidence=consensus_confidence,
                source="autogen_agents"
            )
            
            logger.info(f"Stored AutoGen decision for {ticker}: {consensus_decision}")
            return insight_id
            
        except Exception as e:
            logger.error(f"Error storing AutoGen decision: {e}")
            return ""
    
    async def search_patterns(self, 
                            query: str, 
                            n_results: int = 5,
                            min_success_rate: float = 0.0) -> List[Dict[str, Any]]:
        """Search for trading patterns using semantic search."""
        try:
            if CHROMA_AVAILABLE and self.initialized:
                # Search using ChromaDB
                results = self.collections["patterns"].query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"success_rate": {"$gte": min_success_rate}} if min_success_rate > 0 else None
                )
                
                # Parse results
                patterns = []
                for i, doc in enumerate(results["documents"][0]):
                    pattern_data = json.loads(doc)
                    pattern_data["similarity_score"] = 1.0 - results["distances"][0][i]
                    patterns.append(pattern_data)
                
                return patterns
            else:
                # Mock search
                patterns = []
                for pattern in self.mock_data["patterns"]:
                    if (query.lower() in pattern["name"].lower() or 
                        query.lower() in pattern["description"].lower()):
                        if pattern["success_rate"] >= min_success_rate:
                            patterns.append(pattern)
                
                return patterns[:n_results]
                
        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            return []
    
    async def search_strategies(self,
                              query: str,
                              n_results: int = 5,
                              strategy_type: str = None) -> List[Dict[str, Any]]:
        """Search for trading strategies using semantic search."""
        try:
            if CHROMA_AVAILABLE and self.initialized:
                where_clause = None
                if strategy_type:
                    where_clause = {"type": strategy_type}
                
                results = self.collections["strategies"].query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause
                )
                
                strategies = []
                for i, doc in enumerate(results["documents"][0]):
                    strategy_data = json.loads(doc)
                    strategy_data["similarity_score"] = 1.0 - results["distances"][0][i]
                    strategies.append(strategy_data)
                
                return strategies
            else:
                # Mock search
                strategies = []
                for strategy in self.mock_data["strategies"]:
                    if (query.lower() in strategy["name"].lower() or
                        query.lower() in strategy["description"].lower()):
                        strategies.append(strategy)
                
                return strategies[:n_results]
                
        except Exception as e:
            logger.error(f"Error searching strategies: {e}")
            return []
    
    async def search_insights(self,
                            query: str,
                            ticker: str = None,
                            sentiment: str = None,
                            n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for market insights using semantic search."""
        try:
            if CHROMA_AVAILABLE and self.initialized:
                where_clause = {}
                if ticker:
                    where_clause["tickers"] = {"$contains": ticker}
                if sentiment:
                    where_clause["sentiment"] = sentiment
                
                results = self.collections["insights"].query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )
                
                insights = []
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    insight = {
                        "title": metadata["title"],
                        "content": doc,
                        "sentiment": metadata["sentiment"],
                        "confidence": metadata["confidence"],
                        "source": metadata["source"],
                        "tickers": metadata["tickers"].split(","),
                        "timestamp": metadata["timestamp"],
                        "similarity_score": 1.0 - results["distances"][0][i]
                    }
                    insights.append(insight)
                
                return insights
            else:
                # Mock search
                insights = []
                for insight in self.mock_data["insights"]:
                    match = query.lower() in insight["content"].lower()
                    if ticker:
                        match = match and ticker.upper() in [t.upper() for t in insight["tickers"]]
                    if sentiment:
                        match = match and insight["sentiment"].lower() == sentiment.lower()
                    
                    if match:
                        insights.append(insight)
                
                return insights[:n_results]
                
        except Exception as e:
            logger.error(f"Error searching insights: {e}")
            return []
    
    async def get_similar_market_conditions(self,
                                          current_conditions: Dict[str, Any],
                                          n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar historical market conditions."""
        try:
            # Create query from current conditions
            query_parts = []
            for key, value in current_conditions.items():
                query_parts.append(f"{key}: {value}")
            
            query = " ".join(query_parts)
            
            # Search across insights and analysis
            insights = await self.search_insights(query, n_results=n_results)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error finding similar conditions: {e}")
            return []
    
    async def store_analysis_result(self,
                                  ticker: str,
                                  analysis_type: str,
                                  result: Dict[str, Any],
                                  confidence: float) -> str:
        """Store analysis results for future reference."""
        try:
            analysis_data = {
                "ticker": ticker,
                "analysis_type": analysis_type,
                "result": result,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "type": "analysis"
            }
            
            analysis_id = hashlib.md5(f"{ticker}_{analysis_type}_{datetime.now()}".encode()).hexdigest()
            
            # Create searchable content
            content = f"{analysis_type} analysis for {ticker}: {json.dumps(result)}"
            
            if CHROMA_AVAILABLE and self.initialized:
                self.collections["analysis"].add(
                    documents=[content],
                    metadatas=[{
                        "ticker": ticker,
                        "analysis_type": analysis_type,
                        "confidence": confidence,
                        "timestamp": analysis_data["timestamp"],
                        "type": "analysis"
                    }],
                    ids=[analysis_id]
                )
            else:
                analysis_data["id"] = analysis_id
                analysis_data["content"] = content
                self.mock_data["analysis"].append(analysis_data)
            
            logger.info(f"Stored analysis result for {ticker}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
            return ""
    
    async def get_historical_analysis(self,
                                    ticker: str,
                                    analysis_type: str = None,
                                    days_back: int = 30) -> List[Dict[str, Any]]:
        """Get historical analysis results for a ticker."""
        try:
            # Calculate date threshold
            from datetime import timedelta
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            if CHROMA_AVAILABLE and self.initialized:
                where_clause = {
                    "ticker": ticker,
                    "timestamp": {"$gte": threshold_date}
                }
                if analysis_type:
                    where_clause["analysis_type"] = analysis_type
                
                results = self.collections["analysis"].query(
                    query_texts=[f"analysis for {ticker}"],
                    where=where_clause,
                    n_results=50
                )
                
                analyses = []
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    analysis = {
                        "ticker": metadata["ticker"],
                        "analysis_type": metadata["analysis_type"],
                        "confidence": metadata["confidence"],
                        "timestamp": metadata["timestamp"],
                        "content": doc
                    }
                    analyses.append(analysis)
                
                return analyses
            else:
                # Mock historical analysis
                analyses = []
                for analysis in self.mock_data["analysis"]:
                    if (analysis["ticker"].upper() == ticker.upper() and
                        analysis["timestamp"] >= threshold_date):
                        if not analysis_type or analysis["analysis_type"] == analysis_type:
                            analyses.append(analysis)
                
                return analyses
                
        except Exception as e:
            logger.error(f"Error getting historical analysis: {e}")
            return []
    
    async def get_similar_autogen_decisions(self,
                                          ticker: str,
                                          decision_type: str = None,
                                          days_back: int = 30) -> List[Dict[str, Any]]:
        """Get similar AutoGen decisions for pattern analysis."""
        try:
            # Search for AutoGen decisions
            query = f"AutoGen decision {ticker}"
            if decision_type:
                query += f" {decision_type}"
            
            insights = await self.search_insights(
                query=query,
                ticker=ticker,
                n_results=10
            )
            
            # Filter for AutoGen decisions
            autogen_decisions = []
            for insight in insights:
                if insight.get("source") == "autogen_agents":
                    # Check if within time range
                    from datetime import datetime, timedelta
                    insight_date = datetime.fromisoformat(insight["timestamp"].replace("Z", "+00:00"))
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    
                    if insight_date >= cutoff_date:
                        autogen_decisions.append(insight)
            
            return autogen_decisions
            
        except Exception as e:
            logger.error(f"Error getting similar AutoGen decisions: {e}")
            return []
    
    async def analyze_autogen_decision_accuracy(self, ticker: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze the historical accuracy of AutoGen decisions."""
        try:
            decisions = await self.get_similar_autogen_decisions(ticker, days_back=days_back)
            
            if not decisions:
                return {"error": "No historical decisions found"}
            
            # Simple accuracy analysis (would need actual price data for real accuracy)
            total_decisions = len(decisions)
            buy_decisions = sum(1 for d in decisions if "BUY" in d.get("title", ""))
            sell_decisions = sum(1 for d in decisions if "SELL" in d.get("title", ""))
            hold_decisions = total_decisions - buy_decisions - sell_decisions
            
            avg_confidence = sum(d.get("confidence", 0) for d in decisions) / total_decisions
            
            return {
                "total_decisions": total_decisions,
                "decision_breakdown": {
                    "BUY": buy_decisions,
                    "SELL": sell_decisions,
                    "HOLD": hold_decisions
                },
                "average_confidence": avg_confidence,
                "most_recent": decisions[0] if decisions else None,
                "analysis_period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Error analyzing AutoGen accuracy: {e}")
            return {"error": str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        try:
            stats = {}
            
            if CHROMA_AVAILABLE and self.initialized:
                for collection_type, collection in self.collections.items():
                    stats[collection_type] = collection.count()
            else:
                for collection_type, data in self.mock_data.items():
                    stats[collection_type] = len(data)
            
            stats["total_documents"] = sum(stats.values())
            stats["chroma_available"] = CHROMA_AVAILABLE
            stats["initialized"] = self.initialized
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


# Example usage and integration
async def main():
    """Example of how to use the Trading Knowledge Base."""
    
    # Initialize the knowledge base
    kb = TradingKnowledgeBase({
        "persist_directory": "./chroma_trading_db"
    })
    
    await kb.initialize()
    
    # Store some example trading patterns
    await kb.store_trading_pattern(
        pattern_name="Bullish Divergence",
        description="Price makes lower lows while RSI makes higher lows, indicating potential upward reversal",
        indicators={"RSI": "divergence", "price": "lower_lows"},
        success_rate=0.72,
        examples=[{"ticker": "AAPL", "date": "2024-01-15", "outcome": "success"}]
    )
    
    # Store a trading strategy
    await kb.store_trading_strategy(
        strategy_name="Mean Reversion with RSI",
        description="Buy when RSI < 30, sell when RSI > 70",
        entry_conditions=["RSI < 30", "Price below 20-day SMA"],
        exit_conditions=["RSI > 70", "5% profit target", "3% stop loss"],
        risk_management={"position_size": "2%", "stop_loss": "3%", "take_profit": "5%"},
        backtest_results={"win_rate": 0.65, "avg_return": 0.038, "max_drawdown": 0.12}
    )
    
    # Store market insights
    await kb.store_market_insight(
        title="Tech Sector Momentum",
        content="Technology sector showing strong momentum with AI and cloud companies leading gains. Expect continued outperformance.",
        tickers=["AAPL", "MSFT", "GOOGL", "NVDA"],
        sentiment="bullish",
        confidence=0.8,
        source="technical_analysis"
    )
    
    # Search for patterns
    patterns = await kb.search_patterns("RSI divergence", n_results=3)
    print(f"Found {len(patterns)} RSI divergence patterns")
    
    # Search for strategies
    strategies = await kb.search_strategies("mean reversion", n_results=3)
    print(f"Found {len(strategies)} mean reversion strategies")
    
    # Search for insights about specific ticker
    insights = await kb.search_insights("momentum", ticker="AAPL", n_results=5)
    print(f"Found {len(insights)} insights about AAPL momentum")
    
    # Get collection stats
    stats = kb.get_collection_stats()
    print(f"Knowledge base stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
