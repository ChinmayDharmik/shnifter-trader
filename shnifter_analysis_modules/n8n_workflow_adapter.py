"""
N8N Workflow Integration for Shnifter Trader
Adapts n8n automation workflows for crypto monitoring and portfolio tracking.
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import schedule
import time
import threading

logger = logging.getLogger(__name__)

class N8NWorkflowAdapter:
    """
    Adapts n8n workflows for integration with Shnifter Trader.
    Provides crypto monitoring, portfolio tracking, and market alerts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_workflows = {}
        self.market_data_cache = {}
        self.portfolio_data = {}
        self.alert_thresholds = {}
        self.running = False
        
        # Initialize CoinGecko API (free tier)
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # Initialize webhook server for alerts
        self.webhook_port = self.config.get("webhook_port", 8765)
        
    async def initialize(self):
        """Initialize the workflow adapter."""
        try:
            # Load workflow configurations
            await self._load_crypto_workflows()
            
            # Start background monitoring
            self.running = True
            asyncio.create_task(self._start_monitoring_loop())
            
            logger.info("N8N Workflow Adapter initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize N8N adapter: {e}")
            return False
    
    async def _load_crypto_workflows(self):
        """Load and adapt crypto monitoring workflows from n8n."""
        
        # Crypto Price Monitoring Workflow (adapted from 0177_Coingecko_Cron_Update_Scheduled.json)
        self.active_workflows["crypto_monitor"] = {
            "id": "crypto_price_monitor",
            "name": "Crypto Price Monitor",
            "description": "Monitor cryptocurrency prices and update portfolio values",
            "trigger": "cron",
            "schedule": "*/5 * * * *",  # Every 5 minutes
            "nodes": [
                {
                    "type": "coingecko_api",
                    "operation": "get_prices",
                    "symbols": ["bitcoin", "ethereum", "solana", "cardano", "polygon"]
                },
                {
                    "type": "portfolio_update",
                    "operation": "update_values"
                },
                {
                    "type": "alert_check",
                    "operation": "price_alerts"
                }
            ]
        }
        
        # Portfolio Tracking Workflow
        self.active_workflows["portfolio_tracker"] = {
            "id": "portfolio_tracker",
            "name": "Portfolio Value Tracker",
            "description": "Track portfolio performance and generate reports",
            "trigger": "cron", 
            "schedule": "0 * * * *",  # Every hour
            "nodes": [
                {
                    "type": "portfolio_analysis",
                    "operation": "calculate_pnl"
                },
                {
                    "type": "performance_metrics",
                    "operation": "update_stats"
                }
            ]
        }
        
        # Market Alert Workflow  
        self.active_workflows["market_alerts"] = {
            "id": "market_alerts",
            "name": "Market Alert System",
            "description": "Generate alerts based on market conditions",
            "trigger": "webhook",
            "nodes": [
                {
                    "type": "condition_check",
                    "operation": "evaluate_conditions"
                },
                {
                    "type": "notification",
                    "operation": "send_alert"
                }
            ]
        }
        
        logger.info(f"Loaded {len(self.active_workflows)} crypto workflows")
    
    async def _start_monitoring_loop(self):
        """Start the main monitoring loop for all workflows."""
        while self.running:
            try:
                # Execute scheduled workflows
                current_time = datetime.now()
                
                for workflow_id, workflow in self.active_workflows.items():
                    if workflow["trigger"] == "cron":
                        if self._should_execute_workflow(workflow, current_time):
                            await self._execute_workflow(workflow_id, workflow)
                
                # Wait 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _should_execute_workflow(self, workflow: Dict, current_time: datetime) -> bool:
        """Check if a workflow should be executed based on its schedule."""
        schedule_str = workflow.get("schedule", "")
        
        # Simple cron-like scheduling
        if schedule_str == "*/5 * * * *":  # Every 5 minutes
            return current_time.minute % 5 == 0
        elif schedule_str == "0 * * * *":  # Every hour
            return current_time.minute == 0
        
        return False
    
    async def _execute_workflow(self, workflow_id: str, workflow: Dict):
        """Execute a specific workflow."""
        try:
            logger.info(f"Executing workflow: {workflow['name']}")
            
            if workflow_id == "crypto_monitor":
                await self._execute_crypto_monitor()
            elif workflow_id == "portfolio_tracker":
                await self._execute_portfolio_tracker()
            elif workflow_id == "market_alerts":
                await self._execute_market_alerts()
                
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
    
    async def _execute_crypto_monitor(self):
        """Execute crypto price monitoring workflow."""
        try:
            # Get crypto prices from CoinGecko API
            symbols = ["bitcoin", "ethereum", "solana", "cardano", "polygon"]
            prices = await self._fetch_crypto_prices(symbols)
            
            # Update market data cache
            self.market_data_cache.update(prices)
            
            # Check for price alerts
            await self._check_price_alerts(prices)
            
            # Update portfolio values if applicable
            await self._update_portfolio_values(prices)
            
            logger.info(f"Updated prices for {len(prices)} cryptocurrencies")
            
        except Exception as e:
            logger.error(f"Error in crypto monitor: {e}")
    
    async def _fetch_crypto_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch cryptocurrency prices from CoinGecko API."""
        try:
            symbol_str = ",".join(symbols)
            url = f"{self.coingecko_api}/simple/price"
            params = {
                "ids": symbol_str,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format the response
                        prices = {}
                        for symbol, info in data.items():
                            prices[symbol] = {
                                "price": info["usd"],
                                "change_24h": info.get("usd_24h_change", 0),
                                "market_cap": info.get("usd_market_cap", 0),
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        return prices
                    else:
                        logger.error(f"CoinGecko API error: {response.status}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching crypto prices: {e}")
            return {}
    
    async def _check_price_alerts(self, prices: Dict[str, Any]):
        """Check if any price alerts should be triggered."""
        for symbol, data in prices.items():
            if symbol in self.alert_thresholds:
                thresholds = self.alert_thresholds[symbol]
                current_price = data["price"]
                change_24h = data["change_24h"]
                
                # Check price thresholds
                if "price_above" in thresholds and current_price > thresholds["price_above"]:
                    await self._send_alert(f"{symbol.upper()} price alert", 
                                         f"Price ${current_price:,.2f} is above threshold ${thresholds['price_above']:,.2f}")
                
                if "price_below" in thresholds and current_price < thresholds["price_below"]:
                    await self._send_alert(f"{symbol.upper()} price alert",
                                         f"Price ${current_price:,.2f} is below threshold ${thresholds['price_below']:,.2f}")
                
                # Check change thresholds
                if "change_above" in thresholds and change_24h > thresholds["change_above"]:
                    await self._send_alert(f"{symbol.upper()} change alert",
                                         f"24h change {change_24h:.2f}% is above threshold {thresholds['change_above']:.2f}%")
    
    async def _execute_portfolio_tracker(self):
        """Execute portfolio tracking workflow."""
        try:
            # Calculate portfolio performance
            total_value = 0
            total_change = 0
            
            for symbol, holding in self.portfolio_data.items():
                if symbol in self.market_data_cache:
                    current_price = self.market_data_cache[symbol]["price"]
                    value = holding["amount"] * current_price
                    total_value += value
                    
                    # Calculate change from entry price
                    if "entry_price" in holding:
                        change = ((current_price - holding["entry_price"]) / holding["entry_price"]) * 100
                        total_change += change * (value / total_value) if total_value > 0 else 0
            
            # Update portfolio stats
            self.portfolio_data["_stats"] = {
                "total_value": total_value,
                "total_change": total_change,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Portfolio value: ${total_value:,.2f}, Change: {total_change:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in portfolio tracker: {e}")
    
    async def _execute_market_alerts(self):
        """Execute market alert workflow."""
        # This would be triggered by webhooks in a real implementation
        pass
    
    async def _send_alert(self, title: str, message: str):
        """Send an alert notification."""
        alert = {
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "type": "price_alert"
        }
        
        # In a real implementation, this would integrate with the Shnifter event system
        logger.info(f"ALERT: {title} - {message}")
        
        # TODO: Integrate with Shnifter's event bus/notification system
    
    def set_price_alert(self, symbol: str, **thresholds):
        """Set price alert thresholds for a cryptocurrency."""
        self.alert_thresholds[symbol] = thresholds
        logger.info(f"Set price alerts for {symbol}: {thresholds}")
    
    def add_portfolio_holding(self, symbol: str, amount: float, entry_price: float = None):
        """Add a cryptocurrency holding to the portfolio."""
        self.portfolio_data[symbol] = {
            "amount": amount,
            "entry_price": entry_price,
            "added_at": datetime.now().isoformat()
        }
        logger.info(f"Added {amount} {symbol} to portfolio")
    
    def get_market_data(self, symbol: str = None) -> Dict[str, Any]:
        """Get current market data."""
        if symbol:
            return self.market_data_cache.get(symbol, {})
        return self.market_data_cache
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get current portfolio statistics."""
        return self.portfolio_data.get("_stats", {})
    
    def stop(self):
        """Stop the workflow adapter."""
        self.running = False
        logger.info("N8N Workflow Adapter stopped")


# Example usage and integration
async def main():
    """Example of how to use the N8N Workflow Adapter."""
    
    # Initialize the adapter
    adapter = N8NWorkflowAdapter({
        "webhook_port": 8765
    })
    
    await adapter.initialize()
    
    # Set up some price alerts
    adapter.set_price_alert("bitcoin", price_above=100000, price_below=50000, change_above=10)
    adapter.set_price_alert("ethereum", price_above=5000, price_below=2000, change_above=15)
    
    # Add some portfolio holdings
    adapter.add_portfolio_holding("bitcoin", 0.1, 60000)
    adapter.add_portfolio_holding("ethereum", 2.0, 3000)
    
    # Run for demo purposes
    try:
        await asyncio.sleep(300)  # Run for 5 minutes
    except KeyboardInterrupt:
        adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
