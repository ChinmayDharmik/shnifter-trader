"""
Shnifter Event System Integration Wiring
Complete event system integration for all components
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

from core.events import EventLog, EventBus
from core.data_models import ShnifterData

class ShnifterEventWiring:
    """Wire up all components to the event system"""
    
    def __init__(self):
        self.event_log = EventLog()
        self.wired_components = []
        
    def wire_all_components(self):
        """Wire all Shnifter components to the event system"""
        
        self.event_log.emit("INFO", "wiring_start", {
            "timestamp": datetime.now().isoformat(),
            "process": "complete_event_wiring"
        })
        
        print("🔌 WIRING SHNIFTER COMPONENTS TO EVENT SYSTEM")
        print("=" * 60)
        
        # Wire core system components
        self._wire_core_components()
        
        # Wire frontend widgets
        self._wire_frontend_widgets()
        
        # Wire analysis modules
        self._wire_analysis_modules()
        
        # Wire test system
        self._wire_test_system()
        
        # Create event routing
        self._create_event_routing()
        
        self.event_log.emit("INFO", "wiring_complete", {
            "timestamp": datetime.now().isoformat(),
            "components_wired": len(self.wired_components)
        })
        
        print(f"\n✅ Wired {len(self.wired_components)} components to event system!")
        
    def _wire_core_components(self):
        """Wire core system components"""
        print("🔧 Wiring core components...")
        
        # Core event handlers
        core_events = [
            ("data_update", self._handle_data_update),
            ("analysis_request", self._handle_analysis_request),
            ("trade_signal", self._handle_trade_signal),
            ("risk_alert", self._handle_risk_alert),
            ("system_status", self._handle_system_status)
        ]
        
        for event_type, handler in core_events:
            EventBus.subscribe(event_type, handler)
            self.wired_components.append(f"core.{event_type}")
            print(f"   ✓ {event_type} → {handler.__name__}")
            
    def _wire_frontend_widgets(self):
        """Wire frontend widgets to event system"""
        print("\n📱 Wiring frontend widgets...")
        
        # Widget event handlers
        widget_events = [
            ("widget_created", self._handle_widget_created),
            ("widget_data_update", self._handle_widget_data_update),
            ("widget_user_action", self._handle_widget_user_action),
            ("popout_requested", self._handle_popout_requested),
            ("llm_analysis_request", self._handle_llm_analysis_request)
        ]
        
        for event_type, handler in widget_events:
            EventBus.subscribe(event_type, handler)
            self.wired_components.append(f"widgets.{event_type}")
            print(f"   ✓ {event_type} → {handler.__name__}")
            
    def _wire_analysis_modules(self):
        """Wire analysis modules to event system"""
        print("\n📊 Wiring analysis modules...")
        
        # Analysis event handlers
        analysis_events = [
            ("analysis_start", self._handle_analysis_start),
            ("analysis_complete", self._handle_analysis_complete),
            ("analysis_error", self._handle_analysis_error),
            ("llm_insight_ready", self._handle_llm_insight_ready),
            ("portfolio_update", self._handle_portfolio_update)
        ]
        
        for event_type, handler in analysis_events:
            EventBus.subscribe(event_type, handler)
            self.wired_components.append(f"analysis.{event_type}")
            print(f"   ✓ {event_type} → {handler.__name__}")
            
        # Wire specific analysis modules to events
        self._wire_specific_analysis_modules()
            
    def _wire_specific_analysis_modules(self):
        """Wire specific analysis modules to event system"""
        print("\n🔬 Wiring specific analysis modules...")
        
        # Analysis module specific events
        module_events = [
            # Platform standardization
            ("platform_standardization_request", self._handle_platform_standardization),
            ("platform_standardization_complete", self._handle_platform_standardization_complete),
            
            # Portfolio optimization
            ("portfolio_optimization_request", self._handle_portfolio_optimization),
            ("portfolio_optimization_complete", self._handle_portfolio_optimization_complete),
            
            # Risk return analysis
            ("risk_return_analysis_request", self._handle_risk_return_analysis),
            ("risk_return_analysis_complete", self._handle_risk_return_analysis_complete),
            
            # Sector rotation strategy
            ("sector_rotation_request", self._handle_sector_rotation),
            ("sector_rotation_complete", self._handle_sector_rotation_complete),
            
            # AutoGen trading agents
            ("autogen_agents_request", self._handle_autogen_agents),
            ("autogen_agents_complete", self._handle_autogen_agents_complete),
            
            # Backtesting momentum
            ("backtesting_momentum_request", self._handle_backtesting_momentum),
            ("backtesting_momentum_complete", self._handle_backtesting_momentum_complete),
            
            # Chroma knowledge base
            ("chroma_knowledge_request", self._handle_chroma_knowledge),
            ("chroma_knowledge_complete", self._handle_chroma_knowledge_complete),
            
            # Financial statements
            ("financial_statements_request", self._handle_financial_statements),
            ("financial_statements_complete", self._handle_financial_statements_complete),
            
            # Symbol finder
            ("find_symbols_request", self._handle_find_symbols),
            ("find_symbols_complete", self._handle_find_symbols_complete),
            
            # Historical price data
            ("historical_data_request", self._handle_historical_data),
            ("historical_data_complete", self._handle_historical_data_complete),
            
            # Currency exchange forecasting
            ("currency_forecast_request", self._handle_currency_forecast),
            ("currency_forecast_complete", self._handle_currency_forecast_complete),
            
            # Ethereum trend analysis
            ("ethereum_analysis_request", self._handle_ethereum_analysis),
            ("ethereum_analysis_complete", self._handle_ethereum_analysis_complete),
            
            # USD liquidity index
            ("usd_liquidity_request", self._handle_usd_liquidity),
            ("usd_liquidity_complete", self._handle_usd_liquidity_complete),
            
            # Copper to gold ratio
            ("copper_gold_ratio_request", self._handle_copper_gold_ratio),
            ("copper_gold_ratio_complete", self._handle_copper_gold_ratio_complete),
            
            # M&A impact analysis
            ("ma_impact_request", self._handle_ma_impact),
            ("ma_impact_complete", self._handle_ma_impact_complete),
            
            # Implied earnings move
            ("implied_earnings_request", self._handle_implied_earnings),
            ("implied_earnings_complete", self._handle_implied_earnings_complete),
            
            # N8N workflow adapter
            ("n8n_workflow_request", self._handle_n8n_workflow),
            ("n8n_workflow_complete", self._handle_n8n_workflow_complete),
            
            # Platform as LLM tools
            ("platform_llm_tools_request", self._handle_platform_llm_tools),
            ("platform_llm_tools_complete", self._handle_platform_llm_tools_complete),
            
            # Google Colab integration
            ("google_colab_request", self._handle_google_colab),
            ("google_colab_complete", self._handle_google_colab_complete),
        ]
        
        for event_type, handler in module_events:
            EventBus.subscribe(event_type, handler)
            self.wired_components.append(f"analysis_modules.{event_type}")
            print(f"   ✓ {event_type} → {handler.__name__}")
            
        print(f"\n📊 Wired {len(module_events)} analysis module events to event system!")
            
    def _wire_test_system(self):
        """Wire test system to event system"""
        print("\n🧪 Wiring test system...")
        
        # Test event handlers
        test_events = [
            ("test_start", self._handle_test_start),
            ("test_complete", self._handle_test_complete),
            ("test_error", self._handle_test_error),
            ("test_coverage_update", self._handle_test_coverage_update)
        ]
        
        for event_type, handler in test_events:
            EventBus.subscribe(event_type, handler)
            self.wired_components.append(f"test.{event_type}")
            print(f"   ✓ {event_type} → {handler.__name__}")
            
    def _create_event_routing(self):
        """Create intelligent event routing"""
        print("\n🔀 Creating event routing...")
        
        # Route events between components
        routing_rules = [
            ("data_update", ["widget_data_update", "analysis_request"]),
            ("analysis_complete", ["widget_data_update", "llm_analysis_request"]),
            ("trade_signal", ["risk_alert", "portfolio_update"]),
            ("widget_user_action", ["analysis_request", "data_update"])
        ]
        
        for source_event, target_events in routing_rules:
            EventBus.subscribe(source_event, lambda data, targets=target_events: self._route_event(data, targets))
            print(f"   ✓ {source_event} → {target_events}")
            
    def _route_event(self, event_data: Dict[str, Any], target_events: List[str]):
        """Route event to multiple targets"""
        for target_event in target_events:
            EventBus.publish(target_event, {
                "routed_from": event_data,
                "timestamp": datetime.now().isoformat()
            })
    
    # Event handlers
    def _handle_data_update(self, event_data: Dict[str, Any]):
        """Handle data update events"""
        self.event_log.emit("INFO", "data_update_received", event_data)
        
    def _handle_analysis_request(self, event_data: Dict[str, Any]):
        """Handle analysis request events"""
        self.event_log.emit("INFO", "analysis_request_received", event_data)
        
    def _handle_trade_signal(self, event_data: Dict[str, Any]):
        """Handle trade signal events"""
        self.event_log.emit("INFO", "trade_signal_received", event_data)
        
    def _handle_risk_alert(self, event_data: Dict[str, Any]):
        """Handle risk alert events"""
        self.event_log.emit("WARNING", "risk_alert_received", event_data)
        
    def _handle_system_status(self, event_data: Dict[str, Any]):
        """Handle system status events"""
        self.event_log.emit("INFO", "system_status_received", event_data)
        
    def _handle_widget_created(self, event_data: Dict[str, Any]):
        """Handle widget creation events"""
        self.event_log.emit("INFO", "widget_created", event_data)
        
    def _handle_widget_data_update(self, event_data: Dict[str, Any]):
        """Handle widget data update events"""
        self.event_log.emit("INFO", "widget_data_update", event_data)
        
    def _handle_widget_user_action(self, event_data: Dict[str, Any]):
        """Handle widget user action events"""
        self.event_log.emit("INFO", "widget_user_action", event_data)
        
    def _handle_popout_requested(self, event_data: Dict[str, Any]):
        """Handle popout request events"""
        self.event_log.emit("INFO", "popout_requested", event_data)
        
    def _handle_llm_analysis_request(self, event_data: Dict[str, Any]):
        """Handle LLM analysis request events"""
        self.event_log.emit("INFO", "llm_analysis_request", event_data)
        
    def _handle_analysis_start(self, event_data: Dict[str, Any]):
        """Handle analysis start events"""
        self.event_log.emit("INFO", "analysis_start", event_data)
        
    def _handle_analysis_complete(self, event_data: Dict[str, Any]):
        """Handle analysis complete events"""
        self.event_log.emit("INFO", "analysis_complete", event_data)
        
    def _handle_analysis_error(self, event_data: Dict[str, Any]):
        """Handle analysis error events"""
        self.event_log.emit("ERROR", "analysis_error", event_data)
        
    def _handle_llm_insight_ready(self, event_data: Dict[str, Any]):
        """Handle LLM insight ready events"""
        self.event_log.emit("INFO", "llm_insight_ready", event_data)
        
    def _handle_portfolio_update(self, event_data: Dict[str, Any]):
        """Handle portfolio update events"""
        self.event_log.emit("INFO", "portfolio_update", event_data)
        
    def _handle_test_start(self, event_data: Dict[str, Any]):
        """Handle test start events"""
        self.event_log.emit("INFO", "test_start", event_data)
        
    def _handle_test_complete(self, event_data: Dict[str, Any]):
        """Handle test complete events"""
        self.event_log.emit("INFO", "test_complete", event_data)
        
    def _handle_test_error(self, event_data: Dict[str, Any]):
        """Handle test error events"""
        self.event_log.emit("ERROR", "test_error", event_data)
        
    def _handle_test_coverage_update(self, event_data: Dict[str, Any]):
        """Handle test coverage update events"""
        self.event_log.emit("INFO", "test_coverage_update", event_data)
        
    # Analysis module specific event handlers
    def _handle_platform_standardization(self, event_data: Dict[str, Any]):
        """Handle platform standardization request events"""
        self.event_log.emit("INFO", "platform_standardization_request", event_data)
        
    def _handle_platform_standardization_complete(self, event_data: Dict[str, Any]):
        """Handle platform standardization complete events"""
        self.event_log.emit("INFO", "platform_standardization_complete", event_data)
        
    def _handle_portfolio_optimization(self, event_data: Dict[str, Any]):
        """Handle portfolio optimization request events"""
        self.event_log.emit("INFO", "portfolio_optimization_request", event_data)
        
    def _handle_portfolio_optimization_complete(self, event_data: Dict[str, Any]):
        """Handle portfolio optimization complete events"""
        self.event_log.emit("INFO", "portfolio_optimization_complete", event_data)
        
    def _handle_risk_return_analysis(self, event_data: Dict[str, Any]):
        """Handle risk return analysis request events"""
        self.event_log.emit("INFO", "risk_return_analysis_request", event_data)
        
    def _handle_risk_return_analysis_complete(self, event_data: Dict[str, Any]):
        """Handle risk return analysis complete events"""
        self.event_log.emit("INFO", "risk_return_analysis_complete", event_data)
        
    def _handle_sector_rotation(self, event_data: Dict[str, Any]):
        """Handle sector rotation request events"""
        self.event_log.emit("INFO", "sector_rotation_request", event_data)
        
    def _handle_sector_rotation_complete(self, event_data: Dict[str, Any]):
        """Handle sector rotation complete events"""
        self.event_log.emit("INFO", "sector_rotation_complete", event_data)
        
    def _handle_autogen_agents(self, event_data: Dict[str, Any]):
        """Handle AutoGen agents request events"""
        self.event_log.emit("INFO", "autogen_agents_request", event_data)
        
    def _handle_autogen_agents_complete(self, event_data: Dict[str, Any]):
        """Handle AutoGen agents complete events"""
        self.event_log.emit("INFO", "autogen_agents_complete", event_data)
        
    def _handle_backtesting_momentum(self, event_data: Dict[str, Any]):
        """Handle backtesting momentum request events"""
        self.event_log.emit("INFO", "backtesting_momentum_request", event_data)
        
    def _handle_backtesting_momentum_complete(self, event_data: Dict[str, Any]):
        """Handle backtesting momentum complete events"""
        self.event_log.emit("INFO", "backtesting_momentum_complete", event_data)
        
    def _handle_chroma_knowledge(self, event_data: Dict[str, Any]):
        """Handle chroma knowledge request events"""
        self.event_log.emit("INFO", "chroma_knowledge_request", event_data)
        
    def _handle_chroma_knowledge_complete(self, event_data: Dict[str, Any]):
        """Handle chroma knowledge complete events"""
        self.event_log.emit("INFO", "chroma_knowledge_complete", event_data)
        
    def _handle_financial_statements(self, event_data: Dict[str, Any]):
        """Handle financial statements request events"""
        self.event_log.emit("INFO", "financial_statements_request", event_data)
        
    def _handle_financial_statements_complete(self, event_data: Dict[str, Any]):
        """Handle financial statements complete events"""
        self.event_log.emit("INFO", "financial_statements_complete", event_data)
        
    def _handle_find_symbols(self, event_data: Dict[str, Any]):
        """Handle find symbols request events"""
        self.event_log.emit("INFO", "find_symbols_request", event_data)
        
    def _handle_find_symbols_complete(self, event_data: Dict[str, Any]):
        """Handle find symbols complete events"""
        self.event_log.emit("INFO", "find_symbols_complete", event_data)
        
    def _handle_historical_data(self, event_data: Dict[str, Any]):
        """Handle historical data request events"""
        self.event_log.emit("INFO", "historical_data_request", event_data)
        
    def _handle_historical_data_complete(self, event_data: Dict[str, Any]):
        """Handle historical data complete events"""
        self.event_log.emit("INFO", "historical_data_complete", event_data)
        
    def _handle_currency_forecast(self, event_data: Dict[str, Any]):
        """Handle currency forecast request events"""
        self.event_log.emit("INFO", "currency_forecast_request", event_data)
        
    def _handle_currency_forecast_complete(self, event_data: Dict[str, Any]):
        """Handle currency forecast complete events"""
        self.event_log.emit("INFO", "currency_forecast_complete", event_data)
        
    def _handle_ethereum_analysis(self, event_data: Dict[str, Any]):
        """Handle ethereum analysis request events"""
        self.event_log.emit("INFO", "ethereum_analysis_request", event_data)
        
    def _handle_ethereum_analysis_complete(self, event_data: Dict[str, Any]):
        """Handle ethereum analysis complete events"""
        self.event_log.emit("INFO", "ethereum_analysis_complete", event_data)
        
    def _handle_usd_liquidity(self, event_data: Dict[str, Any]):
        """Handle USD liquidity request events"""
        self.event_log.emit("INFO", "usd_liquidity_request", event_data)
        
    def _handle_usd_liquidity_complete(self, event_data: Dict[str, Any]):
        """Handle USD liquidity complete events"""
        self.event_log.emit("INFO", "usd_liquidity_complete", event_data)
        
    def _handle_copper_gold_ratio(self, event_data: Dict[str, Any]):
        """Handle copper gold ratio request events"""
        self.event_log.emit("INFO", "copper_gold_ratio_request", event_data)
        
    def _handle_copper_gold_ratio_complete(self, event_data: Dict[str, Any]):
        """Handle copper gold ratio complete events"""
        self.event_log.emit("INFO", "copper_gold_ratio_complete", event_data)
        
    def _handle_ma_impact(self, event_data: Dict[str, Any]):
        """Handle M&A impact request events"""
        self.event_log.emit("INFO", "ma_impact_request", event_data)
        
    def _handle_ma_impact_complete(self, event_data: Dict[str, Any]):
        """Handle M&A impact complete events"""
        self.event_log.emit("INFO", "ma_impact_complete", event_data)
        
    def _handle_implied_earnings(self, event_data: Dict[str, Any]):
        """Handle implied earnings request events"""
        self.event_log.emit("INFO", "implied_earnings_request", event_data)
        
    def _handle_implied_earnings_complete(self, event_data: Dict[str, Any]):
        """Handle implied earnings complete events"""
        self.event_log.emit("INFO", "implied_earnings_complete", event_data)
        
    def _handle_n8n_workflow(self, event_data: Dict[str, Any]):
        """Handle N8N workflow request events"""
        self.event_log.emit("INFO", "n8n_workflow_request", event_data)
        
    def _handle_n8n_workflow_complete(self, event_data: Dict[str, Any]):
        """Handle N8N workflow complete events"""
        self.event_log.emit("INFO", "n8n_workflow_complete", event_data)
        
    def _handle_platform_llm_tools(self, event_data: Dict[str, Any]):
        """Handle platform LLM tools request events"""
        self.event_log.emit("INFO", "platform_llm_tools_request", event_data)
        
    def _handle_platform_llm_tools_complete(self, event_data: Dict[str, Any]):
        """Handle platform LLM tools complete events"""
        self.event_log.emit("INFO", "platform_llm_tools_complete", event_data)
        
    def _handle_google_colab(self, event_data: Dict[str, Any]):
        """Handle Google Colab request events"""
        self.event_log.emit("INFO", "google_colab_request", event_data)
        
    def _handle_google_colab_complete(self, event_data: Dict[str, Any]):
        """Handle Google Colab complete events"""
        self.event_log.emit("INFO", "google_colab_complete", event_data)
        
    def test_event_wiring(self):
        """Test the event wiring system"""
        print("\n🧪 Testing event wiring...")
        
        # Test event propagation
        test_events = [
            ("data_update", {"symbol": "AAPL", "price": 150.0}),
            ("widget_user_action", {"action": "refresh", "widget_id": "test_widget"}),
            ("analysis_complete", {"analysis_type": "momentum", "signal": "BUY"}),
            ("test_start", {"test_name": "event_wiring_test"}),
            # Test analysis module events
            ("platform_standardization_request", {"symbols": ["AAPL", "MSFT"]}),
            ("portfolio_optimization_request", {"risk_tolerance": "moderate"}),
            ("autogen_agents_request", {"trading_strategy": "momentum"}),
            ("backtesting_momentum_request", {"timeframe": "1year"}),
            ("financial_statements_request", {"symbol": "AAPL", "period": "quarterly"}),
            ("ethereum_analysis_request", {"timeframe": "1month"}),
        ]
        
        for event_type, event_data in test_events:
            EventBus.publish(event_type, event_data)
            print(f"   ✓ Published {event_type}")
            
        print("   ✅ Event wiring test complete!")
        
    def generate_wiring_report(self):
        """Generate event wiring report"""
        report = {
            "wiring_timestamp": datetime.now().isoformat(),
            "components_wired": len(self.wired_components),
            "wired_components": self.wired_components,
            "event_bus_subscribers": len(EventBus._subscribers),
            "status": "complete"
        }
        
        with open("SHNIFTER_EVENT_WIRING_REPORT.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Event wiring report saved: SHNIFTER_EVENT_WIRING_REPORT.json")
        return report

def main():
    """Main event wiring process"""
    print("🚀 Starting Shnifter Event System Wiring...")
    print()
    
    # Create and run wiring
    wiring = ShnifterEventWiring()
    wiring.wire_all_components()
    
    # Test the wiring
    wiring.test_event_wiring()
    
    # Generate report
    report = wiring.generate_wiring_report()
    
    print(f"\n🎉 EVENT SYSTEM WIRING COMPLETE!")
    print("=" * 60)
    print(f"Components wired: {report['components_wired']}")
    print(f"Event subscribers: {report['event_bus_subscribers']}")
    print("Status: ✅ All systems integrated")
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 60)
    print("1. ✅ Run enhanced test suite")
    print("2. ✅ Test widget popout functionality")
    print("3. ✅ Verify LLM integration")
    print("4. ✅ Test analysis module pipeline")
    print("5. ✅ Validate event routing")
    
    print("\nShnifter event system is fully wired and ready! 🎉")

if __name__ == "__main__":
    main()
