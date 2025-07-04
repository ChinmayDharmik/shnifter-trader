"""
Shnifter Test Coverage Assessment and Enhancement
Complete analysis of test coverage for the converted system
"""

from pathlib import Path
from datetime import datetime
import json

def assess_test_coverage():
    """Assess current test coverage across all Shnifter components"""
    
    print("🔍 SHNIFTER TEST COVERAGE ASSESSMENT")
    print("=" * 60)
    
    # Core system components
    core_files = list(Path("core").glob("*.py"))
    print(f"📁 Core modules: {len(core_files)}")
    for file in core_files:
        print(f"   ├─ {file.name}")
    
    # Frontend widgets
    widget_dirs = list(Path("shnifter_frontend").glob("shnifter_*_widgets"))
    total_widgets = 0
    
    print(f"\n📱 Frontend widget directories: {len(widget_dirs)}")
    for widget_dir in widget_dirs:
        widgets = list(widget_dir.glob("*.py"))
        widget_count = len(widgets) - 1  # Exclude __init__.py
        total_widgets += widget_count
        print(f"   ├─ {widget_dir.name}: {widget_count} widgets")
    
    # Analysis modules
    analysis_dir = Path("shnifter_analysis_modules")
    analysis_modules = []
    if analysis_dir.exists():
        analysis_modules = list(analysis_dir.glob("*_module.py"))
        print(f"\n📊 Analysis modules: {len(analysis_modules)}")
        for module in analysis_modules[:10]:  # Show first 10
            print(f"   ├─ {module.stem}")
        if len(analysis_modules) > 10:
            print(f"   └─ ... and {len(analysis_modules) - 10} more")
    
    # Test files
    test_files = list(Path(".").glob("test_*.py")) + list(Path("tests").glob("test_*.py"))
    print(f"\n🧪 Test files: {len(test_files)}")
    for test_file in test_files:
        print(f"   ├─ {test_file}")
    
    # Calculate coverage ratios
    core_coverage = len([f for f in test_files if "core" in str(f) or "event" in str(f)]) / len(core_files) if core_files else 0
    widget_coverage = len([f for f in test_files if "widget" in str(f) or "frontend" in str(f)]) / len(widget_dirs) if widget_dirs else 0
    analysis_coverage = len([f for f in test_files if "analysis" in str(f)]) / len(analysis_modules) if analysis_modules else 0
    
    print(f"\n📈 COVERAGE ANALYSIS:")
    print("=" * 60)
    print(f"Core system coverage: {core_coverage:.1%}")
    print(f"Widget coverage: {widget_coverage:.1%}")
    print(f"Analysis module coverage: {analysis_coverage:.1%}")
    
    overall_coverage = (len(test_files) / (len(core_files) + len(widget_dirs) + len(analysis_modules))) if (len(core_files) + len(widget_dirs) + len(analysis_modules)) > 0 else 0
    print(f"Overall test coverage: {overall_coverage:.1%}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if core_coverage < 0.8:
        print("📌 Add more core system tests (events, data models, config)")
    
    if widget_coverage < 0.5:
        print("📌 Add widget-specific integration tests")
        
    if analysis_coverage < 0.3:
        print("📌 Add analysis module validation tests")
        
    if overall_coverage > 0.7:
        print("✅ Good test coverage achieved!")
    else:
        print("📌 Consider adding more comprehensive test cases")
    
    return {
        "core_files": len(core_files),
        "total_widgets": total_widgets,
        "analysis_modules": len(analysis_modules),
        "test_files": len(test_files),
        "core_coverage": core_coverage,
        "widget_coverage": widget_coverage,
        "analysis_coverage": analysis_coverage,
        "overall_coverage": overall_coverage
    }

def generate_test_recommendations():
    """Generate specific test enhancement recommendations"""
    
    print("\n🎯 SPECIFIC TEST ENHANCEMENT RECOMMENDATIONS:")
    print("=" * 60)
    
    # Check for missing critical tests
    critical_tests = [
        ("test_event_system.py", "Event system integration"),
        ("test_popout_manager.py", "Popout manager functionality"),
        ("test_llm_integration.py", "LLM provider integration"),
        ("test_risk_management.py", "Risk management system"),
        ("test_paper_trading.py", "Paper trading engine")
    ]
    
    missing_tests = []
    for test_file, description in critical_tests:
        if not Path(test_file).exists() and not any(Path("tests").glob(f"*{test_file.replace('test_', '')}")):
            missing_tests.append((test_file, description))
            
    if missing_tests:
        print("📋 Missing critical tests:")
        for test_file, description in missing_tests:
            print(f"   ❌ {test_file} - {description}")
    else:
        print("✅ All critical test areas covered")
    
    # Widget-specific test recommendations
    widget_dirs = list(Path("shnifter_frontend").glob("shnifter_*_widgets"))
    
    print(f"\n📱 Widget test recommendations:")
    for widget_dir in widget_dirs:
        widget_test_file = f"test_{widget_dir.name}.py"
        if not Path(widget_test_file).exists() and not Path("tests").joinpath(widget_test_file).exists():
            print(f"   📌 Create {widget_test_file} for {widget_dir.name}")
    
    # Analysis module test recommendations
    analysis_dir = Path("shnifter_analysis_modules")
    if analysis_dir.exists():
        print(f"\n📊 Analysis module test recommendations:")
        print("   📌 Create test_analysis_modules.py for batch testing")
        print("   📌 Add integration tests with real market data")
        print("   📌 Test LLM integration in analysis modules")
    
    return missing_tests

def create_test_enhancement_plan():
    """Create a comprehensive test enhancement plan"""
    
    coverage_data = assess_test_coverage()
    missing_tests = generate_test_recommendations()
    
    plan = {
        "assessment_date": datetime.now().isoformat(),
        "current_coverage": coverage_data,
        "missing_critical_tests": missing_tests,
        "enhancement_priorities": [
            "1. Event system comprehensive testing",
            "2. Widget integration and UI testing", 
            "3. Analysis module validation testing",
            "4. LLM provider integration testing",
            "5. End-to-end workflow testing"
        ],
        "next_steps": [
            "Create missing critical test files",
            "Add widget-specific test suites",
            "Implement analysis module batch testing",
            "Add performance and load testing",
            "Set up continuous integration pipeline"
        ]
    }
    
    # Write enhancement plan
    with open("SHNIFTER_TEST_ENHANCEMENT_PLAN.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    print(f"\n📄 Test enhancement plan saved: SHNIFTER_TEST_ENHANCEMENT_PLAN.json")
    
    return plan

if __name__ == "__main__":
    print("🚀 Starting Shnifter Test Coverage Assessment...")
    print()
    
    # Run assessment
    plan = create_test_enhancement_plan()
    
    print(f"\n🎉 ASSESSMENT COMPLETE!")
    print("=" * 60)
    print(f"📊 Current test coverage: {plan['current_coverage']['overall_coverage']:.1%}")
    print(f"📁 Components analyzed: {plan['current_coverage']['core_files'] + plan['current_coverage']['total_widgets'] + plan['current_coverage']['analysis_modules']}")
    print(f"🧪 Test files found: {plan['current_coverage']['test_files']}")
    
    if plan['current_coverage']['overall_coverage'] > 0.6:
        print("✅ Good foundation for testing!")
    else:
        print("📌 Significant testing gaps to address")
    
    print("\nReady for enhanced testing and integration! 🎯")
