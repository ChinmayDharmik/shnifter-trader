import sys
import unittest
import pytest
import os
from pathlib import Path

# Enhanced unified test runner for Shnifter Trader

def run_unittest():
    """Run all unittest-based tests in the project with enhanced reporting."""
    print("[INFO] Running enhanced unittest suite...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Enhanced runner with better output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print enhanced summary
    print("\n" + "=" * 60)
    print("🎯 UNITTEST SUMMARY:")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if hasattr(result, 'skipped'):
        print(f"Skipped: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

def run_pytest_html():
    """Run all pytest-based tests and output an HTML report with enhanced coverage."""
    print("[INFO] Running pytest suite with enhanced HTML report...")
    
    # Enhanced pytest configuration
    pytest_args = [
        "-v",                                    # Verbose output
        "--html=shnifter_test_report.html",     # HTML report
        "--self-contained-html",                 # Self-contained report
        "--tb=short",                           # Short traceback format
        "--maxfail=10",                         # Stop after 10 failures
        "--durations=10",                       # Show 10 slowest tests
        "-x",                                   # Stop on first failure (optional)
    ]
    
    # Add coverage if available
    try:
        import importlib
        pytest_cov_spec = importlib.util.find_spec("pytest_cov")
        if pytest_cov_spec is not None:
            pytest_args.extend([
                "--cov=core",
                "--cov=shnifter_frontend",
                "--cov=shnifter_analysis_modules", 
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
            print("[INFO] Coverage reporting enabled")
        else:
            print("[INFO] Coverage reporting not available (install pytest-cov)")
    except Exception as e:
        print(f"[INFO] Coverage reporting not available: {e}")
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    print("🎯 PYTEST SUMMARY:")
    print("=" * 60)
    print(f"Report generated: shnifter_test_report.html")
    
    if 'pytest_cov' in sys.modules:
        print(f"Coverage report: htmlcov/index.html")
    
    return exit_code == 0

def run_enhanced_tests():
    """Run the enhanced test suite with event system integration."""
    print("[INFO] Running enhanced Shnifter test suite...")
    
    try:
        from test_shnifter_enhanced import run_enhanced_tests
        return run_enhanced_tests()
    except ImportError as e:
        print(f"[ERROR] Enhanced tests not available: {e}")
        return False

def run_widget_tests():
    """Run specific widget integration tests."""
    print("[INFO] Running widget integration tests...")
    
    # Find and run widget-specific tests
    widget_test_files = [
        "test_shnifter_table_widget.py",
        "tests/test_shnifter_widgets.py"
    ]
    
    success = True
    for test_file in widget_test_files:
        if Path(test_file).exists():
            print(f"[INFO] Running {test_file}...")
            try:
                exit_code = pytest.main(["-v", test_file])
                if exit_code != 0:
                    success = False
            except Exception as e:
                print(f"[ERROR] Failed to run {test_file}: {e}")
                success = False
        else:
            print(f"[SKIP] {test_file} not found")
    
    return success

def run_analysis_module_tests():
    """Run tests for converted analysis modules."""
    print("[INFO] Running analysis module tests...")
    
    analysis_modules_dir = Path("shnifter_analysis_modules")
    if not analysis_modules_dir.exists():
        print("[SKIP] Analysis modules directory not found")
        return True
    
    # Test each analysis module
    success = True
    modules_tested = 0
    
    for module_file in analysis_modules_dir.glob("*_module.py"):
        try:
            module_name = module_file.stem
            print(f"[INFO] Testing {module_name}...")
            
            # Import and basic test
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            modules_tested += 1
            print(f"[OK] {module_name} imported successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to test {module_name}: {e}")
            success = False
    
    print(f"[INFO] Tested {modules_tested} analysis modules")
    return success

def generate_test_report():
    """Generate comprehensive test coverage report."""
    print("\n" + "=" * 60)
    print("📊 GENERATING COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    report_lines = []
    report_lines.append("# Shnifter Test Coverage Report")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    
    # Count test files
    test_files = list(Path(".").glob("test_*.py")) + list(Path("tests").glob("test_*.py"))
    report_lines.append(f"## Test Files Found: {len(test_files)}")
    
    for test_file in test_files:
        report_lines.append(f"- {test_file}")
    
    # Count widget files
    widget_dirs = list(Path("shnifter_frontend").glob("shnifter_*_widgets"))
    total_widgets = sum(len(list(widget_dir.glob("*.py"))) - 1 for widget_dir in widget_dirs)  # -1 for __init__.py
    
    report_lines.append(f"\n## Frontend Widgets: {total_widgets}")
    for widget_dir in widget_dirs:
        widget_count = len(list(widget_dir.glob("*.py"))) - 1
        report_lines.append(f"- {widget_dir.name}: {widget_count} widgets")
    
    # Count analysis modules
    analysis_dir = Path("shnifter_analysis_modules")
    if analysis_dir.exists():
        analysis_modules = list(analysis_dir.glob("*_module.py"))
        report_lines.append(f"\n## Analysis Modules: {len(analysis_modules)}")
    
    # Write report
    with open("SHNIFTER_TEST_COVERAGE_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print("📄 Report saved: SHNIFTER_TEST_COVERAGE_REPORT.md")

if __name__ == "__main__":
    from datetime import datetime
    
    if len(sys.argv) > 1:
        if "--pytest" in sys.argv:
            success = run_pytest_html()
        elif "--enhanced" in sys.argv:
            success = run_enhanced_tests()
        elif "--widgets" in sys.argv:
            success = run_widget_tests()
        elif "--analysis" in sys.argv:
            success = run_analysis_module_tests()
        elif "--all" in sys.argv:
            print("🚀 Running complete test suite...")
            success = True
            success &= run_enhanced_tests()
            success &= run_widget_tests()
            success &= run_analysis_module_tests()
            success &= run_pytest_html()
            generate_test_report()
        else:
            success = run_unittest()
    else:
        success = run_unittest()
    
    sys.exit(0 if success else 1)