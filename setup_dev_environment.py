#!/usr/bin/env python3
"""
Shnifter Trader - Development Setup Script
This script sets up the development environment for Shnifter Trader
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Shnifter Trader Development Environment")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "Multi_Model_Trading_Bot.py").exists():
        print("❌ Please run this script from the Shnifter Trader root directory")
        sys.exit(1)
    
    # Check if poetry is installed
    if not run_command("poetry --version", "Checking Poetry installation"):
        print("📦 Poetry not found. Installing Poetry...")
        run_command("pip install poetry", "Installing Poetry")
    
    # Install main requirements
    print("\n📦 Installing main requirements...")
    run_command("pip install -r requirements.txt", "Installing main requirements")
    
    # Setup shnifter platform core
    platform_core_path = current_dir / "shnifter_platform_core"
    if platform_core_path.exists():
        print(f"\n🔧 Setting up Shnifter Platform Core...")
        os.chdir(platform_core_path)
        
        # Run the dev install script
        if (platform_core_path / "dev_install.py").exists():
            run_command("python dev_install.py", "Running Shnifter Platform Core dev install")
        
        # Install in development mode
        run_command("poetry install", "Installing Shnifter Platform Core dependencies")
        
        # Go back to root
        os.chdir(current_dir)
    
    # Create __init__.py files if missing
    print("\n📁 Ensuring Python package structure...")
    init_files = [
        "shnifter_platform_core/__init__.py",
        "shnifter_platform_core/core/__init__.py",
        "shnifter_platform_core/providers/__init__.py",
        "core/__init__.py",
        "shnifter_frontend/__init__.py",
        "llm_manager/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = current_dir / init_file
        if not init_path.exists():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("# Shnifter Trader Package\n")
            print(f"✅ Created {init_file}")
    
    # Install the project in development mode
    print("\n🔧 Installing Shnifter Trader in development mode...")
    run_command("pip install -e .", "Installing project in development mode")
    
    print("\n🎉 Setup completed successfully!")
    print("💡 You can now run: python Multi_Model_Trading_Bot.py")

if __name__ == "__main__":
    main()
