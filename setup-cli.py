#!/usr/bin/env python3
"""Setup script for installing CLI shortcuts globally.

This script creates symlinks in /usr/local/bin/ for easy access to the CLI tools.
"""

import os
import sys
from pathlib import Path


def install_cli_shortcuts():
    """Install CLI shortcuts globally."""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()

    # Define the shortcuts to create
    shortcuts = {"srf": "srf", "srf-run": "srf-run", "srf-list": "srf-list", "srf-test": "srf-test"}

    # Check if we have write permissions to /usr/local/bin
    local_bin = Path("/usr/local/bin")
    if not local_bin.exists():
        print("❌ /usr/local/bin does not exist. Cannot install shortcuts.")
        return False

    if not os.access(local_bin, os.W_OK):
        print("❌ No write permission to /usr/local/bin. Try running with sudo.")
        print("   sudo python setup-cli.py")
        return False

    print("🔧 Installing CLI shortcuts...")

    for script_name, shortcut_name in shortcuts.items():
        script_path = current_dir / script_name
        shortcut_path = local_bin / shortcut_name

        if not script_path.exists():
            print(f"❌ Script {script_name} not found!")
            continue

        try:
            # Create symlink
            if shortcut_path.exists():
                shortcut_path.unlink()

            os.symlink(script_path, shortcut_path)
            print(f"✅ Created shortcut: {shortcut_name}")

        except Exception as e:
            print(f"❌ Failed to create {shortcut_name}: {e}")

    print("\n🎉 Installation complete!")
    print("You can now use:")
    print("  srf run config.yaml")
    print("  srf list --attacks")
    print("  srf test --unit")
    print("  srf-run config.yaml")
    print("  srf-list --attacks")
    print("  srf-test --unit")

    return True


def uninstall_cli_shortcuts():
    """Uninstall CLI shortcuts."""
    local_bin = Path("/usr/local/bin")

    shortcuts = ["srf", "srf-run", "srf-list", "srf-test"]

    print("🗑️  Uninstalling CLI shortcuts...")

    for shortcut in shortcuts:
        shortcut_path = local_bin / shortcut
        if shortcut_path.exists():
            try:
                shortcut_path.unlink()
                print(f"✅ Removed: {shortcut}")
            except Exception as e:
                print(f"❌ Failed to remove {shortcut}: {e}")
        else:
            print(f"⚠️  {shortcut} not found")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        uninstall_cli_shortcuts()
    else:
        install_cli_shortcuts()
