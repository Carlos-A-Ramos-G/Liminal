#!/usr/bin/env python3
"""Thin shim — delegates to the liminal package.  Use: python rbfe_runner.py <command>"""
from liminal.__main__ import main

if __name__ == "__main__":
    main()
