#!/bin/bash
# Automatically format all Python files with Black

echo "Formatting Python files with Black..."
black asteroids/ training/ tools/ *.py

if [ $? -eq 0 ]; then
    echo "✓ All Python files formatted successfully!"
else
    echo "✗ Black formatting failed"
    exit 1
fi
