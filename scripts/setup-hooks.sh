#!/usr/bin/env bash
# Setup script for git hooks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GIT_HOOKS_DIR="$SCRIPT_DIR/../.git/hooks"

echo "Setting up git hooks..."

# Copy the main branch protection hook
cp "$SCRIPT_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"

echo "✓ Git hooks installed successfully"
echo ""
echo "The following hooks are now active:"
echo "  - pre-commit: Blocks direct commits to 'main' branch"
