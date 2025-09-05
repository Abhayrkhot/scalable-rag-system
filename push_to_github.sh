#!/bin/bash

echo "🚀 Pushing Scalable RAG System to GitHub"
echo "========================================"

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "✅ Remote 'origin' already exists"
    git remote -v
else
    echo "📝 Please create a repository on GitHub first:"
    echo "   1. Go to https://github.com/new"
    echo "   2. Repository name: scalable-rag-system"
    echo "   3. Description: A high-performance RAG system for millions of documents"
    echo "   4. Choose Public or Private"
    echo "   5. Don't initialize with README (we have one)"
    echo "   6. Click 'Create repository'"
    echo ""
    echo "Then run:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/scalable-rag-system.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "Replace YOUR_USERNAME with your actual GitHub username"
fi

echo ""
echo "📊 Repository Status:"
echo "===================="
git status --short
echo ""
echo "📈 Commit History:"
echo "=================="
git log --oneline -5
echo ""
echo "📁 Files to be pushed:"
echo "======================"
git ls-files | wc -l | xargs echo "Total files:"
git ls-files | head -10
if [ $(git ls-files | wc -l) -gt 10 ]; then
    echo "... and $(($(git ls-files | wc -l) - 10)) more files"
fi
