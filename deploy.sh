#!/bin/bash

# Deploy script untuk GAT API ke Vercel
echo "🚀 Deploying GAT Prediction API to Vercel..."

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ vercel.json not found. Please run this script from the vercel-gat-api directory."
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📋 Files in directory:"
ls -la

echo ""
echo "📦 Checking requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
    echo "Dependencies:"
    cat requirements.txt
else
    echo "❌ requirements.txt not found"
fi

echo ""
echo "🏗️  Checking project structure..."
echo "API files:"
ls -la api/
echo "Model files:"
ls -la models/

echo ""
echo "🚀 Starting Vercel deployment..."
echo "Note: First time deployment will ask for project configuration."
echo ""

# Deploy to Vercel
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔗 Your API endpoints:"
echo "   Home: https://your-app.vercel.app/"
echo "   Predict: https://your-app.vercel.app/api/predict"
echo "   Health: https://your-app.vercel.app/api/health"
echo "   Model Info: https://your-app.vercel.app/api/model-info"
echo ""
echo "📝 Test your API:"
echo '   curl -X POST https://your-app.vercel.app/api/predict \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"irt_ability": 0.5, "survey_confidence": 0.8}'"'"''
echo ""
echo "🎯 Happy predicting!"