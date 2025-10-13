# Deployment Commands untuk GAT API

## Windows PowerShell
```powershell
# 1. Masuk ke folder
cd "c:\Users\hilmi\OneDrive\Desktop\GNN-adaptive\vercel-gat-api"

# 2. Install Vercel CLI (jika belum ada)
npm install -g vercel

# 3. Test locally (optional)
pip install -r requirements.txt
python api/index.py

# 4. Deploy ke Vercel
vercel

# 5. Deploy production
vercel --prod
```

## Command Line / Bash
```bash
# 1. Masuk ke folder
cd /c/Users/hilmi/OneDrive/Desktop/GNN-adaptive/vercel-gat-api

# 2. Copy model file (optional)
cp ../data/enhanced_gat_complete.pth models/

# 3. Deploy
chmod +x deploy.sh
./deploy.sh
```

## Manual Vercel Dashboard
1. Buka https://vercel.com/dashboard
2. Klik "New Project"
3. Import repository atau upload folder
4. Configure:
   - Framework Preset: Other
   - Build Command: (kosongkan)
   - Output Directory: (kosongkan)
   - Install Command: pip install -r requirements.txt
5. Deploy

## Environment Variables (Optional)
Bisa ditambah di Vercel dashboard:
- `MODEL_PATH`: Path ke model file
- `DEBUG_MODE`: true/false
- `API_VERSION`: 1.0.0

## Testing After Deploy
```bash
# Replace dengan URL Vercel Anda
API_URL="https://your-app.vercel.app"

# Test health
curl $API_URL/api/health

# Test prediction
curl -X POST $API_URL/api/predict \
  -H "Content-Type: application/json" \
  -d '{"irt_ability": 0.5, "survey_confidence": 0.8}'
```