# GAT Prediction API - Vercel Deployment

🚀 **Ready-to-deploy GAT Prediction API untuk Vercel**

## Fitur
- ✅ Flask API dengan GAT model predictions
- ✅ Arsitektur SimplifiedGATModel (sesuai notebook)  
- ✅ Input validation & error handling
- ✅ CORS enabled untuk frontend integration
- ✅ Health check & model info endpoints
- ✅ Vercel serverless deployment ready

## Struktur Folder
```
vercel-gat-api/
├── api/
│   └── index.py          # Main Flask API
├── models/
│   ├── __init__.py       # Package init
│   ├── gat_model.py      # GAT model architecture  
│   └── predictor.py      # Prediction service
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## API Endpoints

### 1. Home / Documentation
- **URL**: `/` atau `/api/`
- **Method**: `GET`
- **Response**: API documentation & usage

### 2. Predict Student Level (Single)
- **URL**: `/api/predict`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "irt_ability": 0.5,
    "survey_confidence": 0.8
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "predicted_level": 2,
      "confidence": 0.723,
      "level_probabilities": [0.156, 0.723, 0.121],

### 3. Predict Student Level (Batch) 🆕
- **URL**: `/api/predict-batch`
- **Method**: `POST` 
- **Body**:
  ```json
  {
    "students": [
      {
        "student_id": "101",
        "irt_ability": -0.5,
        "survey_confidence": 0.6
      },
      {
        "student_id": "102", 
        "irt_ability": 0.3,
        "survey_confidence": 0.8
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "total_students": 2,
      "successful_predictions": 2,
      "failed_predictions": 0,
      "results": [
        {
          "student_id": "101",
          "predicted_level": 2,
          "confidence": 0.7234,
          "features": {
            "irt_ability": -0.5,
            "survey_confidence": 0.6,
            "ability_percentile": 0.3085
          }
        },
        {
          "student_id": "102",
          "predicted_level": 3, 
          "confidence": 0.8756,
          "features": {
            "irt_ability": 0.3,
            "survey_confidence": 0.8,
            "ability_percentile": 0.6179
          }
        }
      ],
      "errors": []
    }
  }
      "reasoning": "GAT Analysis: IRT=0.500, Confidence=80.0% → Level 2",
      "irt_ability": 0.5,
      "survey_confidence": 0.8,
      "model_trained": true
    }
  }
  ```

### 3. Health Check
- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: Model status & health check

### 4. Model Information
- **URL**: `/api/model-info` 
- **Method**: `GET`
- **Response**: Model architecture details

## Quick Deploy ke Vercel

### Step 1: Persiapan
1. Copy folder `vercel-gat-api` ini
2. Optional: Copy model file `enhanced_gat_complete.pth` ke folder `models/`
3. Install Vercel CLI: `npm i -g vercel`

### Step 2: Deploy
```bash
cd vercel-gat-api
vercel
```

### Step 3: Test API
```bash
# Test health
curl https://your-app.vercel.app/api/health

# Test prediction
curl -X POST https://your-app.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"irt_ability": 0.5, "survey_confidence": 0.8}'
```

## Local Testing

```bash
cd vercel-gat-api
pip install -r requirements.txt
python api/index.py
```

Buka: http://localhost:5000

## PHP Integration Example

```php
<?php
function predictWithGAT($irt_ability, $survey_confidence = 0.7) {
    $data = json_encode([
        'irt_ability' => $irt_ability,
        'survey_confidence' => $survey_confidence
    ]);
    
    $ch = curl_init('https://your-app.vercel.app/api/predict');
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    curl_close($ch);
    
    return json_decode($response, true);
}

// Usage
$result = predictWithGAT(0.5, 0.8);
echo "Predicted Level: " . $result['data']['predicted_level'];
?>
```

## Model Files

Jika Anda punya model yang sudah dilatih:
1. Copy file `enhanced_gat_complete.pth` ke folder `models/`
2. Model akan otomatis dimuat saat API start
3. Jika tidak ada, API akan menggunakan untrained model dengan fallback logic

## Environment Variables (Optional)

Bisa tambah di Vercel dashboard:
- `MODEL_PATH`: Path ke model file
- `DEBUG_MODE`: Enable debug mode
- `API_VERSION`: API version override

## Support

Model ini kompatibel 100% dengan notebook GAT yang sudah dibuat. Arsitektur dan feature engineering sama persis.