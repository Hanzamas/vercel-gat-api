"""
Local test script untuk GAT API
Test API functionality sebelum deploy ke Vercel
"""

import requests
import json

# API base URL (change to your deployed URL)
BASE_URL = "http://localhost:5000"
# BASE_URL = "https://your-app.vercel.app"

def test_api():
    print("🧪 Testing GAT Prediction API")
    print("=" * 50)
    
    # Test 1: Home endpoint
    print("\n1. Testing Home endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Home endpoint working")
        else:
            print("   ❌ Home endpoint failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing Health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Model trained: {data.get('model_trained', False)}")
            print("   ✅ Health check passed")
        else:
            print("   ❌ Health check failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing Model Info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/model-info")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', {})
            print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"   Features: {model_info.get('features', 'Unknown')}")
            print("   ✅ Model info retrieved")
        else:
            print("   ❌ Model info failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Predictions
    print("\n4. Testing Predictions...")
    test_cases = [
        {"irt_ability": 0.5, "survey_confidence": 0.8, "name": "Average student"},
        {"irt_ability": -1.0, "survey_confidence": 0.3, "name": "Low ability student"},
        {"irt_ability": 1.5, "survey_confidence": 0.9, "name": "High ability student"},
        {"irt_ability": 0.0, "survey_confidence": 0.7, "name": "Neutral student"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}: {test_case['name']}")
        try:
            payload = {
                "irt_ability": test_case["irt_ability"],
                "survey_confidence": test_case["survey_confidence"]
            }
            
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    result = data.get('data', {})
                    print(f"   Input: IRT={payload['irt_ability']}, Confidence={payload['survey_confidence']}")
                    print(f"   Predicted Level: {result.get('predicted_level', 'N/A')}")
                    print(f"   Confidence: {result.get('confidence', 'N/A')}")
                    print(f"   Model Trained: {result.get('model_trained', False)}")
                    print("   ✅ Prediction successful")
                else:
                    print(f"   ❌ Prediction failed: {data.get('message', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('message', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 5: Error handling
    print("\n5. Testing Error Handling...")
    error_cases = [
        {"payload": {}, "name": "Missing irt_ability"},
        {"payload": {"irt_ability": "invalid"}, "name": "Invalid irt_ability type"},
        {"payload": {"irt_ability": 10.0}, "name": "Out of range irt_ability"},
        {"payload": {"irt_ability": 0.5, "survey_confidence": 2.0}, "name": "Out of range survey_confidence"}
    ]
    
    for i, error_case in enumerate(error_cases, 1):
        print(f"\n   Error Test {i}: {error_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=error_case["payload"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 400:
                data = response.json()
                print(f"   Error Code: {data.get('code', 'N/A')}")
                print(f"   Message: {data.get('message', 'N/A')}")
                print("   ✅ Error handling working")
            else:
                print(f"   ❌ Unexpected status code: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 API Testing Complete!")

if __name__ == "__main__":
    test_api()