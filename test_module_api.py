"""
Test Module Recommendation API
Testing GAT attention-based module recommendations
"""

import requests
import json

# API URL (change to your deployed URL)
API_URL = "http://localhost:5000"  # Local testing
# API_URL = "https://your-app.vercel.app"  # Production

def test_module_recommendations():
    """Test module recommendation endpoint"""
    print("="*80)
    print("🧪 Testing Module Recommendation API")
    print("="*80)
    
    # Test Case 1: Low ability student (should recommend easy modules)
    print("\n📝 Test 1: Low Ability Student (IRT = -1.0)")
    print("-"*80)
    
    data1 = {
        "irt_ability": -1.0,
        "survey_confidence": 0.6
    }
    
    response1 = requests.post(f"{API_URL}/predict-modules", json=data1)
    result1 = response1.json()
    
    print(f"Status: {result1['status']}")
    if result1['status'] == 'success':
        data = result1['data']
        print(f"Predicted Level: {data['predicted_level']}")
        print(f"Confidence: {data['confidence']:.2%}")
        print(f"\n📚 Module Recommendations:")
        print("-"*80)
        
        for module in data['module_recommendations']:
            print(f"Module {module['module_id']}: {module['module_name']}")
            print(f"  Attention Score: {module['attention_score']:.4f}")
            print(f"  Status: {module['unlock_status']}")
            print(f"  Reason: {module['unlock_reason']}")
            print(f"  Difficulty: {module['difficulty']}")
            print(f"  Recommended Order: {module['recommended_order']}")
            print()
        
        print(f"Top 3 Recommended: {data['top_3_modules']}")
    
    # Test Case 2: High ability student (should recommend harder modules)
    print("\n📝 Test 2: High Ability Student (IRT = 1.5)")
    print("-"*80)
    
    data2 = {
        "irt_ability": 1.5,
        "survey_confidence": 0.9
    }
    
    response2 = requests.post(f"{API_URL}/predict-modules", json=data2)
    result2 = response2.json()
    
    if result2['status'] == 'success':
        data = result2['data']
        print(f"Predicted Level: {data['predicted_level']}")
        print(f"Top 3 Recommended: {data['top_3_modules']}")
    
    # Test Case 3: With module progress (unlock based on correct/wrong)
    print("\n📝 Test 3: Student with Module Progress")
    print("-"*80)
    
    data3 = {
        "irt_ability": 0.3,
        "survey_confidence": 0.7,
        "module_progress": {
            "1": {"correct": 8, "wrong": 2},  # Module 1: Passed (80% accuracy)
            "2": {"correct": 3, "wrong": 7},  # Module 2: Failed (30% accuracy)
            "3": {"correct": 6, "wrong": 4}   # Module 3: Passed (60% accuracy)
        }
    }
    
    response3 = requests.post(f"{API_URL}/predict-modules", json=data3)
    result3 = response3.json()
    
    if result3['status'] == 'success':
        data = result3['data']
        print(f"Predicted Level: {data['predicted_level']}")
        print(f"\n📊 Module Status with Progress:")
        print("-"*80)
        
        for module in data['module_recommendations']:
            if module['module_id'] <= 3:  # Show first 3 modules with progress
                print(f"Module {module['module_id']}: {module['unlock_status'].upper()}")
                print(f"  {module['unlock_reason']}")
                print()

def test_comparison():
    """Compare regular prediction vs module recommendation"""
    print("\n"+"="*80)
    print("🔄 Comparison: Regular vs Module Recommendation")
    print("="*80)
    
    test_data = {
        "irt_ability": 0.5,
        "survey_confidence": 0.8
    }
    
    # Regular prediction
    print("\n1️⃣  Regular Prediction (Level Only):")
    print("-"*80)
    response1 = requests.post(f"{API_URL}/predict", json=test_data)
    result1 = response1.json()
    
    if result1['status'] == 'success':
        print(json.dumps(result1['data'], indent=2))
    
    # Module recommendation
    print("\n2️⃣  Module Recommendation (with Attention Weights):")
    print("-"*80)
    response2 = requests.post(f"{API_URL}/predict-modules", json=test_data)
    result2 = response2.json()
    
    if result2['status'] == 'success':
        data = result2['data']
        print(f"Level: {data['predicted_level']}")
        print(f"Top Modules: {data['top_3_modules']}")
        print("\nAttention Distribution:")
        for m in sorted(data['module_recommendations'], key=lambda x: x['attention_score'], reverse=True):
            bar = "█" * int(m['attention_score'] * 50)
            print(f"  Module {m['module_id']}: {bar} {m['attention_score']:.4f}")

def test_unlock_logic():
    """Test unlock logic based on correct/wrong answers"""
    print("\n"+"="*80)
    print("🔐 Testing Unlock Logic")
    print("="*80)
    
    scenarios = [
        {
            "name": "All Correct",
            "progress": {"1": {"correct": 10, "wrong": 0}}
        },
        {
            "name": "Mostly Correct",
            "progress": {"1": {"correct": 7, "wrong": 3}}
        },
        {
            "name": "50-50",
            "progress": {"1": {"correct": 5, "wrong": 5}}
        },
        {
            "name": "Mostly Wrong",
            "progress": {"1": {"correct": 3, "wrong": 7}}
        },
        {
            "name": "All Wrong",
            "progress": {"1": {"correct": 0, "wrong": 10}}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print("-"*40)
        
        data = {
            "irt_ability": 0.0,
            "survey_confidence": 0.7,
            "module_progress": scenario['progress']
        }
        
        response = requests.post(f"{API_URL}/predict-modules", json=data)
        result = response.json()
        
        if result['status'] == 'success':
            module1 = result['data']['module_recommendations'][0]  # First module
            print(f"Module 1 Status: {module1['unlock_status']}")
            print(f"Reason: {module1['unlock_reason']}")

if __name__ == "__main__":
    try:
        # Run all tests
        test_module_recommendations()
        test_comparison()
        test_unlock_logic()
        
        print("\n"+"="*80)
        print("✅ All tests completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print(f"Make sure the API is running at: {API_URL}")
        print("\nTo start the API locally:")
        print("  cd vercel-gat-api")
        print("  python api/index.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
