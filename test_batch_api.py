#!/usr/bin/env python3
"""
Test script untuk batch GAT API
Mensimulasikan data dari PHP IRT processing
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"  # Change to your Vercel URL when deployed
BATCH_ENDPOINT = f"{API_BASE_URL}/predict-batch"

def test_batch_prediction():
    """Test batch prediction dengan simulasi data dari PHP"""
    
    # Simulasi data setelah IRT processing di PHP
    # Format: student_id, irt_ability, survey_confidence
    test_students = [
        {
            "student_id": "101",
            "irt_ability": -0.5,
            "survey_confidence": 0.6
        },
        {
            "student_id": "102", 
            "irt_ability": 0.3,
            "survey_confidence": 0.8
        },
        {
            "student_id": "103",
            "irt_ability": 1.2,
            "survey_confidence": 0.7
        },
        {
            "student_id": "104",
            "irt_ability": -1.1,
            "survey_confidence": 0.5
        },
        {
            "student_id": "105",
            "irt_ability": 0.8,
            "survey_confidence": 0.9
        }
    ]
    
    # Prepare batch request
    batch_data = {
        "students": test_students
    }
    
    print("🧪 Testing Batch GAT Prediction API")
    print("=" * 50)
    print(f"📤 Sending {len(test_students)} students to API...")
    print(f"🔗 Endpoint: {BATCH_ENDPOINT}")
    print()
    
    try:
        # Send batch request
        start_time = time.time()
        response = requests.post(
            BATCH_ENDPOINT,
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        end_time = time.time()
        
        print(f"⏱️ Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS - Batch Prediction Results:")
            print("=" * 50)
            print(f"📈 Total Students: {result['data']['total_students']}")
            print(f"✅ Successful: {result['data']['successful_predictions']}")
            print(f"❌ Failed: {result['data']['failed_predictions']}")
            print()
            
            # Display results for each student
            print("📋 Individual Results:")
            print("-" * 30)
            for i, student_result in enumerate(result['data']['results'], 1):
                print(f"{i}. Student ID: {student_result['student_id']}")
                print(f"   Level: {student_result['predicted_level']}")
                print(f"   Confidence: {student_result['confidence']:.4f}")
                print(f"   Success: {student_result.get('success', 'N/A')}")
                if 'features' in student_result:
                    print(f"   IRT Ability: {student_result['features']['irt_ability']:.4f}")
                    print(f"   Ability Percentile: {student_result['features']['ability_percentile']:.4f}")
                if 'error' in student_result:
                    print(f"   ⚠️ Error: {student_result['error']}")
                print()
            
            # Display errors if any
            if result['data']['errors']:
                print("⚠️ Errors:")
                print("-" * 20)
                for error in result['data']['errors']:
                    print(f"Student {error['student_id']}: {error['error']}")
                print()
            
            # Show format for PHP processing
            print("🔧 PHP Integration Format:")
            print("-" * 30)
            print("// Loop through results in PHP:")
            print("foreach ($api_response['data']['results'] as $student_result) {")
            print("    $student_id = $student_result['student_id'];")
            print("    $level_pre_test = $student_result['predicted_level'];")
            print("    // Insert to database...")
            print("}")
            
        else:
            print("❌ ERROR - API Request Failed:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

def test_invalid_batch():
    """Test batch dengan data invalid"""
    print("\n🧪 Testing Invalid Batch Data")
    print("=" * 30)
    
    # Test dengan data invalid
    invalid_data = {
        "students": [
            {
                "student_id": "999",
                "irt_ability": 10.0,  # Out of range
                "survey_confidence": 0.7
            },
            {
                "student_id": "998",
                "irt_ability": "invalid",  # Wrong type
                "survey_confidence": 0.6
            }
        ]
    }
    
    try:
        response = requests.post(
            BATCH_ENDPOINT,
            json=invalid_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        result = response.json()
        print(f"Status: {response.status_code}")
        print(f"Successful: {result['data']['successful_predictions']}")
        print(f"Failed: {result['data']['failed_predictions']}")
        print("\nErrors:")
        for error in result['data']['errors']:
            print(f"- Student {error['student_id']}: {error['error']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test valid batch
    test_batch_prediction()
    
    # Test invalid batch
    test_invalid_batch()
    
    print("\n" + "=" * 50)
    print("🎯 Ready for PHP Integration!")