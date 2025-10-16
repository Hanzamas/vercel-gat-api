"""
Quick Test Script - Module Recommendation API
Run this for quick validation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.predictor import predictor

def quick_test():
    """Quick sanity check"""
    print("\n" + "="*60)
    print(" 🧪 QUICK TEST: Module Recommendation API")
    print("="*60)
    
    # Test data
    test_cases = [
        {
            'name': 'All Passed',
            'irt': 1.0,
            'confidence': 0.8,
            'pre_test': [1, 1, 1, 1, 1, 1, 1]
        },
        {
            'name': 'All Failed',
            'irt': -1.0,
            'confidence': 0.4,
            'pre_test': [0, 0, 0, 0, 0, 0, 0]
        },
        {
            'name': 'Mixed',
            'irt': 0.5,
            'confidence': 0.7,
            'pre_test': [1, 0, 1, 1, 0, 0, 0]
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 Test: {case['name']}")
        print(f"   Input: IRT={case['irt']}, Conf={case['confidence']}, Pre-test={case['pre_test']}")
        
        try:
            result = predictor.recommend_modules(
                irt_ability=case['irt'],
                survey_confidence=case['confidence'],
                pre_test_results=case['pre_test'],
                strategy='weighted'
            )
            
            if result.get('success'):
                unlocked = sum(1 for r in result['module_recommendations'] if r['unlock'])
                order = result['recommended_order'][:3]
                print(f"   ✅ Success: {unlocked}/7 unlocked, Top 3: M{order[0]}, M{order[1]}, M{order[2]}")
            else:
                print(f"   ⚠️  Success=False: {result.get('error', 'Unknown')}")
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:60]}")
    
    print("\n" + "="*60)
    print(" ✅ Quick test completed!")
    print("="*60 + "\n")

if __name__ == '__main__':
    quick_test()
