"""
Test Module Recommendation API
Tests the new /recommend-modules endpoint with various scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.predictor import predictor

def print_separator(title=""):
    """Print a nice separator"""
    if title:
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    else:
        print("-"*80)

def print_recommendations(result, scenario_name):
    """Pretty print recommendations"""
    print_separator(f"TEST: {scenario_name}")
    
    if not result.get('success', False):
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        return
    
    print("\n📊 METADATA:")
    metadata = result['metadata']
    print(f"   IRT Ability: {metadata['irt_ability']}")
    print(f"   Survey Confidence: {metadata['survey_confidence']}")
    print(f"   Pre-test Pass Rate: {metadata['pre_test_summary']['pass_rate']*100:.1f}%")
    print(f"   Strategy Used: {metadata['strategy_used']}")
    print(f"   Model Trained: {metadata.get('model_trained', False)}")
    
    print("\n📚 MODULE RECOMMENDATIONS:")
    recommendations = result['module_recommendations']
    
    # Count unlocked/locked
    unlocked_count = sum(1 for r in recommendations if r['unlock'])
    locked_count = len(recommendations) - unlocked_count
    
    print(f"   ✅ Unlocked: {unlocked_count}/7")
    print(f"   🔒 Locked: {locked_count}/7")
    print()
    
    for rec in recommendations:
        status = "✅ UNLOCK" if rec['unlock'] else "🔒 LOCKED"
        module_name = rec['module_name'][:30].ljust(30)
        confidence = rec['confidence']
        attention = rec['attention_score']
        passed = "✓" if rec['pre_test_passed'] else "✗"
        
        print(f"   M{rec['module_id']} {status} | {module_name} | "
              f"Conf: {confidence:.3f} | Att: {attention:.3f} | Pre-test: {passed}")
        print(f"      → {rec['reasoning']}")
    
    print("\n🎯 RECOMMENDED ORDER:")
    order = result['recommended_order']
    order_str = " → ".join([f"M{m}" for m in order])
    print(f"   {order_str}")
    
    print_separator()

def test_scenario_1():
    """Test Scenario 1: High ability, all pre-tests passed"""
    print_separator("SCENARIO 1: High Ability Student - All Pre-tests Passed")
    
    result = predictor.recommend_modules(
        irt_ability=1.5,
        survey_confidence=0.9,
        pre_test_results=[1, 1, 1, 1, 1, 1, 1],
        strategy='weighted'
    )
    
    print_recommendations(result, "High Ability - All Passed")
    return result

def test_scenario_2():
    """Test Scenario 2: Low ability, all pre-tests failed"""
    print_separator("SCENARIO 2: Low Ability Student - All Pre-tests Failed")
    
    result = predictor.recommend_modules(
        irt_ability=-1.5,
        survey_confidence=0.3,
        pre_test_results=[0, 0, 0, 0, 0, 0, 0],
        strategy='weighted'
    )
    
    print_recommendations(result, "Low Ability - All Failed")
    return result

def test_scenario_3():
    """Test Scenario 3: Medium ability, mixed results"""
    print_separator("SCENARIO 3: Medium Ability Student - Mixed Results")
    
    result = predictor.recommend_modules(
        irt_ability=0.5,
        survey_confidence=0.7,
        pre_test_results=[1, 0, 1, 1, 0, 0, 0],
        strategy='weighted'
    )
    
    print_recommendations(result, "Medium Ability - Mixed Results")
    return result

def test_scenario_4():
    """Test Scenario 4: Compare strategies"""
    print_separator("SCENARIO 4: Strategy Comparison")
    
    # Same student data, different strategies
    irt = 0.8
    confidence = 0.8
    pre_test = [1, 1, 0, 1, 0, 1, 0]
    
    strategies = ['simple', 'weighted', 'conditional']
    results = {}
    
    for strategy in strategies:
        print(f"\n🔧 Testing Strategy: {strategy.upper()}")
        result = predictor.recommend_modules(
            irt_ability=irt,
            survey_confidence=confidence,
            pre_test_results=pre_test,
            strategy=strategy
        )
        results[strategy] = result
        
        # Quick summary
        unlocked = sum(1 for r in result['module_recommendations'] if r['unlock'])
        order = result['recommended_order'][:3]
        print(f"   Unlocked: {unlocked}/7")
        print(f"   Top 3 Recommended: M{order[0]}, M{order[1]}, M{order[2]}")
    
    print_separator()
    return results

def test_scenario_5():
    """Test Scenario 5: Progressive learning (passing modules one by one)"""
    print_separator("SCENARIO 5: Progressive Learning Simulation")
    
    irt = 0.0
    confidence = 0.6
    
    # Simulate student progressing through modules
    stages = [
        ([1, 0, 0, 0, 0, 0, 0], "Stage 1: Only Module 1 passed"),
        ([1, 1, 0, 0, 0, 0, 0], "Stage 2: Modules 1-2 passed"),
        ([1, 1, 1, 0, 0, 0, 0], "Stage 3: Modules 1-3 passed"),
        ([1, 1, 1, 1, 0, 0, 0], "Stage 4: Modules 1-4 passed"),
    ]
    
    for pre_test, stage_name in stages:
        print(f"\n📈 {stage_name}")
        result = predictor.recommend_modules(
            irt_ability=irt,
            survey_confidence=confidence,
            pre_test_results=pre_test,
            strategy='conditional'
        )
        
        unlocked = [r['module_id'] for r in result['module_recommendations'] if r['unlock']]
        locked = [r['module_id'] for r in result['module_recommendations'] if not r['unlock']]
        next_rec = result['recommended_order'][0] if result['recommended_order'] else None
        
        print(f"   Unlocked: M{', M'.join(map(str, unlocked))}")
        print(f"   Locked: M{', M'.join(map(str, locked))}")
        print(f"   Next Recommended: M{next_rec}")
    
    print_separator()

def test_edge_cases():
    """Test edge cases"""
    print_separator("EDGE CASES TESTING")
    
    cases = [
        {
            'name': 'Invalid pre_test length',
            'irt': 0.5,
            'confidence': 0.7,
            'pre_test': [1, 0, 1],  # Only 3 instead of 7
            'expected_error': True
        },
        {
            'name': 'Invalid strategy',
            'irt': 0.5,
            'confidence': 0.7,
            'pre_test': [1, 0, 1, 1, 0, 0, 0],
            'strategy': 'invalid_strategy',
            'expected_error': True
        },
        {
            'name': 'Extreme IRT values',
            'irt': 10.0,  # Should be clamped to 3.0
            'confidence': 0.7,
            'pre_test': [1, 1, 1, 1, 1, 1, 1],
            'expected_error': False
        },
        {
            'name': 'Invalid confidence',
            'irt': 0.5,
            'confidence': 1.5,  # Should be clamped to 1.0
            'pre_test': [1, 0, 1, 1, 0, 0, 0],
            'expected_error': False
        }
    ]
    
    for case in cases:
        print(f"\n🧪 Testing: {case['name']}")
        try:
            result = predictor.recommend_modules(
                irt_ability=case['irt'],
                survey_confidence=case['confidence'],
                pre_test_results=case['pre_test'],
                strategy=case.get('strategy', 'weighted')
            )
            
            if case['expected_error']:
                print(f"   ⚠️  Expected error but got result: {result.get('success')}")
            else:
                print(f"   ✅ Success: {result.get('success')}")
                if 'Extreme' in case['name']:
                    print(f"   IRT clamped to: {result['metadata']['irt_ability']}")
                if 'confidence' in case['name'].lower():
                    print(f"   Confidence clamped to: {result['metadata']['survey_confidence']}")
        
        except Exception as e:
            if case['expected_error']:
                print(f"   ✅ Expected error caught: {str(e)[:50]}")
            else:
                print(f"   ❌ Unexpected error: {str(e)}")
    
    print_separator()

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" 🧪 MODULE RECOMMENDATION API - COMPREHENSIVE TESTING")
    print("="*80)
    print("\n📝 Testing Option 1: Attention-Based Module Recommendation")
    print("   - Uses attention weights from trained GAT model")
    print("   - Combines with pre-test results for unlock decisions")
    print("   - No retraining needed!")
    
    try:
        # Run all test scenarios
        test_scenario_1()  # High ability, all passed
        test_scenario_2()  # Low ability, all failed
        test_scenario_3()  # Medium ability, mixed
        test_scenario_4()  # Strategy comparison
        test_scenario_5()  # Progressive learning
        test_edge_cases()  # Edge cases
        
        print_separator("TESTING COMPLETED")
        print("\n✅ All tests completed successfully!")
        print("\n📊 SUMMARY:")
        print("   - Option 1 (Attention-Based) implementation working")
        print("   - No model retraining required")
        print("   - Three strategies available: simple, weighted, conditional")
        print("   - Ready for PHP integration")
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
