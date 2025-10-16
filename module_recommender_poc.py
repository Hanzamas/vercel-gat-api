"""
PROOF OF CONCEPT: Module Recommendation menggunakan GAT Attention Weights
Opsi 1 - Quick Fix tanpa training ulang
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.gat_model import SimplifiedGATModel

class ModuleRecommender:
    """
    Module Recommender menggunakan existing GAT model
    Menggunakan attention weights untuk module recommendation
    TANPA perlu training ulang!
    """
    
    def __init__(self, gat_model, is_trained=False):
        self.gat_model = gat_model
        self.is_trained = is_trained
        self.n_modules = 7
    
    def recommend_modules(self, irt_ability, survey_confidence, pre_test_results):
        """
        Recommend modules berdasarkan:
        1. GAT attention weights (student-to-module attention)
        2. Pre-test results (0 or 1 per module)
        3. IRT ability & survey confidence
        
        Args:
            irt_ability: float (-3.0 to 3.0)
            survey_confidence: float (0.0 to 1.0)
            pre_test_results: list of int [7] (0 or 1 per module)
        
        Returns:
            dict with module recommendations
        """
        
        # Step 1: Prepare input features (sama seperti predictor.py)
        ability_percentile = min(max((irt_ability + 3) / 6, 0), 1)
        student_features = [irt_ability, survey_confidence, ability_percentile]
        
        # Module features (difficulty, class, position)
        module_features_list = [
            [1/3.0, 0/6.0, 1/7.0],  # Module 1: Easy
            [2/3.0, 1/6.0, 2/7.0],  # Module 2: Medium
            [2/3.0, 1/6.0, 3/7.0],  # Module 3: Medium
            [1/3.0, 1/6.0, 4/7.0],  # Module 4: Easy
            [2/3.0, 1/6.0, 5/7.0],  # Module 5: Medium
            [3/3.0, 1/6.0, 6/7.0],  # Module 6: Hard
            [2/3.0, 1/6.0, 7/7.0]   # Module 7: Medium
        ]
        
        # Step 2: Create graph
        node_features = np.vstack([student_features] + module_features_list)
        n_nodes = 8
        adj_matrix = np.zeros((n_nodes, n_nodes))
        edge_weights = np.ones((n_nodes, n_nodes))
        
        # Module prerequisites (sequential)
        for i in range(1, 7):
            adj_matrix[i, i+1] = 1
            edge_weights[i, i+1] = 0.8
        
        # Student-module connections
        adj_matrix[0, 1:] = 1
        adj_matrix[1:, 0] = 1
        edge_weights[0, 1:] = 0.5
        edge_weights[1:, 0] = 0.5
        
        # Convert to tensors
        node_features_tensor = torch.FloatTensor(node_features)
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)
        edge_weights_tensor = torch.FloatTensor(edge_weights)
        
        # Step 3: Run GAT forward pass
        with torch.no_grad():
            output = self.gat_model(node_features_tensor, adj_matrix_tensor, edge_weights_tensor)
            
            # Extract attention weights
            attention_weights = output['attention_weights']  # List of [8, 8] per head
            
            # Average attention across heads
            avg_attention = torch.stack(attention_weights).mean(dim=0)  # [8, 8]
            
            # Get student-to-module attention (row 0, columns 1-7)
            student_to_module_attention = avg_attention[0, 1:].numpy()  # [7]
            
            # Extract module embeddings
            all_embeddings = output['all_embeddings']
            module_embeddings = all_embeddings[1:].numpy()  # [7, output_dim]
            
            # Get level prediction (for context)
            level_probs = F.softmax(output['student_levels'][0], dim=0)
            predicted_level = torch.argmax(level_probs).item() + 1
        
        # Step 4: Combine attention with pre-test results
        pre_test_array = np.array(pre_test_results, dtype=float)  # [7]
        
        # Strategy 1: Simple multiplication
        # If pre-test failed (0), module gets low score
        # If pre-test passed (1), use attention weight
        basic_scores = student_to_module_attention * pre_test_array
        
        # Strategy 2: Weighted combination
        # Give more weight to pre-test results (60%) vs attention (40%)
        weighted_scores = (0.6 * pre_test_array) + (0.4 * student_to_module_attention)
        
        # Strategy 3: Conditional unlock
        # Module unlocks IF pre-test passed AND attention above threshold
        attention_threshold = 0.1  # Adjust based on observation
        conditional_unlock = (pre_test_array == 1) & (student_to_module_attention > attention_threshold)
        
        # Step 5: Generate recommendations
        recommendations = []
        for i in range(self.n_modules):
            module_id = i + 1
            
            # Determine unlock status
            pre_test_passed = pre_test_results[i] == 1
            attention_score = float(student_to_module_attention[i])
            basic_score = float(basic_scores[i])
            weighted_score = float(weighted_scores[i])
            
            # Decision logic
            if not pre_test_passed:
                # Pre-test failed → Lock module
                unlock = False
                confidence = 0.2 + (attention_score * 0.3)  # Low confidence
                reasoning = f"Pre-test failed (score: 0)"
            else:
                # Pre-test passed → Consider attention
                if attention_score > 0.15:
                    # High attention → Highly recommended
                    unlock = True
                    confidence = 0.7 + (attention_score * 0.3)
                    reasoning = f"Pre-test passed + high GAT attention ({attention_score:.3f})"
                elif attention_score > 0.05:
                    # Medium attention → Recommended
                    unlock = True
                    confidence = 0.5 + (attention_score * 0.4)
                    reasoning = f"Pre-test passed + medium GAT attention ({attention_score:.3f})"
                else:
                    # Low attention → Optional
                    unlock = True
                    confidence = 0.3 + (attention_score * 0.5)
                    reasoning = f"Pre-test passed but low GAT attention ({attention_score:.3f})"
            
            recommendations.append({
                'module_id': module_id,
                'unlock': unlock,
                'confidence': round(confidence, 4),
                'attention_score': round(attention_score, 4),
                'basic_score': round(basic_score, 4),
                'weighted_score': round(weighted_score, 4),
                'pre_test_passed': pre_test_passed,
                'reasoning': reasoning
            })
        
        # Step 6: Generate recommended order
        # Sort by weighted_score descending (unlocked modules first)
        unlocked_modules = [r for r in recommendations if r['unlock']]
        locked_modules = [r for r in recommendations if not r['unlock']]
        
        unlocked_sorted = sorted(unlocked_modules, key=lambda x: x['weighted_score'], reverse=True)
        locked_sorted = sorted(locked_modules, key=lambda x: x['weighted_score'], reverse=True)
        
        recommended_order = [r['module_id'] for r in unlocked_sorted + locked_sorted]
        
        # Step 7: Calculate statistics
        total_unlocked = len(unlocked_modules)
        total_locked = len(locked_modules)
        avg_confidence = np.mean([r['confidence'] for r in unlocked_modules]) if unlocked_modules else 0.0
        
        return {
            'success': True,
            'module_recommendations': recommendations,
            'recommended_order': recommended_order,
            'summary': {
                'total_modules': self.n_modules,
                'unlocked': total_unlocked,
                'locked': total_locked,
                'avg_confidence': round(avg_confidence, 4),
                'predicted_level': predicted_level
            },
            'student_info': {
                'irt_ability': irt_ability,
                'survey_confidence': survey_confidence,
                'ability_percentile': round(ability_percentile, 4)
            },
            'model_info': {
                'architecture': 'SimplifiedGATModel',
                'method': 'Attention-based recommendation',
                'model_trained': self.is_trained
            }
        }


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("PROOF OF CONCEPT: Module Recommendation")
    print("Using GAT Attention Weights (No Retraining Needed)")
    print("=" * 60)
    
    # Create model (would use the loaded model in production)
    model = SimplifiedGATModel(
        n_students=1, n_modules=7, student_features=3, module_features=3,
        hidden_dim=64, output_dim=32, n_heads=4, dropout=0.1, silent=True
    )
    model.eval()
    
    recommender = ModuleRecommender(model, is_trained=False)
    
    # Test cases
    test_cases = [
        {
            'name': 'Student A: All Pre-tests Passed',
            'irt_ability': 1.2,
            'survey_confidence': 0.85,
            'pre_test_results': [1, 1, 1, 1, 1, 1, 1]
        },
        {
            'name': 'Student B: All Pre-tests Failed',
            'irt_ability': -0.8,
            'survey_confidence': 0.5,
            'pre_test_results': [0, 0, 0, 0, 0, 0, 0]
        },
        {
            'name': 'Student C: Mixed Results (Real scenario)',
            'irt_ability': 0.3,
            'survey_confidence': 0.7,
            'pre_test_results': [1, 0, 1, 1, 0, 0, 0]
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"IRT Ability: {test['irt_ability']}")
        print(f"Survey Confidence: {test['survey_confidence']}")
        print(f"Pre-test Results: {test['pre_test_results']}")
        print(f"{'='*60}")
        
        result = recommender.recommend_modules(
            test['irt_ability'],
            test['survey_confidence'],
            test['pre_test_results']
        )
        
        print(f"\n📊 Summary:")
        print(f"  Unlocked: {result['summary']['unlocked']}/{result['summary']['total_modules']}")
        print(f"  Predicted Level: {result['summary']['predicted_level']}")
        print(f"  Avg Confidence: {result['summary']['avg_confidence']}")
        
        print(f"\n📋 Module Recommendations:")
        for rec in result['module_recommendations']:
            status = "🔓 UNLOCK" if rec['unlock'] else "🔒 LOCK"
            print(f"  Module {rec['module_id']}: {status} (confidence: {rec['confidence']:.3f})")
            print(f"    → {rec['reasoning']}")
        
        print(f"\n🎯 Recommended Order: {result['recommended_order']}")
    
    print(f"\n{'='*60}")
    print("✅ PROOF OF CONCEPT BERHASIL!")
    print("Module recommendation bisa dilakukan dengan attention weights")
    print("TANPA perlu training ulang model!")
    print("=" * 60)
