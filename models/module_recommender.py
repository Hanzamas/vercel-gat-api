"""
Module Recommender using Attention Weights from GAT Model
Option 1: Attention-Based Approach (No Retraining Needed)

This module extracts attention weights from the trained GAT model
and combines them with pre-test results to generate module recommendations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class ModuleRecommender:
    """
    Recommends modules based on:
    1. Pre-test results (binary per module)
    2. Student-module attention weights from GAT
    3. Student features (IRT ability, survey confidence)
    """
    
    def __init__(self, gat_model, model_config):
        """
        Initialize ModuleRecommender with trained GAT model
        
        Args:
            gat_model: Trained SimplifiedGATModel instance
            model_config: Model configuration dict
        """
        self.model = gat_model
        self.config = model_config
        self.n_modules = 7
        
        # Module metadata
        self.module_info = {
            1: {'name': 'Pengenalan Sel', 'difficulty': 1, 'prerequisites': []},
            2: {'name': 'Komponen Kimia Sel', 'difficulty': 2, 'prerequisites': [1]},
            3: {'name': 'Struktur Sel Pro & Eukariotik', 'difficulty': 2, 'prerequisites': [1, 2]},
            4: {'name': 'Perbedaan Sel Hewan & Tumbuhan', 'difficulty': 1, 'prerequisites': [3]},
            5: {'name': 'Transpor Membran', 'difficulty': 2, 'prerequisites': [3]},
            6: {'name': 'Sintesis Protein', 'difficulty': 3, 'prerequisites': [5]},
            7: {'name': 'Reproduksi Sel', 'difficulty': 2, 'prerequisites': [5]}
        }
        
        # Thresholds for decision making
        self.attention_threshold = 0.4  # Minimum attention for unlock
        self.confidence_threshold = 0.3  # Minimum confidence for recommendation
        
    def recommend_modules(
        self, 
        irt_ability: float, 
        survey_confidence: float, 
        pre_test_results: List[int],
        strategy: str = 'weighted'
    ) -> Dict:
        """
        Generate module recommendations using attention weights
        
        Args:
            irt_ability: Student IRT ability score (-3 to 3)
            survey_confidence: Student self-assessment confidence (0 to 1)
            pre_test_results: Binary list [7] indicating pass (1) or fail (0) per module
            strategy: 'simple', 'weighted', or 'conditional' (default: 'weighted')
        
        Returns:
            Dict containing module_recommendations, recommended_order, and metadata
        """
        
        # Validate inputs
        if len(pre_test_results) != self.n_modules:
            raise ValueError(f"pre_test_results must have {self.n_modules} elements")
        
        # Get attention weights from model
        attention_scores = self._extract_attention_weights(
            irt_ability, 
            survey_confidence
        )
        
        # Generate recommendations based on strategy
        if strategy == 'simple':
            recommendations = self._simple_strategy(pre_test_results, attention_scores)
        elif strategy == 'weighted':
            recommendations = self._weighted_strategy(
                pre_test_results, 
                attention_scores, 
                irt_ability, 
                survey_confidence
            )
        elif strategy == 'conditional':
            recommendations = self._conditional_strategy(
                pre_test_results, 
                attention_scores, 
                irt_ability
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Generate recommended order
        recommended_order = self._generate_order(recommendations)
        
        return {
            'module_recommendations': recommendations,
            'recommended_order': recommended_order,
            'metadata': {
                'irt_ability': round(irt_ability, 4),
                'survey_confidence': round(survey_confidence, 4),
                'pre_test_summary': {
                    'total_passed': sum(pre_test_results),
                    'total_failed': self.n_modules - sum(pre_test_results),
                    'pass_rate': round(sum(pre_test_results) / self.n_modules, 4)
                },
                'strategy_used': strategy,
                'attention_threshold': self.attention_threshold,
                'model_trained': hasattr(self.model, 'training') and not self.model.training
            }
        }
    
    def _extract_attention_weights(
        self, 
        irt_ability: float, 
        survey_confidence: float
    ) -> np.ndarray:
        """
        Extract student-module attention weights from GAT model
        
        Returns:
            np.ndarray: Attention scores for 7 modules [7]
        """
        
        # Create student features (same as predictor.py)
        ability_percentile = min(max((irt_ability + 3) / 6, 0), 1)
        student_features = [irt_ability, survey_confidence, ability_percentile]
        
        # Create module features (same as predictor.py)
        module_features_list = [
            [1/3.0, 0/6.0, 1/7.0],  # Module 1: Easy
            [2/3.0, 1/6.0, 2/7.0],  # Module 2: Medium
            [2/3.0, 1/6.0, 3/7.0],  # Module 3: Medium
            [1/3.0, 1/6.0, 4/7.0],  # Module 4: Easy
            [2/3.0, 1/6.0, 5/7.0],  # Module 5: Medium
            [3/3.0, 1/6.0, 6/7.0],  # Module 6: Hard
            [2/3.0, 1/6.0, 7/7.0]   # Module 7: Medium
        ]
        
        # Create graph structure
        node_features = np.vstack([student_features] + module_features_list)
        
        # Create adjacency matrix (same as predictor.py)
        n_nodes = 8
        adj_matrix = np.zeros((n_nodes, n_nodes))
        edge_weights = np.ones((n_nodes, n_nodes))
        
        # Module prerequisites
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
        
        # Get model output with attention weights
        with torch.no_grad():
            output = self.model(
                node_features_tensor, 
                adj_matrix_tensor, 
                edge_weights_tensor
            )
            
            # Extract attention weights
            # attention_weights is returned as list from GAT model
            attention_weights = output['attention_weights']
            
            # Convert list of [N, N] tensors to [n_heads, N, N] tensor
            if isinstance(attention_weights, list):
                attention_weights = torch.stack(attention_weights, dim=0)
            
            # Extract student (node 0) to modules (nodes 1-7) attention
            # Average across all attention heads
            student_module_attention = attention_weights[:, 0, 1:8]  # [n_heads, 7]
            avg_attention = student_module_attention.mean(dim=0)  # [7]
            
            # Normalize to 0-1 range
            attention_scores = F.softmax(avg_attention, dim=0).cpu().numpy()
            
        return attention_scores
    
    def _simple_strategy(
        self, 
        pre_test_results: List[int], 
        attention_scores: np.ndarray
    ) -> List[Dict]:
        """
        Simple strategy: Unlock if passed pre-test AND attention > threshold
        """
        recommendations = []
        
        for i in range(self.n_modules):
            module_id = i + 1
            passed_pretest = pre_test_results[i] == 1
            attention = float(attention_scores[i])
            
            # Decision: Unlock if passed AND attention is high
            unlock = passed_pretest and (attention > self.attention_threshold)
            
            # Confidence is just the attention score
            confidence = attention
            
            # Reasoning
            if passed_pretest and unlock:
                reasoning = f"Passed pre-test (attention: {attention:.2f})"
            elif passed_pretest and not unlock:
                reasoning = f"Passed pre-test but low attention ({attention:.2f})"
            else:
                reasoning = "Failed pre-test"
            
            recommendations.append({
                'module_id': module_id,
                'module_name': self.module_info[module_id]['name'],
                'unlock': unlock,
                'confidence': round(confidence, 4),
                'attention_score': round(attention, 4),
                'pre_test_passed': passed_pretest,
                'reasoning': reasoning
            })
        
        return recommendations
    
    def _weighted_strategy(
        self, 
        pre_test_results: List[int], 
        attention_scores: np.ndarray,
        irt_ability: float,
        survey_confidence: float
    ) -> List[Dict]:
        """
        Weighted strategy: Combine attention, pre-test, IRT, and confidence
        """
        recommendations = []
        
        # Normalize IRT to 0-1
        irt_normalized = (irt_ability + 3) / 6
        
        for i in range(self.n_modules):
            module_id = i + 1
            passed_pretest = pre_test_results[i] == 1
            attention = float(attention_scores[i])
            difficulty = self.module_info[module_id]['difficulty']
            
            # Calculate weighted score
            weights = {
                'pre_test': 0.4,      # 40% weight on pre-test result
                'attention': 0.3,     # 30% weight on attention score
                'irt_match': 0.2,     # 20% weight on IRT-difficulty match
                'confidence': 0.1     # 10% weight on self-confidence
            }
            
            # IRT-difficulty match (higher score if IRT matches difficulty)
            irt_match = 1.0 - abs((difficulty / 3.0) - irt_normalized)
            
            # Weighted score
            weighted_score = (
                weights['pre_test'] * (1.0 if passed_pretest else 0.0) +
                weights['attention'] * attention +
                weights['irt_match'] * irt_match +
                weights['confidence'] * survey_confidence
            )
            
            # Decision: Unlock if weighted score > threshold
            unlock = weighted_score > 0.5 and passed_pretest
            
            # Confidence is the weighted score
            confidence = weighted_score
            
            # Reasoning
            if unlock:
                reasoning = f"Strong match (score: {weighted_score:.2f}, attention: {attention:.2f})"
            elif passed_pretest:
                reasoning = f"Passed pre-test but weak match (score: {weighted_score:.2f})"
            else:
                reasoning = "Failed pre-test - needs review"
            
            recommendations.append({
                'module_id': module_id,
                'module_name': self.module_info[module_id]['name'],
                'unlock': unlock,
                'confidence': round(confidence, 4),
                'attention_score': round(attention, 4),
                'pre_test_passed': passed_pretest,
                'weighted_score': round(weighted_score, 4),
                'difficulty': difficulty,
                'reasoning': reasoning
            })
        
        return recommendations
    
    def _conditional_strategy(
        self, 
        pre_test_results: List[int], 
        attention_scores: np.ndarray,
        irt_ability: float
    ) -> List[Dict]:
        """
        Conditional strategy: Consider prerequisites and IRT level
        """
        recommendations = []
        unlocked_modules = set()
        
        for i in range(self.n_modules):
            module_id = i + 1
            passed_pretest = pre_test_results[i] == 1
            attention = float(attention_scores[i])
            difficulty = self.module_info[module_id]['difficulty']
            prerequisites = self.module_info[module_id]['prerequisites']
            
            # Check if prerequisites are met
            prereqs_met = all(prereq in unlocked_modules for prereq in prerequisites)
            
            # Check if IRT level is appropriate for difficulty
            irt_appropriate = self._check_irt_level(irt_ability, difficulty)
            
            # Decision logic
            unlock = False
            confidence = 0.0
            reasoning = ""
            
            if not passed_pretest:
                unlock = False
                confidence = 0.0
                reasoning = "Failed pre-test - must review and retake"
            elif not prereqs_met and prerequisites:
                unlock = False
                confidence = attention * 0.5
                prereq_names = [str(p) for p in prerequisites]
                reasoning = f"Prerequisites not met (needs: M{', M'.join(prereq_names)})"
            elif not irt_appropriate:
                unlock = False
                confidence = attention * 0.7
                reasoning = f"Difficulty mismatch (module: {difficulty}, IRT suggests different level)"
            elif attention < self.attention_threshold:
                unlock = False
                confidence = attention
                reasoning = f"Low attention score ({attention:.2f}) - may need more preparation"
            else:
                unlock = True
                confidence = min(attention * 1.2, 1.0)  # Boost confidence for unlocked
                reasoning = f"All conditions met (attention: {attention:.2f})"
                unlocked_modules.add(module_id)
            
            recommendations.append({
                'module_id': module_id,
                'module_name': self.module_info[module_id]['name'],
                'unlock': unlock,
                'confidence': round(confidence, 4),
                'attention_score': round(attention, 4),
                'pre_test_passed': passed_pretest,
                'prerequisites_met': prereqs_met,
                'irt_appropriate': irt_appropriate,
                'difficulty': difficulty,
                'reasoning': reasoning
            })
        
        return recommendations
    
    def _check_irt_level(self, irt_ability: float, difficulty: int) -> bool:
        """
        Check if student's IRT level is appropriate for module difficulty
        
        IRT mapping:
        - Level 1 (Easy): IRT < 0.0
        - Level 2 (Medium): IRT 0.0 to 1.0
        - Level 3 (Hard): IRT > 1.0
        """
        if difficulty == 1:
            return irt_ability >= -2.0  # Easy modules for most students
        elif difficulty == 2:
            return irt_ability >= -0.5  # Medium modules need some ability
        else:  # difficulty == 3
            return irt_ability >= 0.5   # Hard modules need high ability
    
    def _generate_order(self, recommendations: List[Dict]) -> List[int]:
        """
        Generate recommended order based on unlock status and confidence
        
        Priority:
        1. Unlocked modules (sorted by confidence desc)
        2. Locked modules (sorted by how close they are to unlocking)
        """
        unlocked = [r for r in recommendations if r['unlock']]
        locked = [r for r in recommendations if not r['unlock']]
        
        # Sort unlocked by confidence (descending)
        unlocked_sorted = sorted(
            unlocked, 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        # Sort locked by attention score (descending) - closest to unlocking
        locked_sorted = sorted(
            locked,
            key=lambda x: x['attention_score'],
            reverse=True
        )
        
        # Combine: unlocked first, then locked
        ordered_modules = unlocked_sorted + locked_sorted
        
        return [m['module_id'] for m in ordered_modules]
    
    def get_module_info(self) -> Dict:
        """Get metadata about all modules"""
        return self.module_info
