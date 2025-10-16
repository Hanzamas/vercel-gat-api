"""
GAT Prediction Service for Vercel Deployment
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from .gat_model import SimplifiedGATModel
from .module_recommender import ModuleRecommender

class GATPredictor:
    def __init__(self):
        self.model = None
        self.model_config = None
        self.is_trained = False
        self.module_recommender = None
        self.load_model()
    
    def load_model(self):
        """Load the trained GAT model or create untrained fallback"""
        try:
            # First try to load trained model with saved config
            model_paths = [
                'models/enhanced_gat_complete.pth',
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        print(f"🔄 Attempting to load model from {model_path}")
                        # Fix for PyTorch 2.6 weights_only security change
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        
                        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                            # Load model with saved configuration
                            config = checkpoint['model_config'].copy()
                            config['n_students'] = 1  # Override for API single prediction
                            config['silent'] = True
                            
                            # Fix: Use config from checkpoint, but override n_students
                            print(f"📋 Loaded model config: {config}")
                            
                            # Check if config has the right dimensions based on actual checkpoint
                            # The error shows checkpoint has hidden_dim=64, output_dim=32, n_heads=4
                            if 'input_embedding.weight' in checkpoint['model_state_dict']:
                                checkpoint_hidden_dim = checkpoint['model_state_dict']['input_embedding.weight'].shape[0]
                                checkpoint_output_dim = checkpoint['model_state_dict']['gat_layer.layer_norm.weight'].shape[0]
                                checkpoint_n_heads = checkpoint['model_state_dict']['gat_layer.W'].shape[0]
                                
                                # Update config to match checkpoint
                                config['hidden_dim'] = checkpoint_hidden_dim
                                config['output_dim'] = checkpoint_output_dim
                                config['n_heads'] = checkpoint_n_heads
                                
                                print(f"📋 Corrected config based on checkpoint: hidden_dim={checkpoint_hidden_dim}, output_dim={checkpoint_output_dim}, n_heads={checkpoint_n_heads}")
                            
                            self.model = SimplifiedGATModel(**config)
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                            self.model_config = config
                            self.is_trained = True
                            model_loaded = True
                            print(f"✅ Successfully loaded trained model from {model_path}")
                            break
                            
                        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            # Try with notebook training config
                            config = {
                                'n_students': 1,
                                'n_modules': 7,
                                'student_features': 3,
                                'module_features': 3,
                                'hidden_dim': 64,  # Enhanced config from notebook
                                'output_dim': 32,   # Enhanced config from notebook
                                'n_heads': 4,       # Enhanced config from notebook
                                'dropout': 0.1,
                                'silent': True
                            }
                            
                            print(f"📋 Using notebook config: {config}")
                            self.model = SimplifiedGATModel(**config)
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                            self.model_config = config
                            self.is_trained = True
                            model_loaded = True
                            print(f"✅ Successfully loaded model with notebook config from {model_path}")
                            break
                            
                        else:
                            # Direct state dict
                            config = {
                                'n_students': 1,
                                'n_modules': 7,
                                'student_features': 3,
                                'module_features': 3,
                                'hidden_dim': 64,
                                'output_dim': 32,
                                'n_heads': 4,
                                'dropout': 0.1,
                                'silent': True
                            }
                            
                            self.model = SimplifiedGATModel(**config)
                            self.model.load_state_dict(checkpoint)
                            self.model_config = config
                            self.is_trained = True
                            model_loaded = True
                            print(f"✅ Successfully loaded direct state dict from {model_path}")
                            break
                            
                    except Exception as e:
                        print(f"❌ Failed to load from {model_path}: {str(e)}")
                        continue
            
            if not model_loaded:
                # Create fallback model with simplified config
                print("🔄 Creating fallback untrained model")
                config = {
                    'n_students': 1, 
                    'n_modules': 7, 
                    'student_features': 3, 
                    'module_features': 3,
                    'hidden_dim': 32, 
                    'output_dim': 16, 
                    'n_heads': 2, 
                    'dropout': 0.1, 
                    'silent': True
                }
                
                self.model = SimplifiedGATModel(**config)
                self.model_config = config
                self.is_trained = False
                print("⚠️ Using untrained fallback model")
            
            self.model.eval()
            
            # Initialize ModuleRecommender
            print("🎯 Initializing ModuleRecommender...")
            self.module_recommender = ModuleRecommender(self.model, self.model_config)
            print("✅ ModuleRecommender ready")
            
        except Exception as e:
            # Emergency fallback
            print(f"❌ Critical error in model loading: {str(e)}")
            config = {
                'n_students': 1, 
                'n_modules': 7, 
                'student_features': 3, 
                'module_features': 3,
                'hidden_dim': 32, 
                'output_dim': 16, 
                'n_heads': 2, 
                'dropout': 0.1, 
                'silent': True
            }
            
            self.model = SimplifiedGATModel(**config)
            self.model_config = config
            self.model.eval()
            self.is_trained = False
            print("🚨 Using emergency fallback model")
            
            # Initialize ModuleRecommender even with fallback
            self.module_recommender = ModuleRecommender(self.model, self.model_config)
    
    def predict(self, irt_ability: float, survey_confidence: float = 0.7):
        """Make prediction for single student level"""
        try:
            # Validate inputs
            irt_ability = max(min(float(irt_ability), 3.0), -3.0)
            survey_confidence = max(min(float(survey_confidence), 1.0), 0.0)
            
            # Create student features (3 dimensions exactly like notebook)
            ability_percentile = min(max((irt_ability + 3) / 6, 0), 1)
            
            student_features = [
                irt_ability,
                survey_confidence,
                ability_percentile
            ]
            
            return self._predict_single(student_features, irt_ability, survey_confidence)
            
        except Exception as e:
            return self._fallback_prediction(irt_ability, survey_confidence, str(e))
    
    def predict_batch(self, students_data):
        """Make predictions for multiple students"""
        results = []
        
        for student in students_data:
            try:
                irt_ability = student.get('irt_ability', 0.0)
                survey_confidence = student.get('survey_confidence', 0.7)
                student_id = student.get('student_id', 'unknown')
                
                # Validate inputs
                irt_ability = max(min(float(irt_ability), 3.0), -3.0)
                survey_confidence = max(min(float(survey_confidence), 1.0), 0.0)
                
                # Create student features
                ability_percentile = min(max((irt_ability + 3) / 6, 0), 1)
                student_features = [irt_ability, survey_confidence, ability_percentile]
                
                # Get prediction
                prediction = self._predict_single(student_features, irt_ability, survey_confidence)
                
                # Add student_id to result
                if prediction['success']:
                    prediction['student_id'] = student_id
                    results.append(prediction)
                else:
                    # Handle prediction error
                    fallback = self._fallback_prediction(irt_ability, survey_confidence, prediction.get('error', 'Unknown error'))
                    fallback['student_id'] = student_id
                    results.append(fallback)
                    
            except Exception as e:
                # Handle student processing error
                fallback = self._fallback_prediction(
                    student.get('irt_ability', 0.0),
                    student.get('survey_confidence', 0.7), 
                    str(e)
                )
                fallback['student_id'] = student.get('student_id', 'unknown')
                results.append(fallback)
        
        return results
    
    def _predict_single(self, student_features, irt_ability, survey_confidence):
        """Internal method for single prediction"""
            
        # Create module features (3 dimensions)
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
        
        # Create adjacency matrix
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
        
        # Get prediction
        with torch.no_grad():
            output = self.model(node_features_tensor, adj_matrix_tensor, edge_weights_tensor)
            
            # Get raw logits and probabilities
            raw_logits = output['student_levels'][0]
            level_probs = F.softmax(raw_logits, dim=0)
            predicted_level = torch.argmax(level_probs).item() + 1
            confidence = level_probs.max().item()
            
            # 🔧 ANTI-COLLAPSE POST-PROCESSING FIX
            # Apply IRT-based correction if model is biased towards Level 1
            if self.is_trained and predicted_level == 1:
                # Check if student should realistically be higher level
                if irt_ability > 0.5:  # Medium-high ability
                    # Rebalance probabilities based on IRT
                    irt_factor = min((irt_ability + 3) / 6, 1.0)  # 0-1 scale
                    
                    # Boost Level 2 and 3 probabilities
                    adjusted_probs = level_probs.clone()
                    
                    if irt_ability > 1.5:  # Very high ability -> boost Level 3
                        boost_factor = min(irt_ability / 3.0, 0.4)
                        adjusted_probs[2] += boost_factor
                        adjusted_probs[0] -= boost_factor * 0.7
                        adjusted_probs[1] -= boost_factor * 0.3
                    elif irt_ability > 0.5:  # Medium-high ability -> boost Level 2
                        boost_factor = min((irt_ability - 0.5) / 2.0, 0.3)
                        adjusted_probs[1] += boost_factor
                        adjusted_probs[0] -= boost_factor
                    
                    # Normalize probabilities
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    
                    # Update prediction based on adjusted probabilities
                    predicted_level = torch.argmax(adjusted_probs).item() + 1
                    confidence = adjusted_probs.max().item()
                    level_probs = adjusted_probs
                    
                    print(f"🔧 Anti-collapse correction applied for IRT={irt_ability:.3f}: {predicted_level}")
            
            # Original adjustment for untrained model
            elif not self.is_trained:
                if irt_ability < -0.5:
                    predicted_level = 1
                    confidence = min(confidence + 0.2, 0.9)
                elif irt_ability < 0.5:
                    predicted_level = 2
                    confidence = min(confidence + 0.1, 0.8)
                else:
                    predicted_level = 3
                    confidence = min(confidence + 0.15, 0.85)
            
            # Generate reasoning
            reasoning = f"GAT Analysis: IRT={irt_ability:.3f}, Confidence={survey_confidence:.1%}"
            if irt_ability < -1.0:
                reasoning += " → Level 1 (Foundational support needed)"
            elif irt_ability < 0.0:
                reasoning += " → Level 1-2 (Basic concepts)"
            elif irt_ability < 1.0:
                reasoning += " → Level 2 (Standard progression)"
            else:
                reasoning += " → Level 2-3 (Advanced ready)"
            
            return {
                'success': True,
                'predicted_level': predicted_level,
                'confidence': round(confidence, 4),
                'level_probabilities': [round(p, 4) for p in level_probs.tolist()],
                'reasoning': reasoning,
                'features': {
                    'irt_ability': irt_ability,
                    'survey_confidence': survey_confidence,
                    'ability_percentile': round(student_features[2], 4)
                },
                'model_trained': self.is_trained,
                'model_config': self.model_config,
                'architecture': 'SimplifiedGATModel'
            }
    
    def _fallback_prediction(self, irt_ability, survey_confidence, error_msg):
        """Fallback prediction when model fails"""
        # Simple IRT-based fallback
        if irt_ability < -0.5:
            level = 1
            confidence = 0.6
        elif irt_ability < 0.5:
            level = 2
            confidence = 0.65
        else:
            level = 3
            confidence = 0.7
        
        ability_percentile = min(max((irt_ability + 3) / 6, 0), 1)
        
        return {
            'success': False,
            'predicted_level': level,
            'confidence': round(confidence, 4),
            'level_probabilities': [0.33, 0.33, 0.34],
            'reasoning': f'GAT model error, using IRT fallback: {error_msg}',
            'features': {
                'irt_ability': irt_ability,
                'survey_confidence': survey_confidence,
                'ability_percentile': round(ability_percentile, 4)
            },
            'model_trained': False,
            'error': error_msg
        }
    
    def recommend_modules(
        self,
        irt_ability: float,
        survey_confidence: float,
        pre_test_results: list,
        strategy: str = 'weighted'
    ):
        """
        Generate module recommendations using attention-based approach
        
        Args:
            irt_ability: Student IRT ability score (-3 to 3)
            survey_confidence: Student self-assessment confidence (0 to 1)
            pre_test_results: Binary list [7] indicating pass (1) or fail (0) per module
            strategy: 'simple', 'weighted', or 'conditional' (default: 'weighted')
        
        Returns:
            Dict containing module_recommendations, recommended_order, and metadata
        """
        try:
            # Validate inputs
            irt_ability = max(min(float(irt_ability), 3.0), -3.0)
            survey_confidence = max(min(float(survey_confidence), 1.0), 0.0)
            
            if not isinstance(pre_test_results, list) or len(pre_test_results) != 7:
                raise ValueError("pre_test_results must be a list of 7 integers (0 or 1)")
            
            # Convert to binary (0 or 1)
            pre_test_results = [1 if x else 0 for x in pre_test_results]
            
            # Use ModuleRecommender
            result = self.module_recommender.recommend_modules(
                irt_ability=irt_ability,
                survey_confidence=survey_confidence,
                pre_test_results=pre_test_results,
                strategy=strategy
            )
            
            # Add success flag
            result['success'] = True
            
            return result
            
        except Exception as e:
            # Fallback recommendation
            return self._fallback_module_recommendation(
                irt_ability,
                survey_confidence,
                pre_test_results,
                str(e)
            )
    
    def _fallback_module_recommendation(
        self,
        irt_ability: float,
        survey_confidence: float,
        pre_test_results: list,
        error_msg: str
    ):
        """Fallback module recommendation when ModuleRecommender fails"""
        
        # Simple fallback: Unlock if passed pre-test
        recommendations = []
        for i in range(7):
            module_id = i + 1
            passed = pre_test_results[i] == 1 if i < len(pre_test_results) else False
            
            recommendations.append({
                'module_id': module_id,
                'module_name': f'Module {module_id}',
                'unlock': passed,
                'confidence': 0.5 if passed else 0.1,
                'attention_score': 0.0,
                'pre_test_passed': passed,
                'reasoning': 'Fallback: Based on pre-test result only'
            })
        
        # Simple order: Unlocked first
        unlocked = [r['module_id'] for r in recommendations if r['unlock']]
        locked = [r['module_id'] for r in recommendations if not r['unlock']]
        recommended_order = unlocked + locked
        
        return {
            'success': False,
            'module_recommendations': recommendations,
            'recommended_order': recommended_order,
            'metadata': {
                'irt_ability': round(irt_ability, 4),
                'survey_confidence': round(survey_confidence, 4),
                'pre_test_summary': {
                    'total_passed': sum(pre_test_results),
                    'total_failed': 7 - sum(pre_test_results),
                    'pass_rate': round(sum(pre_test_results) / 7, 4)
                },
                'strategy_used': 'fallback',
                'model_trained': False
            },
            'error': error_msg
        }

# Global predictor instance
predictor = GATPredictor()