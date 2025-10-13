"""
GAT Prediction Service for Vercel Deployment
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from .gat_model import SimplifiedGATModel

class GATPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.load_model()
    
    def load_model(self):
        """Load the trained GAT model or create untrained fallback"""
        try:
            # Model configuration matching notebook
            model_config = {
                'n_students': 1,
                'n_modules': 7,
                'student_features': 3,
                'module_features': 3,
                'hidden_dim': 64,
                'output_dim': 32,
                'n_heads': 4,
                'dropout': 0.1,
                'silent': True  # Don't print during API calls
            }
            
            self.model = SimplifiedGATModel(**model_config)
            
            # Try to load trained weights if available
            model_paths = [
                'models/enhanced_gat_complete.pth',
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        
                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                            elif 'model_config' in checkpoint:
                                # Complete model package
                                config = checkpoint['model_config']
                                config['n_students'] = 1
                                config['silent'] = True
                                self.model = SimplifiedGATModel(**config)
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                self.model.load_state_dict(checkpoint)
                        else:
                            self.model.load_state_dict(checkpoint)
                        
                        self.is_trained = True
                        break
                    except Exception:
                        continue
            
            self.model.eval()
            
        except Exception as e:
            # Fallback model
            self.model = SimplifiedGATModel(
                n_students=1, n_modules=7, student_features=3, module_features=3,
                hidden_dim=32, output_dim=16, n_heads=2, dropout=0.1, silent=True
            )
            self.model.eval()
            self.is_trained = False
    
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
            
            level_probs = F.softmax(output['student_levels'][0], dim=0)
            predicted_level = torch.argmax(level_probs).item() + 1
            confidence = level_probs.max().item()
            
            # Adjust prediction if model is not trained
            if not self.is_trained:
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

# Global predictor instance
predictor = GATPredictor()