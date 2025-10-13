"""
Main Flask API for GAT Prediction - Vercel Deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.predictor import predictor

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """API Home endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'GAT Prediction API is running!',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health': '/api/health (GET)',
            'model_info': '/api/model-info (GET)'
        },
        'usage': {
            'predict': {
                'method': 'POST',
                'body': {
                    'irt_ability': 'float (-3.0 to 3.0)',
                    'survey_confidence': 'float (0.0 to 1.0, optional, default: 0.7)'
                },
                'example': {
                    'irt_ability': 0.5,
                    'survey_confidence': 0.8
                }
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'code': 'MISSING_DATA'
            }), 400
        
        # Extract parameters
        irt_ability = data.get('irt_ability')
        survey_confidence = data.get('survey_confidence', 0.7)
        
        # Validate required parameters
        if irt_ability is None:
            return jsonify({
                'status': 'error',
                'message': 'irt_ability is required',
                'code': 'MISSING_IRT_ABILITY'
            }), 400
        
        # Validate parameter types
        try:
            irt_ability = float(irt_ability)
            survey_confidence = float(survey_confidence)
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'irt_ability and survey_confidence must be numbers',
                'code': 'INVALID_PARAMETER_TYPE'
            }), 400
        
        # Validate parameter ranges
        if not (-5.0 <= irt_ability <= 5.0):
            return jsonify({
                'status': 'error',
                'message': 'irt_ability must be between -5.0 and 5.0',
                'code': 'IRT_ABILITY_OUT_OF_RANGE'
            }), 400
        
        if not (0.0 <= survey_confidence <= 1.0):
            return jsonify({
                'status': 'error',
                'message': 'survey_confidence must be between 0.0 and 1.0',
                'code': 'SURVEY_CONFIDENCE_OUT_OF_RANGE'
            }), 400
        
        # Make prediction
        result = predictor.predict(irt_ability, survey_confidence)
        
        # Return result
        return jsonify({
            'status': 'success',
            'data': result,
            'api_version': '1.0.0',
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple students"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'code': 'MISSING_DATA'
            }), 400
        
        # Extract batch data
        students = data.get('students', [])
        
        if not students or not isinstance(students, list):
            return jsonify({
                'status': 'error',
                'message': 'students array is required',
                'code': 'MISSING_STUDENTS'
            }), 400
        
        if len(students) > 100:  # Limit batch size
            return jsonify({
                'status': 'error',
                'message': 'Maximum 100 students per batch',
                'code': 'BATCH_SIZE_EXCEEDED'
            }), 400
        
        # Validate batch data first
        validated_students = []
        errors = []
        
        for i, student in enumerate(students):
            try:
                # Validate student data structure
                if not isinstance(student, dict):
                    errors.append({
                        'index': i,
                        'error': 'Student data must be an object',
                        'student_id': student.get('student_id', 'unknown') if isinstance(student, dict) else 'invalid'
                    })
                    continue
                
                student_id = student.get('student_id')
                irt_ability = student.get('irt_ability')
                survey_confidence = student.get('survey_confidence', 0.7)
                
                # Validate required fields
                if student_id is None:
                    errors.append({
                        'index': i,
                        'error': 'student_id is required',
                        'student_id': 'missing'
                    })
                    continue
                
                if irt_ability is None:
                    errors.append({
                        'index': i,
                        'error': 'irt_ability is required',
                        'student_id': student_id
                    })
                    continue
                
                # Validate and convert types
                try:
                    irt_ability = float(irt_ability)
                    survey_confidence = float(survey_confidence)
                except (ValueError, TypeError):
                    errors.append({
                        'index': i,
                        'error': 'irt_ability and survey_confidence must be numbers',
                        'student_id': student_id
                    })
                    continue
                
                # Validate ranges
                if not (-5.0 <= irt_ability <= 5.0):
                    errors.append({
                        'index': i,
                        'error': 'irt_ability must be between -5.0 and 5.0',
                        'student_id': student_id
                    })
                    continue
                
                if not (0.0 <= survey_confidence <= 1.0):
                    errors.append({
                        'index': i,
                        'error': 'survey_confidence must be between 0.0 and 1.0',
                        'student_id': student_id
                    })
                    continue
                
                # Add to validated list
                validated_students.append({
                    'student_id': str(student_id),
                    'irt_ability': irt_ability,
                    'survey_confidence': survey_confidence
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': f'Validation failed: {str(e)}',
                    'student_id': student.get('student_id', 'unknown') if isinstance(student, dict) else 'invalid'
                })
        
        # Process validated students using batch predictor
        results = []
        if validated_students:
            try:
                batch_results = predictor.predict_batch(validated_students)
                results = batch_results
            except Exception as e:
                # Fallback to individual predictions if batch fails
                for student in validated_students:
                    try:
                        result = predictor.predict(student['irt_ability'], student['survey_confidence'])
                        result['student_id'] = student['student_id']
                        results.append(result)
                    except Exception as individual_error:
                        errors.append({
                            'error': f'Individual prediction failed: {str(individual_error)}',
                            'student_id': student['student_id']
                        })
        
        # Return batch results
        return jsonify({
            'status': 'success',
            'data': {
                'total_students': len(students),
                'successful_predictions': len(results),
                'failed_predictions': len(errors),
                'results': results,
                'errors': errors
            },
            'api_version': '1.0.0',
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Batch processing failed: {str(e)}',
            'code': 'BATCH_PROCESSING_ERROR'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test model loading
        test_result = predictor.predict(0.0, 0.7)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'model_trained': predictor.is_trained,
            'test_prediction': {
                'input': {'irt_ability': 0.0, 'survey_confidence': 0.7},
                'output': test_result['predicted_level']
            },
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({
        'status': 'success',
        'model_info': {
            'architecture': 'SimplifiedGATModel',
            'features': '3D (IRT_Ability, Survey_Confidence, Ability_Percentile)',
            'model_loaded': predictor.model is not None,
            'model_trained': predictor.is_trained,
            'parameters': {
                'n_students': 1,
                'n_modules': 7,
                'student_features': 3,
                'module_features': 3,
                'hidden_dim': 64 if predictor.is_trained else 32,
                'output_dim': 32 if predictor.is_trained else 16,
                'n_heads': 4 if predictor.is_trained else 2
            },
            'prediction_levels': [1, 2, 3],
            'input_ranges': {
                'irt_ability': [-3.0, 3.0],
                'survey_confidence': [0.0, 1.0]
            }
        }
    })

# For Vercel serverless function
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)