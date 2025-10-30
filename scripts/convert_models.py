#!/usr/bin/env python3
"""
NNCP Model Conversion Script
Converts PyTorch/ONNX models to CoreML for optimal Apple Silicon performance
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: CoreML Tools not available")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelConverter:
    """Handles conversion of various model formats to CoreML"""
    
    def __init__(self, optimization_profile: str = "default"):
        self.optimization_profile = optimization_profile
        self.supported_profiles = {
            "default": self._get_default_config(),
            "speed": self._get_speed_config(),
            "size": self._get_size_config(),
            "battery": self._get_battery_config()
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default balanced configuration"""
        return {
            "compute_precision": ct.precision.FLOAT16,
            "quantize_weights": False,
            "optimize_for_neural_engine": True,
            "minimum_deployment_target": ct.target.macOS13
        }
    
    def _get_speed_config(self) -> Dict[str, Any]:
        """Speed-optimized configuration"""
        return {
            "compute_precision": ct.precision.FLOAT32,
            "quantize_weights": False,
            "optimize_for_neural_engine": True,
            "minimum_deployment_target": ct.target.macOS13
        }
    
    def _get_size_config(self) -> Dict[str, Any]:
        """Size-optimized configuration"""
        return {
            "compute_precision": ct.precision.FLOAT16,
            "quantize_weights": True,
            "quantization_mode": "linear",
            "optimize_for_neural_engine": True,
            "minimum_deployment_target": ct.target.macOS13
        }
    
    def _get_battery_config(self) -> Dict[str, Any]:
        """Battery-optimized configuration"""
        return {
            "compute_precision": ct.precision.FLOAT16,
            "quantize_weights": True,
            "quantization_mode": "linear",
            "optimize_for_neural_engine": True,
            "minimum_deployment_target": ct.target.macOS13,
            "compute_units": ct.ComputeUnit.CPU_AND_NE  # Prefer Neural Engine
        }
    
    def convert_pytorch_to_coreml(self, 
                                 pytorch_path: str, 
                                 output_path: str,
                                 input_shape: tuple,
                                 model_class: Optional[str] = None) -> bool:
        """Convert PyTorch model to CoreML"""
        if not PYTORCH_AVAILABLE or not COREML_AVAILABLE:
            logger.error("PyTorch and CoreML Tools are required for this conversion")
            return False
        
        try:
            logger.info(f"Loading PyTorch model from {pytorch_path}")
            
            # Load the model
            if model_class:
                # Load model with custom class
                model = torch.load(pytorch_path, map_location='cpu')
            else:
                # Load standard saved model
                model = torch.load(pytorch_path, map_location='cpu')
            
            model.eval()
            
            # Create example input
            example_input = torch.randn(input_shape)
            
            # Trace the model
            logger.info("Tracing PyTorch model...")
            traced_model = torch.jit.trace(model, example_input)
            
            # Get configuration for optimization profile
            config = self.supported_profiles[self.optimization_profile]
            
            # Convert to CoreML
            logger.info(f"Converting to CoreML with profile: {self.optimization_profile}")
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape)],
                compute_precision=config["compute_precision"],
                minimum_deployment_target=config["minimum_deployment_target"],
                compute_units=config.get("compute_units", ct.ComputeUnit.ALL)
            )
            
            # Apply quantization if requested
            if config.get("quantize_weights", False):
                logger.info("Applying weight quantization...")
                if config.get("quantization_mode") == "linear":
                    coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=8)
            
            # Set metadata
            coreml_model.short_description = "NNCP Neural Network Model"
            coreml_model.author = "NNCP Converter"
            coreml_model.license = "MIT"
            coreml_model.version = "1.0"
            
            # Save the model
            logger.info(f"Saving CoreML model to {output_path}")
            coreml_model.save(output_path)
            
            # Print model info
            logger.info(f"Model conversion successful!")
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Optimization profile: {self.optimization_profile}")
            logger.info(f"Compute precision: {config['compute_precision']}")
            logger.info(f"Quantized: {config.get('quantize_weights', False)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert PyTorch model: {e}")
            return False
    
    def convert_onnx_to_coreml(self, 
                              onnx_path: str, 
                              output_path: str) -> bool:
        """Convert ONNX model to CoreML"""
        if not ONNX_AVAILABLE or not COREML_AVAILABLE:
            logger.error("ONNX and CoreML Tools are required for this conversion")
            return False
        
        try:
            logger.info(f"Loading ONNX model from {onnx_path}")
            
            # Get configuration
            config = self.supported_profiles[self.optimization_profile]
            
            # Convert to CoreML
            logger.info(f"Converting ONNX to CoreML with profile: {self.optimization_profile}")
            coreml_model = ct.convert(
                onnx_path,
                compute_precision=config["compute_precision"],
                minimum_deployment_target=config["minimum_deployment_target"],
                compute_units=config.get("compute_units", ct.ComputeUnit.ALL)
            )
            
            # Apply quantization if requested
            if config.get("quantize_weights", False):
                logger.info("Applying weight quantization...")
                coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=8)
            
            # Save the model
            logger.info(f"Saving CoreML model to {output_path}")
            coreml_model.save(output_path)
            
            logger.info("ONNX to CoreML conversion successful!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert ONNX model: {e}")
            return False
    
    def validate_coreml_model(self, model_path: str, test_input: Optional[torch.Tensor] = None) -> bool:
        """Validate the converted CoreML model"""
        if not COREML_AVAILABLE:
            logger.error("CoreML Tools required for validation")
            return False
        
        try:
            logger.info(f"Validating CoreML model: {model_path}")
            
            # Load the model
            model = ct.models.MLModel(model_path)
            
            # Print model information
            spec = model.get_spec()
            logger.info(f"Model description: {spec.description}")
            
            # Get input/output info
            input_desc = spec.description.input[0]
            output_desc = spec.description.output[0]
            
            logger.info(f"Input: {input_desc.name}, shape: {input_desc.type}")
            logger.info(f"Output: {output_desc.name}, shape: {output_desc.type}")
            
            # Test prediction if test input provided
            if test_input is not None:
                logger.info("Running test prediction...")
                input_dict = {input_desc.name: test_input.numpy()}
                prediction = model.predict(input_dict)
                logger.info(f"Test prediction successful, output keys: {list(prediction.keys())}")
            
            logger.info("Model validation successful!")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Convert models to CoreML for NNCP")
    parser.add_argument("input", help="Input model path")
    parser.add_argument("output", help="Output CoreML model path")
    parser.add_argument("--format", choices=["pytorch", "onnx"], required=True,
                       help="Input model format")
    parser.add_argument("--profile", choices=["default", "speed", "size", "battery"],
                       default="default", help="Optimization profile")
    parser.add_argument("--input-shape", nargs="+", type=int,
                       help="Input tensor shape (for PyTorch models)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the converted model")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter
    converter = ModelConverter(args.profile)
    
    # Convert based on format
    success = False
    if args.format == "pytorch":
        if not args.input_shape:
            logger.error("--input-shape is required for PyTorch models")
            sys.exit(1)
        
        input_shape = tuple(args.input_shape)
        success = converter.convert_pytorch_to_coreml(args.input, args.output, input_shape)
    
    elif args.format == "onnx":
        success = converter.convert_onnx_to_coreml(args.input, args.output)
    
    if not success:
        logger.error("Model conversion failed")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        if not converter.validate_coreml_model(args.output):
            logger.error("Model validation failed")
            sys.exit(1)
    
    logger.info("Model conversion completed successfully!")


if __name__ == "__main__":
    main()
