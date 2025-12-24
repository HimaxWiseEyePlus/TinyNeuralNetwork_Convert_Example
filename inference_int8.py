import numpy as np
from PIL import Image
import argparse
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

def load_interpreter(model_path):
    """Loads the TFLite interpreter."""
    print(f"Loading model from {model_path}...")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_output_details(interpreter):
    """Gets input and output details from the interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def preprocess_image(image_path, input_shape, input_dtype, quantization_details):
    """Preprocesses the image for the model."""
    height, width = input_shape[1], input_shape[2]
    
    # Load and resize image
    if image_path:
        img = Image.open(image_path).convert('RGB')
    else:
        print("No image provided. Creating a dummy random image.")
        img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
        
    img = img.resize((width, height))
    img_array = np.array(img)

    # Handle Quantization
    # For int8 models, we often expect input in range [-128, 127] or similar, based on quantization params.
    # q = x / scale + zero_point
    # However, usually we can just convert input (0-255) to the representative float range and then quantize,
    # OR if the model expects int8 directly from standard image uint8, we strictly follow the params.
    
    scale, zero_point = quantization_details['quantization']
    print(f"Input Quantization - Scale: {scale}, Zero Point: {zero_point}, Dtype: {input_dtype}")

    if input_dtype == np.int8 or input_dtype == np.uint8:
        # Assuming the model was trained such that:
        # real_value = (int8_value - zero_point) * scale
        # We want to find int8_value corresponding to our image pixel value.
        # But wait, usually image pixels are 0-255 (uint8). 
        # Deep learning models usually work on normalized float inputs (e.g. 0-1 or -1 to 1).
        # We need to know the float range the model expects if we do it manually.
        # BUT, standard TFLite quantization usually maps the input range to the int8 range.
        
        # Standard approach: Normalize to float, then quantize.
        # If the expected input in float was [0, 1] (common):
        # input_float = img_array / 255.0
        # Check if scale is approx 1/255 or something else.
        
        # Let's try to infer from typical MobileNet preprocessing.
        # MobileNet usually expects [-1, 1] float input.
        # float_input = (img_array / 127.5) - 1.0
        
        # To be safe, let's use the quantization parameters directly if they make sense.
        # int_val = input_float / scale + zero_point
        # If we assume input_float is the "real world value", we need to know what "real world value" the model expects.
        # However, for int8/uint8 inputs, often we can simplistically trust:
        # input_tensor = (input_float / scale) + zero_point
        
        # Let's try assuming the model expects normalized floats [-1, 1] (common for mobilenet)
        input_float = (img_array.astype(np.float32) / 127.5) - 1.0
        input_quantized = (input_float / scale) + zero_point
        input_quantized = np.round(input_quantized).astype(input_dtype)
        
        # Provide a warning if we are just guessing the normalization.
        print("NOTE: Assuming MobileNet-style input normalization [-1, 1] before quantization.")
        
        return np.expand_dims(input_quantized, axis=0)
    else:
        # Float model
        input_float = (img_array.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(input_float, axis=0)


def run_inference(interpreter, input_data):
    """Runs inference."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, output_details

def main():
    parser = argparse.ArgumentParser(description="Run TFLite Int8 Inference")
    parser.add_argument("--model_path", type=str, default="qat_model.tflite", help="Path to .tflite model")
    parser.add_argument("--image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        return

    interpreter = load_interpreter(args.model_path)
    input_details, output_details = get_input_output_details(interpreter)

    print("\nInput Details:")
    print(input_details)
    print("\nOutput Details:")
    print(output_details)

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Process Image
    input_data = preprocess_image(args.image_path, input_shape, input_dtype, input_details[0])
    
    # Inference
    print("\nRunning Inference...")
    output_data, out_details = run_inference(interpreter, input_data)
    
    # Post Processing
    out_scale, out_zero_point = out_details[0]['quantization']
    print(f"\nOutput Quantization - Scale: {out_scale}, Zero Point: {out_zero_point}")
    
    if out_scale > 0:
        # Dequantize
        output_float = (output_data.astype(np.float32) - out_zero_point) * out_scale
        print("\nDequantized Output (Logits/Probs):")
        print(output_float)
        
        # Top prediction
        predicted_idx = np.argmax(output_float)
        print(f"\nPredicted Class Index: {predicted_idx}")
    else:
        print("\nRaw Output:")
        print(output_data)
        predicted_idx = np.argmax(output_data)
        print(f"\nPredicted Class Index: {predicted_idx}")
        
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if predicted_idx < len(cifar10_labels):
        print(f"Predicted Label: {cifar10_labels[predicted_idx]}")

if __name__ == "__main__":
    main()
