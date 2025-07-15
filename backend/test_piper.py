#!/usr/bin/env python3
"""
Test script for Piper TTS functionality.
Run this script to verify that Piper TTS is properly installed and configured.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_piper_installation():
    """Test if Piper TTS is installed and accessible."""
    print("Testing Piper TTS installation...")
    
    try:
        result = subprocess.run(["piper", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Piper TTS is installed: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Piper TTS command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ Piper TTS not found. Please install Piper TTS first.")
        return False
    except Exception as e:
        print(f"✗ Error testing Piper TTS: {e}")
        return False

def test_model_files():
    """Test if the required model files exist."""
    print("\nTesting model files...")
    
    model_path = Path("C:/Users/ankit/piper/en_US-lessac-medium.onnx")
    config_path = Path("C:/Users/ankit/piper/en_en_US_lessac_medium_en_US-lessac-medium.onnx.json")
    
    if model_path.exists():
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Please download the model file from:")
        print("  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
    
    if config_path.exists():
        print(f"✓ Config file found: {config_path}")
    else:
        print(f"✗ Config file not found: {config_path}")
        print("  Please download the config file from:")
        print("  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
    
    return model_path.exists() and config_path.exists()

def test_voice_generation():
    """Test voice generation with Piper TTS."""
    print("\nTesting voice generation...")

    test_text = "Hello, this is a test of Piper TTS with the Lessac voice model."
    output_file = "test_output.wav"
    model_file = "C:/Users/ankit/piper/en_US-lessac-medium.onnx"
    config_file = "C:/Users/ankit/piper/en_en_US_lessac_medium_en_US-lessac-medium.onnx.json"

    try:
        # Piper command
        cmd = [
            "piper",
            "--model", model_file,
            "--config", config_file,
            "--output_file", output_file
        ]

        # Create a temp file for the test text
        temp_text_file = "temp_test.txt"
        with open(temp_text_file, "w", encoding="utf-8") as f:
            f.write(test_text)

        try:
            # Feed the file content into Piper
            with open(temp_text_file, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    stdin=f,
                    text=True,
                    capture_output=True,
                    timeout=30
                )

            if result.returncode == 0:
                if Path(output_file).exists() and Path(output_file).stat().st_size > 0:
                    print(f"✓ Voice generation successful: {output_file}")
                    print(f"  File size: {Path(output_file).stat().st_size} bytes")

                    # Clean up output
                    try:
                        os.remove(output_file)
                        print("  Test file cleaned up")
                    except:
                        pass

                    return True
                else:
                    print("✗ Voice generation failed: Empty or missing output file")
                    return False
            else:
                print(f"✗ Voice generation failed: {result.stderr}")
                return False

        finally:
            try:
                os.remove(temp_text_file)
            except:
                pass

    except Exception as e:
        print(f"✗ Error during voice generation: {e}")
        return False

def test_python_dependencies():
    """Test if required Python packages are installed."""
    print("\nTesting Python dependencies...")
    
    required_packages = ['sounddevice', 'soundfile']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            print(f"  Install with: pip install {package}")
    
    return all(__import__(package) for package in required_packages)

def main():
    """Run all tests."""
    print("Piper TTS Test Suite")
    print("=" * 50)
    
    tests = [
        ("Piper Installation", test_piper_installation),
        ("Model Files", test_model_files),
        ("Voice Generation", test_voice_generation),
        ("Python Dependencies", test_python_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Piper TTS is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 