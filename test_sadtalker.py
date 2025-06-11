#!/usr/bin/env python3
"""
Test script for SadTalker integration
"""

import os
import sys
from sadtalker_service import SadTalkerService

def test_sadtalker():
    """Test SadTalker service with existing files"""
    print("Testing SadTalker integration...")
    
    # Initialize service
    service = SadTalkerService(data_dir="data")
    
    # Check if source image exists
    source_image = service.get_source_image()
    if not source_image:
        print("❌ No source image found in data directory")
        return False
    
    print(f"✅ Source image found: {source_image}")
    
    # Check if audio files exist
    temp_audio_dir = os.path.join("data", "temp_audio")
    latest_audio = service.get_latest_audio_file(temp_audio_dir)
    if not latest_audio:
        print("❌ No audio files found in temp_audio directory")
        return False
        
    print(f"✅ Latest audio file found: {latest_audio}")
    
    # Test video generation
    print("\n🎬 Generating video with SadTalker...")
    video_path = service.generate_video_from_latest_audio(
        expression_scale=1.0,
        still=True
    )
    
    if video_path:
        print(f"✅ Video generated successfully: {video_path}")
        print(f"📁 File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        return True
    else:
        print("❌ Video generation failed")
        return False

def main():
    """Main test function"""
    print("SadTalker Integration Test")
    print("=" * 40)
    
    # Check if Docker is available
    try:
        import subprocess
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
        else:
            print("❌ Docker not available or not responding")
            return
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return
    
    # Check if SadTalker image is available
    try:
        result = subprocess.run(["docker", "images", "wawa9000/sadtalker"], 
                              capture_output=True, text=True, timeout=10)
        if "wawa9000/sadtalker" in result.stdout:
            print("✅ SadTalker Docker image found")
        else:
            print("⚠️  SadTalker Docker image not found locally")
            print("   It will be downloaded automatically when first used")
    except Exception as e:
        print(f"⚠️  Could not check for SadTalker image: {e}")
    
    print("\n" + "=" * 40)
    
    # Run the test
    success = test_sadtalker()
    
    if success:
        print("\n🎉 SadTalker integration test PASSED!")
        print("   You can now use the AI Digital Twin with video generation")
    else:
        print("\n❌ SadTalker integration test FAILED!")
        print("   Please check the error messages above")

if __name__ == "__main__":
    main()
