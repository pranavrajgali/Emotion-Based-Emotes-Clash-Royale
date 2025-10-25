"""
Real-Time Facial Emotion Recognition with Clash Royale Emotes
Author: Expert AI/ML DevOps Engineer
Platform: Windows + Anaconda + VS Code
"""

import cv2
import numpy as np
from fer import FER
from PIL import Image
import os
import sys

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

EMOTE_SIZE = (150, 150)  # Size of emote overlay (width, height)
EMOTE_OFFSET_Y = -160    # Vertical offset above face (negative = above)
EMOTE_OFFSET_X = 0       # Horizontal offset (0 = centered)
CONFIDENCE_THRESHOLD = 0.3  # Minimum emotion confidence (0-1 scale)

# Emotion to emote file mapping
EMOTION_EMOTE_MAP = {
    'angry': 'angry.png',
    'disgust': 'disgust.png',
    'fear': 'fear.png',
    'happy': 'happy.png',
    'sad': 'sad.png',
    'surprise': 'surprise.png',
    'neutral': 'neutral.png'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_emotes(emote_folder='emotes'):
    """
    Load all emote PNG images with transparency into memory.
    
    Args:
        emote_folder (str): Path to folder containing emote PNG files
    
    Returns:
        dict: Dictionary mapping emotion names to RGBA numpy arrays
    """
    emotes = {}
    
    if not os.path.exists(emote_folder):
        print(f"❌ ERROR: Emote folder '{emote_folder}' not found!")
        print(f"📁 Please create folder: {os.path.abspath(emote_folder)}")
        sys.exit(1)
    
    for emotion, filename in EMOTION_EMOTE_MAP.items():
        filepath = os.path.join(emote_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  WARNING: Missing emote file: {filepath}")
            continue
        
        # Load with PIL to preserve alpha channel
        pil_image = Image.open(filepath).convert('RGBA')
        
        # Resize to target dimensions
        pil_image = pil_image.resize(EMOTE_SIZE, Image.LANCZOS)
        
        # Convert to numpy array (RGBA format)
        emotes[emotion] = np.array(pil_image)
        
        print(f"✅ Loaded: {filename} → {emotion}")
    
    if len(emotes) == 0:
        print("❌ ERROR: No emote files loaded! Check your emotes folder.")
        sys.exit(1)
    
    print(f"\n✅ Successfully loaded {len(emotes)} emotes\n")
    return emotes


def overlay_transparent(background, overlay, x, y):
    """
    Overlay a transparent PNG (RGBA) onto a background image (BGR).
    Uses alpha blending for smooth transparency.
    
    Args:
        background (np.array): Background image in BGR format
        overlay (np.array): Overlay image in RGBA format
        x (int): X coordinate for overlay placement (top-left)
        y (int): Y coordinate for overlay placement (top-left)
    
    Returns:
        np.array: Composite image with overlay applied
    """
    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]
    
    # Boundary checks - clip overlay to fit within background
    if y < 0:
        overlay = overlay[-y:, :]
        overlay_height += y
        y = 0
    
    if x < 0:
        overlay = overlay[:, -x:]
        overlay_width += x
        x = 0
    
    if y + overlay_height > bg_height:
        overlay = overlay[:bg_height - y, :]
        overlay_height = bg_height - y
    
    if x + overlay_width > bg_width:
        overlay = overlay[:, :bg_width - x]
        overlay_width = bg_width - x
    
    # Extract alpha channel (transparency mask)
    alpha = overlay[:, :, 3] / 255.0  # Normalize to 0-1
    
    # Extract RGB channels
    overlay_rgb = overlay[:, :, :3]
    
    # Extract region of interest from background
    roi = background[y:y+overlay_height, x:x+overlay_width]
    
    # Alpha blending formula: result = foreground * alpha + background * (1 - alpha)
    for c in range(3):  # Loop through B, G, R channels
        roi[:, :, c] = (overlay_rgb[:, :, c] * alpha + 
                        roi[:, :, c] * (1 - alpha[:, :]))
    
    return background


def draw_emotion_info(frame, emotion_data, x, y, w, h):
    """
    Draw emotion label and confidence bar on the frame.
    
    Args:
        frame (np.array): Video frame
        emotion_data (dict): Dictionary with emotion scores
        x, y, w, h (int): Bounding box coordinates
    """
    if not emotion_data:
        return
    
    # Get dominant emotion
    dominant_emotion = max(emotion_data, key=emotion_data.get)
    confidence = emotion_data[dominant_emotion]
    
    # Prepare text
    text = f"{dominant_emotion.upper()}: {confidence:.2f}"
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text below face
    text_x = x
    text_y = y + h + 30
    
    # Draw background rectangle for text
    cv2.rectangle(frame, 
                  (text_x, text_y - text_height - 5),
                  (text_x + text_width, text_y + baseline),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    
    # Draw confidence bar
    bar_width = w
    bar_height = 10
    bar_x = x
    bar_y = y + h + 35
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Confidence fill
    fill_width = int(bar_width * confidence)
    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main execution function for real-time emotion recognition system.
    """
    print("=" * 70)
    print("🎭 REAL-TIME FACIAL EMOTION RECOGNITION SYSTEM")
    print("=" * 70)
    
    # Initialize FER detector
    print("\n🔧 Initializing FER emotion detector...")
    detector = FER(mtcnn=True)  # Use MTCNN for better face detection
    print("✅ Detector initialized")
    
    # Load emote images
    print("\n📦 Loading emote assets...")
    emotes = load_emotes('emotes')
    
    # Initialize webcam
    print("📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot access webcam!")
        print("🔍 Troubleshooting:")
        print("   1. Check if another application is using the camera")
        print("   2. Verify camera permissions in Windows Settings")
        print("   3. Try reconnecting the camera")
        sys.exit(1)
    
    # Set camera properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam initialized")
    print("\n" + "=" * 70)
    print("🎬 SYSTEM ACTIVE - Press 'Q' to quit")
    print("=" * 70 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  Warning: Failed to grab frame")
                continue
            
            # Mirror the frame for more intuitive user experience
            frame = cv2.flip(frame, 1)
            
            # Detect emotions (process every 3rd frame for performance)
            if frame_count % 3 == 0:
                result = detector.detect_emotions(frame)
            
            # Process detected faces
            if result:
                for face in result:
                    # Extract face bounding box
                    (x, y, w, h) = face["box"]
                    emotions = face["emotions"]
                    
                    # Get dominant emotion
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw emotion info
                    draw_emotion_info(frame, emotions, x, y, w, h)
                    
                    # Overlay emote if confidence is sufficient
                    if confidence > CONFIDENCE_THRESHOLD and dominant_emotion in emotes:
                        emote = emotes[dominant_emotion]
                        
                        # Calculate emote position (centered above face)
                        emote_x = x + (w // 2) - (EMOTE_SIZE[0] // 2) + EMOTE_OFFSET_X
                        emote_y = y + EMOTE_OFFSET_Y
                        
                        # Apply overlay
                        frame = overlay_transparent(frame, emote, emote_x, emote_y)
            
            # Display FPS counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Emotion Recognition - Press Q to Quit', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Shutting down system...")
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Resources released. Goodbye!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()