import os
import torch
import cv2
import math
import numpy as np
import pandas as pd
from torchvision import transforms, models
from ultralytics import YOLO
from tqdm import tqdm
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib# For TensorFlow device placement
import tensorflow as tf
import time  # Import time module
from collections import deque
import os
import easyocr
from openai import OpenAI

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
import easyocr
from openai import OpenAI

# Global EasyOCR reader 
reader = easyocr.Reader(['en'])


client = OpenAI(api_key="OPEN_AI_KEY")  # Replace with your actual API key




warnings.filterwarnings("ignore", category=FutureWarning)

###########################################
# 1) LOADING MODELS
###########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Restrict to single GPU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for ALL GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable XLA after setting memory growth
        tf.config.optimizer.set_jit(True)
        
        # Validate configuration
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        
    except RuntimeError as e:
        print(f"GPU config error: {e}")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Court Keypoint Model (ResNet-18) ---
court_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
court_model.fc = torch.nn.Linear(court_model.fc.in_features, 14 * 2)
court_model.load_state_dict(torch.load(
    r"Mmodels\keypoints_model_final.pth",
    map_location=device
))
court_model = court_model.to(device)
court_model.eval()
print("Court keypoint model loaded.")

# YOLO model inference
yolo_model = YOLO(r"models\last.pt").to("cuda")


# Modified version with optimizations
with torch.no_grad(), torch.cuda.amp.autocast():
    # 1. Fuse Conv+BN layers (safe optimization)
    yolo_model.fuse()  # In-place operation
    
    # 2. Half-precision conversion (maintains accuracy)
    yolo_model.model.half().to("cuda")  # Explicit FP16 conversion
    
    # 3. Warmup pass
    dummy_input = torch.zeros(1, 3, 640, 640).half().to("cuda")
    _ = yolo_model(dummy_input, verbose=False)
print(f"YOLO model (players/net) loaded on {yolo_model.device}")

# --- YOLO pose model ---
# YOLO model inference
pose_model = YOLO(r"yolov8s-pose.pt").to("cuda")


# Modified version with optimizations
with torch.no_grad(), torch.cuda.amp.autocast():
    # 1. Fuse Conv+BN layers (safe optimization)
    pose_model.fuse()  # In-place operation
    
    # 2. Half-precision conversion (maintains accuracy)
    pose_model.model.half().to("cuda")  # Explicit FP16 conversion
    
    # 3. Warmup pass
    dummy_input = torch.zeros(1, 3, 640, 640).half().to("cuda")
    _ = pose_model(dummy_input, verbose=False)
print(f"YOLO pose model loaded on {pose_model.device}")

# --- TrackNet for specialized ball tracking ---
from model import BallTrackerNet  # Ensure 'model.py' is in the same folder
tracknet_model = BallTrackerNet()
tracknet_model.load_state_dict(torch.load(
    r"models\model_best.pt",
    map_location=device
))
tracknet_model = tracknet_model.to(device)
tracknet_model.eval()
print("TrackNet model loaded.")

#  Load  Action-Recognition Model & Scaler ===
action_model_path = r"models\trained_action_model.pkl"
action_scaler_path = r"models\scaler.pkl"

action_clf = joblib.load(action_model_path)
action_scaler = joblib.load(action_scaler_path)
print("Action recognition model & scaler loaded.")


# --- Rally Detection Model ---
rally_model = load_model('/models/tennis_game_rally_model_final.h5' , compile=False  # Disable automatic compilation
)

# Manual compilation with safe settings
rally_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    run_eagerly=False,
    experimental_run_tf_function=False
)
print("Rally detection model loaded.")


print(f"Court Model on GPU: {next(court_model.parameters()).is_cuda}")
print(f"TrackNet Model on GPU: {next(tracknet_model.parameters()).is_cuda}")
print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))
print(f"YOLO model device: {yolo_model.device}")


# Load API key from environment variable
# Initialize OpenAI client



# Global EasyOCR reader
reader = easyocr.Reader(['en'])


client = OpenAI(api_key="OPEN_AI_KEY")  # Replace with your actual API key






# Add this class for managing pending predictions
class PendingPrediction:
    def __init__(self, start_frame_idx):
        self.start_frame = start_frame_idx
        self.ball_positions = []
        self.max_frames = 15  # Prevents predictions from staying in memory forever

    def add_position(self, ball_pos):
        if ball_pos[0] is not None and ball_pos[1] is not None:
            self.ball_positions.append(ball_pos)

    def is_complete(self):
        valid_count = sum(1 for p in self.ball_positions if p[0] is not None and p[1] is not None)
        return valid_count >= 10

    def has_expired(self, current_frame):
        return (current_frame - self.start_frame) > self.max_frames

    def get_prediction(self):
        if not self.is_complete():
            return f"DEBUG: Prediction incomplete ({len(self.ball_positions)} positions stored)"

        valid = [p for p in self.ball_positions if p[0] is not None and p[1] is not None]
        if len(valid) < 2:
            return "DEBUG: Not enough valid ball positions"

        x_start, y_start = valid[0]
        x_end, y_end = valid[-1]

        angle = np.degrees(np.arctan2(y_end - y_start, x_end - x_start))
        angle = (angle + 360) % 360

        if 45 <= angle < 135:
            return "Down the line"
        elif 135 <= angle < 225:
            return "Cross-court"
        elif 225 <= angle < 315:
            return "Behind baseline"
        else:
            return "Straight ahead"


###########################################
# 2) HELPER FUNCTIONS
###########################################
def measure_distance_2d(p1, p2):
    """Euclidean distance between two points p1 and p2."""
    if not p1 or not p2:
        return None
    if p1[0] is None or p1[1] is None or p2[0] is None or p2[1] is None:
        return None
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_bbox_center(bbox):
    """Return (x_center, y_center) of given bbox [x1, y1, x2, y2]."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_center_bottom_of_bbox(bbox):
    """Return (x_center, y_bottom) for bounding box's bottom edge."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)

def predict_court_keypoints(frame, court_model, transform, device):
    """Predict 14 base keypoints for the tennis court."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = court_model(inp)
    kps = out.reshape(-1, 2)  # Avoid CPU transfer unless necessary

    h, w = frame.shape[:2]
    kps[:, 0] *= w / 224.0
    kps[:, 1] *= h / 224.0
    return kps

def calculate_additional_keypoints(kps):
    """
    Add 4 extra derived points (midpoints, etc.) if indices 4..7 exist.
    """
    if len(kps) < 8:
        return kps.tolist()
    bir = kps[7]
    bil = kps[5]
    tir = kps[6]
    til = kps[4]

    bml = [(bir[0] + bil[0]) / 2, (bir[1] + bil[1]) / 2]
    tmu = [(tir[0] + til[0]) / 2, (tir[1] + til[1]) / 2]
    tmu_tir_mid = [(tmu[0] + tir[0]) / 2, (tmu[1] + tir[1]) / 2]
    tmu_til_mid = [(tmu[0] + til[0]) / 2, (tmu[1] + til[1]) / 2]

    return kps.tolist() + [bml, tmu, tmu_tir_mid, tmu_til_mid]

def detect_players_and_net(frame, yolo_model):
    """Use YOLO to detect only 'player' and 'net' (no ball)."""
    results = yolo_model(frame)[0]
    players = []
    net = None
    for box in results.boxes:
        xyxy = box.xyxy.tolist()[0]
        cls_id = int(box.cls.tolist()[0])
        cls_name = results.names[cls_id]
        if cls_name == "player":
            players.append(xyxy)
        elif cls_name == "net":
            net = xyxy
    return players, net

def detect_pose_on_player(frame, bbox, pose_model):
    """Crop bounding box from frame, run YOLOv8 pose, map keypoints => original coords."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x1 >= x2 or y1 >= y2:
        return None

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    pose_results = pose_model(cropped)[0]
    if len(pose_results.boxes) == 0 or not hasattr(pose_results, 'keypoints'):
        return None

    boxes = pose_results.boxes.xyxy.cpu().numpy()
    kpts_xy = pose_results.keypoints.xy.cpu().numpy()
    kpts_conf = pose_results.keypoints.conf.cpu().numpy()

    # pick largest detection
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    largest_idx = np.argmax(areas)

    pose_keypoints = np.hstack((
        kpts_xy[largest_idx],
        kpts_conf[largest_idx].reshape(-1, 1)
    ))  # shape: (17,3)

    # map back
    pose_keypoints[:,0] += x1
    pose_keypoints[:,1] += y1
    return pose_keypoints

def choose_and_filter_players(court_kps, players, frame_height):
    """
    Picks exactly one 'top player' and one 'bottom player'
    using distance to top/bottom 'court keypoints' + vertical threshold.
    """
    if len(court_kps) < 8:
        return None, None

    top_pts = [court_kps[4], court_kps[6]]
    bottom_pts = [court_kps[5], court_kps[7]]

    top_bbox = None
    bottom_bbox = None
    min_top_dist = float('inf')
    min_bottom_dist = float('inf')
    vertical_mid = frame_height / 2.0

    for bbox in players:
        c = get_bbox_center(bbox)
        btm = get_center_bottom_of_bbox(bbox)
        if c is None or btm is None:
            continue

        # top player candidate
        if c[1] < vertical_mid:
            for kp in top_pts:
                d = measure_distance_2d(btm, kp)
                if d is not None and d < min_top_dist:
                    min_top_dist = d
                    top_bbox = bbox
        else:
            # bottom player candidate
            for kp in bottom_pts:
                d = measure_distance_2d(c, kp)
                if d is not None and d < min_bottom_dist:
                    min_bottom_dist = d
                    bottom_bbox = bbox

    return top_bbox, bottom_bbox



def choose_and_filter_players_with_midpoints(court_keypoints, players):
    """
    This function assigns exactly one bounding box to each
    top/bottom set by picking the minimal distance from newly
    created midpoints and assigned keypoints.

    Returns:
      top_player, bottom_player: bounding boxes for top/bottom
      top_deciding_lines: list of (kp, bbox_center) for the chosen top bounding box
      bottom_deciding_lines: list of (kp, bbox_center) for the chosen bottom bounding box
      all_lines: list of (kp, bbox_center) for all keypoints
    """

    # Quick check if we have enough keypoints or no players
    if len(court_keypoints) < 14 or not players:
        return None, None, [], [], []

    # --- Assign the baseline keypoints (top + bottom) ---
    top_indices = [0, 4, 6, 1, 8, 12, 9]
    bottom_indices = [2, 5, 10, 13, 11, 7, 3]

    # Convert to arrays
    top_keypoints = [court_keypoints[i] for i in top_indices]
    bottom_keypoints = [court_keypoints[i] for i in bottom_indices]

    # Create new midpoints
    def create_midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) / 2.0, (ptA[1] + ptB[1]) / 2.0)

    # Midpoint for top (0,1)
    if 0 < len(court_keypoints) and 1 < len(court_keypoints):
        midpoint_top = create_midpoint(court_keypoints[0], court_keypoints[1])
        top_keypoints.append(midpoint_top)

    # Midpoint for bottom (2,3)
    if 2 < len(court_keypoints) and 3 < len(court_keypoints):
        midpoint_bottom = create_midpoint(court_keypoints[2], court_keypoints[3])
        bottom_keypoints.append(midpoint_bottom)

    # Prepare results
    top_player = None
    bottom_player = None
    min_top_dist = float('inf')
    min_bottom_dist = float('inf')
    top_deciding_lines = []
    bottom_deciding_lines = []
    all_lines = []

    # Precompute bottom center of each bounding box
    player_centers = []
    for bbox in players:
        x1, y1, x2, y2 = bbox
        center_pt = (
            (x1 + x2) // 2,
            int(y2)
        )
        player_centers.append((bbox, center_pt))

    # --- For each top keypoint, find bounding box with minimal distance ---
    for tk in top_keypoints:
        best_dist = float('inf')
        best_bbox = None
        best_center = None
        for (bbox, center_pt) in player_centers:
            dist = ((center_pt[0] - tk[0])**2 + (center_pt[1] - tk[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_bbox = bbox
                best_center = center_pt

        # White line for all lines
        all_lines.append((tk, best_center))

        # Update global top player if new min
        if best_dist < min_top_dist:
            min_top_dist = best_dist
            top_player = best_bbox
            top_deciding_lines = [(tk, best_center)]

    # --- For each bottom keypoint, find bounding box with minimal distance ---
    for bk in bottom_keypoints:
        best_dist = float('inf')
        best_bbox = None
        best_center = None
        for (bbox, center_pt) in player_centers:
            dist = ((center_pt[0] - bk[0])**2 + (center_pt[1] - bk[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_bbox = bbox
                best_center = center_pt

        all_lines.append((bk, best_center))

        # Update global bottom player if new min
        if best_dist < min_bottom_dist:
            min_bottom_dist = best_dist
            bottom_player = best_bbox
            bottom_deciding_lines = [(bk, best_center)]

    return top_player, bottom_player, top_deciding_lines, bottom_deciding_lines, all_lines






def add_movement_and_direction_features(df):
    """Add movement metrics + directional info, based on feet coords."""
    MOVEMENT_THRESHOLD = 50
    df['top_player_vertical_movement'] = df['top_player_feet_y'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)
    df['bottom_player_vertical_movement'] = df['bottom_player_feet_y'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)

    df['top_player_horizontal_movement'] = df['top_player_feet_x'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)
    df['bottom_player_horizontal_movement'] = df['bottom_player_feet_x'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)

    df['top_player_direction'] = np.arctan2(df['top_player_feet_y'].diff(), df['top_player_feet_x'].diff())
    df['bottom_player_direction'] = np.arctan2(df['bottom_player_feet_y'].diff(), df['bottom_player_feet_x'].diff())

    return df

def add_hit_detection_with_cooldown(df, bbox_expansion=25, cooldown=10):
    """
    Detects hits by checking if the ball enters the player's bounding box.
    Bounding box is expanded by `bbox_expansion` pixels on each side.
    A 10-frame cooldown prevents multiple detections for the same hit.
    """
    # Expand bounding boxes
    df['top_player_x1_exp'] = df['top_player_x1'] - bbox_expansion
    df['top_player_y1_exp'] = df['top_player_y1'] - bbox_expansion
    df['top_player_x2_exp'] = df['top_player_x2'] + bbox_expansion
    df['top_player_y2_exp'] = df['top_player_y2'] + bbox_expansion

    df['bottom_player_x1_exp'] = df['bottom_player_x1'] - bbox_expansion
    df['bottom_player_y1_exp'] = df['bottom_player_y1'] - bbox_expansion
    df['bottom_player_x2_exp'] = df['bottom_player_x2'] + bbox_expansion
    df['bottom_player_y2_exp'] = df['bottom_player_y2'] + bbox_expansion

    # Check if ball enters expanded bounding box
    df['top_player_hit'] = (
        (df['ball_x'] >= df['top_player_x1_exp']) & (df['ball_x'] <= df['top_player_x2_exp']) &
        (df['ball_y'] >= df['top_player_y1_exp']) & (df['ball_y'] <= df['top_player_y2_exp'])
    )

    df['bottom_player_hit'] = (
        (df['ball_x'] >= df['bottom_player_x1_exp']) & (df['ball_x'] <= df['bottom_player_x2_exp']) &
        (df['ball_y'] >= df['bottom_player_y1_exp']) & (df['ball_y'] <= df['bottom_player_y2_exp'])
    )

    # Apply cooldown logic
    last_hit = -np.inf
    for i in range(len(df)):
        if df.at[i, 'top_player_hit'] or df.at[i, 'bottom_player_hit']:
            if i - last_hit > cooldown:
                last_hit = i
            else:
                df.at[i, 'top_player_hit'] = False
                df.at[i, 'bottom_player_hit'] = False

    df['top_player_hit'] = df['top_player_hit'].astype(int)
    df['bottom_player_hit'] = df['bottom_player_hit'].astype(int)
    df['ball_hit_frame'] = (df['top_player_hit'] | df['bottom_player_hit']).astype(int)

    return df




def tracknet_infer_ball_positions(
    buffer_3_frames,
    tracknet_model,
    device,
    resized_w=640,  # Typically the width TrackNet expects
    resized_h=360   # Typically the height TrackNet expects
):
   

    if len(buffer_3_frames) < 3:
       #  print("DEBUG: buffer_3_frames has fewer than 3 frames. Returning None.")
        return (None, None)

    # 1) Show shapes of original frames
   #  print("DEBUG: Original frame shapes:")
    for i, f in enumerate(buffer_3_frames):
        print(f"  Frame {i}: {f.shape}")

    # 2) Resize frames
    frame_oldest = cv2.resize(buffer_3_frames[0], (resized_w, resized_h))
    frame_middle = cv2.resize(buffer_3_frames[1], (resized_w, resized_h))
    frame_current = cv2.resize(buffer_3_frames[2], (resized_w, resized_h))

   #  print("DEBUG: After resizing => shapes:")
    # print(f"  Oldest:  {frame_oldest.shape}")
   #  print(f"  Middle:  {frame_middle.shape}")
   #  print(f"  Current: {frame_current.shape}")

    # 3) Concatenate frames => [H, W, 9] and reorder => [9, H, W]
    combined = np.concatenate((frame_current, frame_middle, frame_oldest), axis=2).astype(np.float32) / 255.0
  #   print("DEBUG: combined.shape before rollaxis =", combined.shape)  # Expect (360,640,9)

    combined = np.rollaxis(combined, 2, 0)  # => shape [9, H, W]
    # print("DEBUG: combined.shape after rollaxis =", combined.shape)   # Expect (9, 360, 640)

    # 4) Turn into PyTorch tensor [1, 9, H, W]
    inp = torch.from_numpy(combined).unsqueeze(0).to(device)
   #  print("DEBUG: inp.shape for the model =", inp.shape)  # Expect [1, 9, 360, 640]

    # 5) Forward pass
    with torch.no_grad():
        out = tracknet_model(inp)  # Expected shape [1, C, 360, 640]
    # print("DEBUG: out.shape =", out.shape)

    # 6) Argmax over channel dim => shape [B, H, W]
    seg = out.argmax(dim=1)
   #  print("DEBUG: seg.shape after argmax =", seg.shape)   # Expect [1, 360, 640]

    segmap = seg[0].cpu().numpy()
    # print("DEBUG: segmap.shape after indexing [0] =", segmap.shape)  # Expect (360, 640)

    # 7) If segmap is 1D, forcibly reshape
    if segmap.ndim == 1:
        # We guess it should be (resized_h, resized_w)
       #  print("WARNING: segmap is 1D. Attempting to reshape => (resized_h, resized_w).")
        segmap = segmap.reshape(resized_h, resized_w)
       #  print("DEBUG: segmap.shape after forced reshape =", segmap.shape)

    # 8) Find argmax index => unravel => (y_pred, x_pred)
    idx_max = segmap.argmax()
    # print("DEBUG: idx_max =", idx_max)
    # print("DEBUG: segmap.shape =", segmap.shape)

    y_pred, x_pred = np.unravel_index(idx_max, segmap.shape)
    # print(f"DEBUG: (y_pred, x_pred) => ({y_pred}, {x_pred})")

    # 9) Scale coordinates back to original resolution
    orig_h, orig_w = buffer_3_frames[0].shape[:2]  # Get original frame dimensions
    scale_x = orig_w / float(resized_w)  # e.g., 1920 / 640 = 3.0
    scale_y = orig_h / float(resized_h)  # e.g., 1080 / 360 = 3.0

    final_x = int(x_pred * scale_x)
    final_y = int(y_pred * scale_y)

    # print(f"DEBUG: Scaled ball position => ({final_x}, {final_y})")

    return (final_x, final_y)




###########################################
# 3) ACTION-PREDICTION HELPER (Window-based)
###########################################
def predict_action_for_window(window_rows, selected_features, action_clf, action_scaler):
    """
    Given a subset of row dicts (or DataFrame subset),
    1) fill NaNs,
    2) scale using the pre-fit scaler,
    3) predict using the loaded model,
    4) return the majority vote label.
    """
    df_window = pd.DataFrame(window_rows)

    # rename if needed:
    if 'ball_x' in df_window.columns:
        df_window.rename(columns={'ball_x':'ball_center_x', 'ball_y':'ball_center_y'},
                         inplace=True)

    # fill missing columns with 0
    for col in selected_features:
        if col not in df_window.columns:
            df_window[col] = 0

    X_window = df_window[selected_features].fillna(0)
    X_scaled = action_scaler.transform(X_window)

    preds = action_clf.predict(X_scaled)
    values, counts = np.unique(preds, return_counts=True)
    return values[np.argmax(counts)]


def calculate_baseline_distances(court_keypoints, top_player_bbox, bottom_player_bbox):
    """
    Calculate the vertical distance from each player's feet to their respective baseline.
    - Uses the Y-coordinates of court keypoints to determine the baseline.
    - Uses the bottom Y-coordinate of the player bounding box as their foot position.
    """
    top_baseline_y = (court_keypoints[0][1] + court_keypoints[1][1]) / 2  # Average of top left & top right
    bottom_baseline_y = (court_keypoints[2][1] + court_keypoints[3][1]) / 2  # Average of bottom left & bottom right

    def get_player_feet_y(player_bbox):
        if player_bbox:
            return player_bbox[3]  # Y2 = bottom of bounding box (player feet)
        return None

    top_player_feet_y = get_player_feet_y(top_player_bbox)
    bottom_player_feet_y = get_player_feet_y(bottom_player_bbox)

    # Compute vertical distances
    top_player_baseline_distance = abs(top_player_feet_y - top_baseline_y) if top_player_feet_y is not None else None
    bottom_player_baseline_distance = abs(bottom_player_feet_y - bottom_baseline_y) if bottom_player_feet_y is not None else None

    return top_player_baseline_distance, bottom_player_baseline_distance


def add_movement_and_direction_features(df):
    """
    Adds movement-related features:
    - Vertical & horizontal movement (absolute pixel values)
    - Horizontal direction (left/right movement)
    - Ball movement angle (radians & degrees)
    """
    MOVEMENT_THRESHOLD = 20  # Pixels per frame (to filter outliers)

    # Compute movement distances (limit extreme jumps)
    df['top_player_vertical_movement_pixels'] = df['top_player_feet_y'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)
    df['bottom_player_vertical_movement_pixels'] = df['bottom_player_feet_y'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)
    df['top_player_horizontal_movement_pixels'] = df['top_player_feet_x'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)
    df['bottom_player_horizontal_movement_pixels'] = df['bottom_player_feet_x'].diff().abs().clip(upper=MOVEMENT_THRESHOLD)

    # Determine movement direction (left/right)
    df['top_player_horizontal_direction'] = df['top_player_feet_x'].diff().apply(
        lambda x: 'Right' if x > 0 else 'Left' if x < 0 else None
    )
    df['bottom_player_horizontal_direction'] = df['bottom_player_feet_x'].diff().apply(
        lambda x: 'Right' if x > 0 else 'Left' if x < 0 else None
    )

    # Compute ball movement direction
    df['ball_direction'] = np.arctan2(df['ball_y'].diff(), df['ball_x'].diff())
    df['ball_direction_degrees'] = np.degrees(df['ball_direction'])

    return df

def compute_player_feet_positions(df):
    """
    Compute feet positions using ankle keypoints (if available).
    If ankle keypoints are missing, default to using bounding box bottom (y2).
    """

    # Average ankle keypoints for feet position
    df['top_player_feet_x'] = (df['top_player_kp15_x'] + df['top_player_kp16_x']) / 2
    df['top_player_feet_y'] = (df['top_player_kp15_y'] + df['top_player_kp16_y']) / 2

    df['bottom_player_feet_x'] = (df['bottom_player_kp15_x'] + df['bottom_player_kp16_x']) / 2
    df['bottom_player_feet_y'] = (df['bottom_player_kp15_y'] + df['bottom_player_kp16_y']) / 2

    # Fallback to bounding box bottom (y2) if keypoints are missing
    df['top_player_feet_y'].fillna(df['top_player_y2'], inplace=True)
    df['bottom_player_feet_y'].fillna(df['bottom_player_y2'], inplace=True)

    df['top_player_feet_x'].fillna((df['top_player_x1'] + df['top_player_x2']) / 2, inplace=True)
    df['bottom_player_feet_x'].fillna((df['bottom_player_x1'] + df['bottom_player_x2']) / 2, inplace=True)

    return df


def get_baseline_sections(court_kps, player_type):
    """Get left, middle, right x-coordinates for baseline sections."""
    if player_type == "top":
        # Keypoints 4, 15, 6 (assuming order is left, middle, right)
        left_x = court_kps[4][0]
        middle_x = court_kps[15][0]
        right_x = court_kps[6][0]
    elif player_type == "bottom":
        # Keypoints 5, 14, 7
        left_x = court_kps[5][0]
        middle_x = court_kps[14][0]
        right_x = court_kps[7][0]
    else:
        return None, None, None
    return left_x, middle_x, right_x

def get_horizontal_zone(feet_x, left_x, middle_x, right_x):
    """Classify player's horizontal position as left/middle/right."""
    if feet_x is None or None in [left_x, middle_x, right_x]:
        return None
    if feet_x < left_x:
        return "left"
    elif left_x <= feet_x < middle_x:
        return "middle"
    elif middle_x <= feet_x < right_x:
        return "middle"  # Optional: Add "middle_right" if finer zones needed
    else:
        return "right"


def predict_ball_direction(ball_positions, current_frame_idx, max_frames_ahead=10):
    """
    Predicts the ball's direction after a hit using the next 10 frames.

    Parameters:
    - ball_positions (list of tuples): A list containing the ball's (x, y) positions for each frame.
    - current_frame_idx (int): The index of the current frame where the hit is detected.
    - max_frames_ahead (int): The number of frames to look ahead after the hit to predict the direction.

    Returns:
    - str or None: A direction label (e.g., "Down the line", "Cross-court") or None if insufficient data.
    """

    # Step 1: Determine the range of frames to analyze after the hit
    start_idx = current_frame_idx + 1  # Start from the frame immediately after the hit
    end_idx = min(current_frame_idx + max_frames_ahead + 1, len(ball_positions))  # Ensure we don't exceed the total frames

    # Extract the ball's trajectory in the next 10 frames (or fewer if near the end of the video)
    trajectory = ball_positions[start_idx:end_idx]

    # Step 2: Filter out invalid positions (where the ball is not detected, i.e., x or y is None)
    valid_points = [(x, y) for x, y in trajectory if x is not None and y is not None]

    # If there are fewer than 2 valid points, we can't calculate a direction
    if len(valid_points) < 2:
        return None  # Insufficient data to predict direction

    # Step 3: Calculate the direction vector using the first and last valid points
    x_start, y_start = valid_points[0]  # Ball's position in the first valid frame after the hit
    x_end, y_end = valid_points[-1]    # Ball's position in the last valid frame within the 10-frame window

    # Step 4: Calculate the angle of the ball's movement relative to the horizontal axis
    # - np.arctan2(dy, dx) calculates the angle in radians between the positive x-axis and the line to (x_end, y_end)
    # - dy = y_end - y_start (change in y-coordinate)
    # - dx = x_end - x_start (change in x-coordinate)
    angle = np.degrees(np.arctan2(y_end - y_start, x_end - x_start))

    # Normalize the angle to a range of 0° to 359° to simplify direction classification
    angle = (angle + 360) % 360  # Ensures the angle is always positive

    # Step 5: Classify the ball's direction based on the angle
    # - 0°: Ball moving horizontally to the right
    # - 90°: Ball moving vertically downward
    # - 180°: Ball moving horizontally to the left
    # - 270°: Ball moving vertically upward
    # The court's geometry is taken into account when assigning direction labels:
    # - "Down the line": Ball is moving diagonally toward the sideline (45° to 135°)
    # - "Cross-court": Ball is moving diagonally toward the opposite corner (135° to 225°)
    # - "Behind baseline": Ball is moving backward toward the baseline (225° to 315°)
    # - "Straight ahead": Ball is moving forward or sideways (other angles)

    if 45 <= angle < 135:
        return "Down the line"  # Ball is moving diagonally toward the sideline
    elif 135 <= angle < 225:
        return "Cross-court"    # Ball is moving diagonally toward the opposite corner
    elif 225 <= angle < 315:
        return "Behind baseline"  # Ball is moving backward toward the baseline
    else:
        return "Straight ahead"  # Ball is moving forward or sideways

def calculate_net_proximity(df):
    """
    Computes proximity of players to the net by calculating their vertical distance from the net's bottom.
    A player is considered 'near the net' if their distance is smaller than their baseline distance.
    """
    # Compute the net bottom Y position
    df['net_bottom_y'] = df[['net_y1', 'net_y2']].max(axis=1)

    # Compute vertical distance from players' feet to net bottom
    df['top_player_net_vertical_dist'] = (df['top_player_feet_y'] - df['net_bottom_y']).abs()
    df['bottom_player_net_vertical_dist'] = (df['bottom_player_feet_y'] - df['net_bottom_y']).abs()

    # Determine if the player is near the net
    df['top_player_near_net'] = (df['top_player_net_vertical_dist'] < df['top_player_baseline_distance']).fillna(False).astype(int)
    df['bottom_player_near_net'] = (df['bottom_player_net_vertical_dist'] < df['bottom_player_baseline_distance']).fillna(False).astype(int)

    # Drop the temporary net bottom Y column if not needed
    df.drop(columns=['net_bottom_y'], inplace=True, errors='ignore')

    return df

def compute_movement_direction(previous_pos, current_pos):
    """
    Compute the direction of movement from previous to current position.
    Returns one of: "Up", "Down", "Left", "Right", "Up-Left", "Up-Right", "Down-Left", "Down-Right"
    """
    if previous_pos is None or current_pos is None:
        return None

    prev_x, prev_y = previous_pos
    curr_x, curr_y = current_pos

    dx = curr_x - prev_x
    dy = curr_y - prev_y

    # Define movement thresholds (to avoid tiny fluctuations)
    THRESHOLD = 10  # Pixels

    if abs(dx) < THRESHOLD and abs(dy) < THRESHOLD:
        return "Stationary"

    if abs(dx) > abs(dy):  # Horizontal movement is dominant
        if dx > 0:
            return "Right"
        else:
            return "Left"
    elif abs(dy) > abs(dx):  # Vertical movement is dominant
        if dy > 0:
            return "Down"
        else:
            return "Up"
    else:  # Diagonal movement
        if dx > 0 and dy > 0:
            return "Down-Right"
        elif dx > 0 and dy < 0:
            return "Up-Right"
        elif dx < 0 and dy > 0:
            return "Down-Left"
        elif dx < 0 and dy < 0:
            return "Up-Left"

def compute_player_trajectory(position_buffer):
    """
    Computes player trajectory over the last 10 frames.
    Uses the angle between the first and last valid points in the buffer.
    """
    # Filter out None values
    valid_positions = [(x, y) for x, y in position_buffer if x is not None and y is not None]

    if len(valid_positions) < 2:
        return None  # Not enough data to compute trajectory

    start_x, start_y = valid_positions[0]
    end_x, end_y = valid_positions[-1]

    angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
    angle = (angle + 360) % 360  # Normalize angle to 0-359 degrees

    if 45 <= angle < 135:
        return "Moving Down"
    elif 135 <= angle < 225:
        return "Moving Left"
    elif 225 <= angle < 315:
        return "Moving Up"
    else:
        return "Moving Right"

def compute_player_movement_and_trajectory(position_buffer):
    """
    Computes player movement direction and trajectory using the last 10 valid frames.
    - Movement is determined based on the overall displacement trend.
    - Trajectory is calculated using the angle between the first and last valid positions.

    Returns:
        movement_direction (str): General movement direction over the last 10 frames.
        trajectory (str): The player's trajectory based on the start and end points.
    """
    # Filter out None values
    valid_positions = [(x, y) for x, y in position_buffer if x is not None and y is not None]

    if len(valid_positions) < 2:
        return None, None  # Not enough data to compute

    # First and last points for trajectory calculation
    start_x, start_y = valid_positions[0]
    end_x, end_y = valid_positions[-1]

    # Compute movement direction based on displacement
    dx_total = end_x - start_x
    dy_total = end_y - start_y

    # Compute trajectory angle
    angle = np.degrees(np.arctan2(dy_total, dx_total))
    angle = (angle + 360) % 360  # Normalize to 0-359 degrees

    # Determine trajectory category
    if 45 <= angle < 135:
        trajectory = "Moving Down"
    elif 135 <= angle < 225:
        trajectory = "Moving Left"
    elif 225 <= angle < 315:
        trajectory = "Moving Up"
    else:
        trajectory = "Moving Right"

    # Determine movement direction based on dominant displacement
    if abs(dx_total) > abs(dy_total):  # Horizontal movement dominant
        movement_direction = "Right" if dx_total > 0 else "Left"
    elif abs(dy_total) > abs(dx_total):  # Vertical movement dominant
        movement_direction = "Down" if dy_total > 0 else "Up"
    else:  # Diagonal movement
        if dx_total > 0 and dy_total > 0:
            movement_direction = "Down-Right"
        elif dx_total > 0 and dy_total < 0:
            movement_direction = "Up-Right"
        elif dx_total < 0 and dy_total > 0:
            movement_direction = "Down-Left"
        elif dx_total < 0 and dy_total < 0:
            movement_direction = "Up-Left"
        else:
            movement_direction = "Stationary"

    return movement_direction, trajectory




def generate_tennis_commentary(frames_data):
    """
    Uses OpenAI GPT API to generate fast-paced tennis commentary
    based on the last 10 processed frames.
    """


    # Construct the prompt for ChatGPT
    prompt = (
        "You are a professional live tennis commentator. Your goal is to provide concise, high-energy, and technically insightful commentary based on the real-time match data of the last 10 frames. Your focus is to deliver **engaging, dynamic, and precise** play-by-play analysis, just as a real commentator would during a match."
        "based on the match data of the past 10 frames. Keep the commentary engaging, dynamic, and relevant to the play happening. "
        "**Key Guidelines:**"
        "Jump straight into the action**—no openers, fillers, or setup phrases.\n\n"
        "Use only natural tennis terminology**—avoid terms like “predicted” or any programming-related language.\n\n"
        "Prioritize key match details:** shot type, ball trajectory, player movement, positioning, and impact on the rally.\n\n"
        "all of this data is in mind game so do not do any openers jump start into action of what is happening\n\n"
        "Be emotionally engaging:** highlight intensity, pressure, and tactical shifts when relevant.\n\n"
         "keep reponses short and more direct this is executed after player hits a shot in a tennis game  \n\n"
         "Keep it fluid and natural:** structure sentences smoothly and ensure seamless transitions.   \n\n"
        "Limit response to ONE sentence.** No extra information beyond what the game data provides."

    )

    for frame in frames_data:
        prompt += (
            f"- Frame: {frame['frame']}\n"
            f"- Rally Ongoing: {'Yes' if frame['is_rally'] else 'No'}\n"
            f"- Predicted Action: {frame.get('action_recognition', 'N/A')}\n"
            f"- Ball Hit: {'Top Player' if frame['top_player_hit'] else 'Bottom Player' if frame['bottom_player_hit'] else 'No'}\n"
            f"- Predicted Ball Direction: {frame.get('predicted_ball_direction', 'Unknown')}\n"
            f"- Top Player Position: {frame['top_player_horizontal_zone']} | Near Net: {frame['top_player_near_net']}\n"
            f"- Bottom Player Position: {frame['bottom_player_horizontal_zone']} | Near Net: {frame['bottom_player_near_net']}\n"
            f"- Top Player Movement: {frame.get('top_player_direction', 'Unknown')} | Trajectory: {frame.get('top_player_trajectory', 'Unknown')}\n"
            f"- Bottom Player Movement: {frame.get('bottom_player_direction', 'Unknown')} | Trajectory: {frame.get('bottom_player_trajectory', 'Unknown')}\n"
        )

    prompt += "\nNow, **deliver a single, high-impact, play-by-play commentary line** based on the **shot action, ball trajectory, player movement, and rally context** in a way that feels natural for a live broadcast."


     # Print the generated commentary for debugging
    print("\n======= DEBUG: ChatGPT Response =======")
    print(prompt)
    print("========================================\n")
    # Make API request using the correct OpenAI structure
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    # Extract response text
    commentary = response.choices[0].message.content.strip()
    return commentary








def generate_non_rally_commentary_from_ocr(ocr_text):
    """
    Generates tennis commentary using OCR-extracted text during non-rally moments
    such as breaks, ads, between rallies, or stats display.

    Parameters:
        ocr_text (str): OCR-extracted text from the current frame.

    Returns:
        commentary (str): Generated commentary from ChatGPT.
    """

    # Construct the ChatGPT prompt for non-rally scenario
    prompt = (
        "You are an expert tennis commentator currently covering a live tennis match. "
        "this can be before match start, during a break, advertisement, statistics display, or between rallies,use your reasoning to identify and adapt you rcommentary as where you think which part of the match,  this is you see the text extracted via OCR: \n\n"
        f"'{ocr_text}'\n\n"
        "Provide a concise, engaging, and insightful commentary analyzing or summarizing what is currently displayed. "
        "Do not directly repeat OCR details; instead, use them to enrich your commentary contextually. "
        "Keep the tone professional, informative, and engaging, setting the stage for the upcoming rally or summarizing match progress."
        ""
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    commentary = response.choices[0].message.content.strip()
    return commentary







def frame_to_text_with_grouping_single_method(frame):
    """
    Performs OCR on a frame using EasyOCR, groups nearby text lines, sorts them,
    and returns a single formatted string with confidence scores.

    Returns:
        str: Formatted text with confidence scores, or empty string if no text found.
    """

    # 1. Perform text detection and recognition
    results = reader.readtext(frame, detail=1)  # Returns list of (bbox, text, conf)
    if not results:
        return ""

    # 2. Helper function to calculate bounding box center
    def get_bbox_center(bbox):
        """Returns (x_center, y_center) of a bounding box"""
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        return (sum(x_coords)/4, sum(y_coords)/4)

    # 3. Group nearby text items vertically
    vertical_threshold = 20  # Pixels between lines to consider same group
    groups = []

    for item in results:
        bbox, text, conf = item
        current_center = get_bbox_center(bbox)
        added = False

        # Check against existing groups
        for group in groups:
            for group_item in group:
                g_bbox, _, _ = group_item
                group_center = get_bbox_center(g_bbox)

                # Vertical proximity check
                if abs(current_center[1] - group_center[1]) <= vertical_threshold:
                    group.append(item)
                    added = True
                    break
            if added:
                break

        # Create new group if no nearby group found
        if not added:
            groups.append([item])

    # 4. Sort groups vertically and items horizontally
    # Sort each group's items left-to-right
    for group in groups:
        group.sort(key=lambda x: get_bbox_center(x[0])[0])  # Sort by x-center

    # Sort groups by vertical position (using top-most item's Y coordinate)
    groups.sort(key=lambda g: min([item[0][0][1] for item in g]))

    # 5. Format final output
    output_lines = []
    for group in groups:
        group_texts = []
        for bbox, text, conf in group:
            group_texts.append(f"{text.strip()} (conf={conf:.2f})")
        output_lines.append(" ".join(group_texts))

    return " | ".join(output_lines) if output_lines else ""



###########################################
# 4) REAL-TIME PROCESSING WITH IMMEDIATE HIT DETECTION + ACTION
###########################################
def process_single_video_with_ball_realtime(
    video_path,
    csv_output_path,
    rally_model,
    court_model,
    yolo_model,
    pose_model,
    tracknet_model,
    action_clf,
    action_scaler,
    transform,
    device
):


    # Add these variables for prediction tracking
    pending_predictions = []
    prediction_results = {}
    top_player_positions = deque(maxlen=5)
    bottom_player_positions = deque(maxlen=5)


    tracknet_buffer = deque(maxlen=3)  # For 3-frame TrackNet input
    ball_positions = []               # Store (bx, by) if needed for direction

    rows = []  # Will store final results for each frame in a list of dicts

    # For rally and hits tracking
    last_rally_frame_idx = None
    rally_hit_count = 0
    non_rally_counter = 0
    last_hit_frame_top = -math.inf
    last_hit_frame_bottom = -math.inf
    row_count = 0
    #  Manage court keypoints once per rally ======
    in_rally = False
    stored_court_kps = None  # We'll set this when a new rally starts

    # EXACT features used for your action recognition model
    selected_features = [
        "top_player_to_ball_distance", "bottom_player_to_ball_distance",
        "ball_to_net_distance", "ball_center_x", "ball_center_y",
        "top_player_x1", "top_player_x2", "top_player_y1", "top_player_y2",
        "bottom_player_x1", "bottom_player_x2", "bottom_player_y1", "bottom_player_y2",
        "top_left_wrist_dist", "top_right_wrist_dist",
        "bottom_left_wrist_dist", "bottom_right_wrist_dist",
        "net_x1", "net_x2", "net_y1", "net_y2",
        "players_feet_distance"
    ]
    # Add court keypoints
    for i in range(18):
        selected_features.append(f"court_kp_{i}_x")
        selected_features.append(f"court_kp_{i}_y")
    # Add pose keypoints
    for i in range(17):
        selected_features.append(f"top_player_kp{i}_x")
        selected_features.append(f"top_player_kp{i}_y")
        selected_features.append(f"top_player_kp{i}_conf")
        selected_features.append(f"bottom_player_kp{i}_x")
        selected_features.append(f"bottom_player_kp{i}_y")
        selected_features.append(f"bottom_player_kp{i}_conf")

    cap = cv2.VideoCapture(video_path)
    idx = 0

    frame_count = 0  # Counter for every 3rd frame processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames
        idx += 1

        # Skip frames that are not every 3rd frame
        if idx % 2 != 0:
            continue


        # ========== TrackNet (3-frame rolling buffer) ==========
        tracknet_buffer.append(frame)
        if len(tracknet_buffer) == 3:
            bx, by = tracknet_infer_ball_positions(tracknet_buffer, tracknet_model, device)
        else:
            bx, by = (None, None)

        ball_positions.append((bx, by))

        # ========== Rally Detection ==========
        rally_frame = cv2.resize(frame, (224, 224))
        rally_frame_rgb = cv2.cvtColor(rally_frame, cv2.COLOR_BGR2RGB)
        rally_frame_preproc = preprocess_input(rally_frame_rgb)
        rally_frame_preproc = np.expand_dims(rally_frame_preproc, axis=0)
        is_rally = rally_model.predict(rally_frame_preproc, verbose=0)[0][0] > 0.5





        row = {
            "frame": idx,
            "ball_x": bx,
            "ball_y": by,
            "is_rally": int(is_rally),
            "action_recognition": None,
            "top_player_hit": 0,
            "bottom_player_hit": 0,
            "ball_hit_frame": 0,
            "rally_hits": None
        }

        # Append the row
        #rows.append(row)

        # ========== If we are NOT in a rally, reset or skip logic ==========
        if not is_rally:
            if in_rally and last_rally_frame_idx is not None:
                if len(rows) > 0:  # Ensure there is at least one row
                    # Ensure last_rally_frame_idx is within valid range
                    safe_index = min(last_rally_frame_idx, len(rows) - 1)
                    rows[safe_index]["rally_hits"] = rally_hit_count
    
    
                row["top_player_horizontal_zone"] = None
                row["bottom_player_horizontal_zone"] = None
                row["top_player_near_net"] = None
                row["bottom_player_near_net"] = None
    
    
                # Reset rally-related stuff
                in_rally = False
                stored_court_kps = None
                rally_hit_count = 0
                last_rally_frame_idx = None
    
                # A new column so we can store recognized text per frame
                row["frame_text"] = None
    
                # (1) Increment non-rally counter
                non_rally_counter += 1
    
                # (2) Check if we have 10 consecutive non-rally frames
                if non_rally_counter >= 10:
                    # (3) Perform OCR on the current frame
                    recognized_text = frame_to_text_with_grouping_single_method(frame)
                    row["frame_text"] = recognized_text
                              
                    if recognized_text:  # Only make API call if OCR returned something
                        non_rally_commentary = generate_non_rally_commentary_from_ocr(recognized_text)
                        row["commentary"] = non_rally_commentary
                    else:
                        row["commentary"] = "No text recognized"
                    
                    # Reset the counter
                    non_rally_counter = 0
    
                # Append row for this frame, skip advanced logic
                rows.append(row)
                continue


        # ========== If we get here, is_rally == True ==========
        # Check if we just transitioned from not-in-rally to in-rally
        if not in_rally:
            # This is a NEW rally
            in_rally = True
            # Detect court keypoints ONCE for this entire rally
            base_kps = predict_court_keypoints(frame, court_model, transform, device)
            stored_court_kps = calculate_additional_keypoints(base_kps)
           #  print(f"[DEBUG] New rally started at frame={idx}. Court keypoints detected once.")

        # Update last_rally_frame_idx to track the current rally's last frame
        last_rally_frame_idx = idx

        # ========== Use stored court keypoints for the entire rally ==========
        if stored_court_kps is not None:
            extended_kps = stored_court_kps
            top_left_x, top_middle_x, top_right_x = get_baseline_sections(stored_court_kps, "top")
            bottom_left_x, bottom_middle_x, bottom_right_x = get_baseline_sections(stored_court_kps, "bottom")
        else:
            extended_kps = []

        # ========== YOLO => players, net ==========
        players, net_bbox = detect_players_and_net(frame, yolo_model)
        #top_bbox, bottom_bbox = choose_and_filter_players(extended_kps, players, frame.shape[0])

        top_bbox, bottom_bbox, _, _, _ = choose_and_filter_players_with_midpoints(extended_kps, players)




        # ========== Pose detection ==========
        top_pose = detect_pose_on_player(frame, top_bbox, pose_model)
        bottom_pose = detect_pose_on_player(frame, bottom_bbox, pose_model)

        # ========== Distances & Additional Info ==========
        if net_bbox is not None:
            row["net_x1"], row["net_y1"], row["net_x2"], row["net_y2"] = net_bbox
        else:
            row["net_x1"] = row["net_y1"] = None
            row["net_x2"] = row["net_y2"] = None


        # Compute net bottom Y position
        if net_bbox is not None:
            net_bottom_y = max(net_bbox[1], net_bbox[3])
        else:
            net_bottom_y = None


        # top bbox
        if top_bbox is not None:
            row["top_player_x1"] = top_bbox[0]
            row["top_player_y1"] = top_bbox[1]
            row["top_player_x2"] = top_bbox[2]
            row["top_player_y2"] = top_bbox[3]
        else:
            row["top_player_x1"] = row["top_player_y1"] = None
            row["top_player_x2"] = row["top_player_y2"] = None

        # bottom bbox
        if bottom_bbox is not None:
            row["bottom_player_x1"] = bottom_bbox[0]
            row["bottom_player_y1"] = bottom_bbox[1]
            row["bottom_player_x2"] = bottom_bbox[2]
            row["bottom_player_y2"] = bottom_bbox[3]
        else:
            row["bottom_player_x1"] = row["bottom_player_y1"] = None
            row["bottom_player_x2"] = row["bottom_player_y2"] = None

        # Store the *same* court keypoints for every frame in the rally
        for i_kp, pt in enumerate(extended_kps):
            row[f"court_kp_{i_kp}_x"] = pt[0]
            row[f"court_kp_{i_kp}_y"] = pt[1]

        # Pose keypoints
        if top_pose is not None:
            for i_kp in range(top_pose.shape[0]):
                row[f"top_player_kp{i_kp}_x"]    = top_pose[i_kp, 0]
                row[f"top_player_kp{i_kp}_y"]    = top_pose[i_kp, 1]
                row[f"top_player_kp{i_kp}_conf"] = top_pose[i_kp, 2]
        else:
            for i_kp in range(17):
                row[f"top_player_kp{i_kp}_x"]    = None
                row[f"top_player_kp{i_kp}_y"]    = None
                row[f"top_player_kp{i_kp}_conf"] = None

        if bottom_pose is not None:
            for i_kp in range(bottom_pose.shape[0]):
                row[f"bottom_player_kp{i_kp}_x"]    = bottom_pose[i_kp, 0]
                row[f"bottom_player_kp{i_kp}_y"]    = bottom_pose[i_kp, 1]
                row[f"bottom_player_kp{i_kp}_conf"] = bottom_pose[i_kp, 2]
        else:
            for i_kp in range(17):
                row[f"bottom_player_kp{i_kp}_x"]    = None
                row[f"bottom_player_kp{i_kp}_y"]    = None
                row[f"bottom_player_kp{i_kp}_conf"] = None

        # measure distances...
        top_center = get_bbox_center(top_bbox)
        bottom_center = get_bbox_center(bottom_bbox)
        top_ball_dist = measure_distance_2d(top_center, (bx, by)) if bx is not None else None
        bottom_ball_dist = measure_distance_2d(bottom_center, (bx, by)) if bx is not None else None
        row["top_player_to_ball_distance"] = top_ball_dist
        row["bottom_player_to_ball_distance"] = bottom_ball_dist

        # net distance
        if net_bbox is not None and (by is not None):
            net_center_y = (net_bbox[1] + net_bbox[3]) / 2.0
            row["ball_to_net_distance"] = abs(by - net_center_y)
        else:
            row["ball_to_net_distance"] = None

        # distance between top & bottom
        feet_dist = measure_distance_2d(top_center, bottom_center)
        row["players_feet_distance"] = feet_dist

        # baseline
        top_player_baseline_distance, bottom_player_baseline_distance = calculate_baseline_distances(extended_kps, top_bbox, bottom_bbox)
        row["top_player_baseline_distance"] = top_player_baseline_distance
        row["bottom_player_baseline_distance"] = bottom_player_baseline_distance

        # feet positions from ankles or fallback
        if top_pose is not None:
            row["top_player_feet_x"] = (row["top_player_kp15_x"] + row["top_player_kp16_x"]) / 2
            row["top_player_feet_y"] = (row["top_player_kp15_y"] + row["top_player_kp16_y"]) / 2
        else:
            row["top_player_feet_x"], row["top_player_feet_y"] = None, None

        if bottom_pose is not None:
            row["bottom_player_feet_x"] = (row["bottom_player_kp15_x"] + row["bottom_player_kp16_x"]) / 2
            row["bottom_player_feet_y"] = (row["bottom_player_kp15_y"] + row["bottom_player_kp16_y"]) / 2
        else:
            row["bottom_player_feet_x"], row["bottom_player_feet_y"] = None, None

        # fallback if feet_y is None
        if row["top_player_feet_y"] is None and row["top_player_y2"] is not None:
            row["top_player_feet_y"] = row["top_player_y2"]
        if row["bottom_player_feet_y"] is None and row["bottom_player_y2"] is not None:
            row["bottom_player_feet_y"] = row["bottom_player_y2"]

        # Determine top player horizontal zone
        row["top_player_horizontal_zone"] = get_horizontal_zone(
            row["top_player_feet_x"], top_left_x, top_middle_x, top_right_x
        ) if stored_court_kps else None

        # Determine bottom player horizontal zone
        row["bottom_player_horizontal_zone"] = get_horizontal_zone(
            row["bottom_player_feet_x"], bottom_left_x, bottom_middle_x, bottom_right_x
        ) if stored_court_kps else None


        # net_center_y
        if net_bbox is not None:
            row["net_center_y"] = (row["net_y1"] + row["net_y2"]) / 2
        else:
            row["net_center_y"] = None



        # Compute vertical distance from players' feet to net bottom
        if net_bottom_y is not None:
            row["top_player_net_vertical_dist"] = abs(row["top_player_feet_y"] - net_bottom_y) if row["top_player_feet_y"] is not None else None
            row["bottom_player_net_vertical_dist"] = abs(row["bottom_player_feet_y"] - net_bottom_y) if row["bottom_player_feet_y"] is not None else None
        else:
            row["top_player_net_vertical_dist"] = None
            row["bottom_player_net_vertical_dist"] = None

        # Determine if the player is near the net (distance to net < baseline distance)
        if row["top_player_net_vertical_dist"] is not None and row["top_player_baseline_distance"] is not None:
            row["top_player_near_net"] = int(row["top_player_net_vertical_dist"] < row["top_player_baseline_distance"])
        else:
            row["top_player_near_net"] = None

        if row["bottom_player_net_vertical_dist"] is not None and row["bottom_player_baseline_distance"] is not None:
            row["bottom_player_near_net"] = int(row["bottom_player_net_vertical_dist"] < row["bottom_player_baseline_distance"])
        else:
            row["bottom_player_near_net"] = None

        # Update rolling buffers for player feet positions
        top_player_positions.append((row["top_player_feet_x"], row["top_player_feet_y"]))
        bottom_player_positions.append((row["bottom_player_feet_x"], row["bottom_player_feet_y"]))
        '''
        # Compute movement direction for the current frame (only if previous and current positions are valid)
        if len(top_player_positions) > 1 and all(top_player_positions[-2]) and all(top_player_positions[-1]):
            row["top_player_direction"] = compute_movement_direction(top_player_positions[-2], top_player_positions[-1])
        else:
            row["top_player_direction"] = None

        if len(bottom_player_positions) > 1 and all(bottom_player_positions[-2]) and all(bottom_player_positions[-1]):
            row["bottom_player_direction"] = compute_movement_direction(bottom_player_positions[-2], bottom_player_positions[-1])
        else:
            row["bottom_player_direction"] = None


        # Compute trajectory every 10 frames
        if len(top_player_positions) == 10:
            row["top_player_trajectory"] = compute_player_trajectory(top_player_positions)
        else:
            row["top_player_trajectory"] = None

        if len(bottom_player_positions) == 10:
            row["bottom_player_trajectory"] = compute_player_trajectory(bottom_player_positions)
        else:
            row["bottom_player_trajectory"] = None
        '''
        # Update rolling buffers for player feet positions
        top_player_positions.append((row["top_player_feet_x"], row["top_player_feet_y"]))
        bottom_player_positions.append((row["bottom_player_feet_x"], row["bottom_player_feet_y"]))

        # Compute movement and trajectory only every 10th frame
        if idx % 10 == 0:
            row["top_player_direction"], row["top_player_trajectory"] = compute_player_movement_and_trajectory(top_player_positions)
            row["bottom_player_direction"], row["bottom_player_trajectory"] = compute_player_movement_and_trajectory(bottom_player_positions)
        else:
            row["top_player_direction"], row["top_player_trajectory"] = None, None
            row["bottom_player_direction"], row["bottom_player_trajectory"] = None, None



        # ========== Handle pending predictions ==========
        # Add current ball position to all active predictions
        current_ball_pos = (bx, by)
        for pp in pending_predictions:
            pp.add_position(current_ball_pos)

        # Check for completed predictions and update rows
        completed = []
        for pp in pending_predictions:
            if pp.is_complete():
                # Get the predicted direction
                direction = pp.get_prediction()
                # Find the corresponding row (frame where the hit occurred)
                row_index = pp.start_frame - 1  # Assuming frame starts at 1
                if 0 <= row_index < len(rows):
                    rows[row_index]["predicted_ball_direction"] = direction
                completed.append(pp)

        # Remove completed predictions from tracking
        for pp in completed:
            pending_predictions.remove(pp)



        # ========== Immediate Hit Detection + Action Recognition ==========
        top_hit = False
        bottom_hit = False

        if (bx is not None) and (by is not None):
            if top_pose is not None:
                top_left_wrist_dist = math.hypot(bx - top_pose[9,0],  by - top_pose[9,1])
                top_right_wrist_dist = math.hypot(bx - top_pose[10,0], by - top_pose[10,1])
            else:
                top_left_wrist_dist = None
                top_right_wrist_dist = None

            if bottom_pose is not None:
                bottom_left_wrist_dist = math.hypot(bx - bottom_pose[9,0],  by - bottom_pose[9,1])
                bottom_right_wrist_dist = math.hypot(bx - bottom_pose[10,0], by - bottom_pose[10,1])
            else:
                bottom_left_wrist_dist = None
                bottom_right_wrist_dist = None

            row["top_left_wrist_dist"] = top_left_wrist_dist
            row["top_right_wrist_dist"] = top_right_wrist_dist
            row["bottom_left_wrist_dist"] = bottom_left_wrist_dist
            row["bottom_right_wrist_dist"] = bottom_right_wrist_dist

            # 50-frame cooldown
            if top_left_wrist_dist and top_left_wrist_dist < 70 and (idx - last_hit_frame_top > 50):
                top_hit = True
            if top_right_wrist_dist and top_right_wrist_dist < 70 and (idx - last_hit_frame_top > 50):
                top_hit = True
            if bottom_left_wrist_dist and bottom_left_wrist_dist < 50 and (idx - last_hit_frame_bottom > 50):
                bottom_hit = True
            if bottom_right_wrist_dist and bottom_right_wrist_dist < 50 and (idx - last_hit_frame_bottom > 50):
                bottom_hit = True

        if top_hit or bottom_hit:
            
            # Start new prediction for this hit frame
            new_pred = PendingPrediction(idx)
            pending_predictions.append(new_pred)

            # Immediately set status to "Calculating..."
            row["predicted_ball_direction"] = "Calculating..."



            row["top_player_hit"] = int(top_hit)
            row["bottom_player_hit"] = int(bottom_hit)
            row["ball_hit_frame"] = 1

            rally_hit_count += 1

            if top_hit:    last_hit_frame_top = idx
            if bottom_hit: last_hit_frame_bottom = idx

            # Action recognition on last 10 frames + current
            start_idx_window = max(0, len(rows) - 9)
            window_rows = rows[start_idx_window:] + [row.copy()]
            predicted_action = predict_action_for_window(
                window_rows, selected_features, action_clf, action_scaler
            )
            row["action_recognition"] = predicted_action

            print(f"[DEBUG] Ball positions before prediction: {ball_positions[-15:]}")  # Print last 15 positions

            '''
            predicted_dir = predict_ball_direction(ball_positions, len(ball_positions) - 1, max_frames_ahead=10)
            print(f"[DEBUG] Predicted ball direction at frame {idx}: {predicted_dir}")  # Debug output
            row["predicted_ball_direction"] = predicted_dir
            '''
            # ========== Add prediction result to output ==========
            if idx in prediction_results:
                row["predicted_ball_direction"] = prediction_results[idx]
            else:
                row["predicted_ball_direction"] = None

            # Always trigger ChatGPT API using available frames (up to the last 10)
            frames_to_use = rows[-9:] + [row.copy()] if len(rows) >= 9 else rows + [row.copy()]
            
            commentary = generate_tennis_commentary(frames_to_use)
            row["commentary"] = commentary  # Save ChatGPT response in a new column




        # Finally, append row
        rows.append(row)

    # end of while loop
    cap.release()
   #  print(f"DEBUG: last_rally_frame_idx={last_rally_frame_idx}, len(rows)={len(rows)}")
    if in_rally and last_rally_frame_idx is not None:
        if len(rows) > 0:  # Ensure there is at least one row
            if last_rally_frame_idx < len(rows):  # Ensure valid index
                rows[last_rally_frame_idx]["rally_hits"] = rally_hit_count
            else:
                print(f"WARNING: last_rally_frame_idx ({last_rally_frame_idx}) is out of bounds for rows (length={len(rows)})")
        else:
            print("WARNING: No rows available to update rally_hits.")




    # Convert to DataFrame
    df = pd.DataFrame(rows)
    df = add_hit_detection_with_cooldown(df, bbox_expansion=25, cooldown=10)
    if "commentary" not in df.columns:
        df["commentary"] = None  # Ensure the column exists before saving

    df.to_csv(csv_output_path, index=False)

    print(f"Realtime CSV saved -> {csv_output_path}")

###########################################
# 5) EXAMPLE USAGE WITH TIMER
###########################################
if __name__ == '__main__':

    video_path = r"INPUT VIDDEO MP4"
    output_folder = r"\outputs"
    
    # Extract the base filename without extension
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Construct the CSV output path
    csv_output_path = os.path.join(output_folder, f"{video_filename}.csv")




    # Start timing before execution
    start_time = time.time()

    # Process the video (this function already loops through frames)
    process_single_video_with_ball_realtime(
        video_path=video_path,
        csv_output_path=csv_output_path,
        rally_model=rally_model,
        court_model=court_model,
        yolo_model=yolo_model,
        pose_model=pose_model,
        tracknet_model=tracknet_model,
        action_clf=action_clf,
        action_scaler=action_scaler,
        transform=transform,
        device=device
    )

    # End total timing
    end_time = time.time()
    total_processing_time = end_time - start_time

    print("\n======= Processing Summary =======")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print("=================================")
    print("Done (realtime version)!")

