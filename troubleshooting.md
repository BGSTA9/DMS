❌ FAIL  opencv-python  →  No module named 'cv2'
❌ FAIL  mediapipe  →  No module named 'mediapipe'
❌ FAIL  numpy  →  No module named 'numpy'
❌ FAIL  torch  →  No module named 'torch'
❌ FAIL  torchvision  →  No module named 'torchvision'
❌ FAIL  ultralytics  →  No module named 'ultralytics'
❌ FAIL  pygame  →  No module named 'pygame'
❌ FAIL  scipy  →  No module named 'scipy'
❌ FAIL  filterpy  →  No module named 'filterpy'
❌ FAIL  Pillow  →  No module named 'PIL'
❌ FAIL  timm  →  No module named 'timm'
❌ FAIL  deepface  →  No module named 'deepface'
❌ FAIL  imutils  →  No module named 'imutils'
❌ FAIL  Webcam check failed: No module named 'cv2'

Troubleshooting 

~/Downloads/DMS $ ./.venv/bin/python verify_setup.py
(base) soheilsanati@Soheils-MacBook-Pro DMS % py
(base) soheilsanati@Soheils-MacBook-Pro DMS % python -m pip install mediapipe 2>&1 | t
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda info --envs 2>/dev/null && ech
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda create -n dms python=3.12 -y
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda run -n dms pip install -r requ
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda run -n dms python -c "import s
(base) soheilsanati@Soheils-MacBook-Pro DMS % 
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda run -n dms python -m pip insta
(base) soheilsanati@Soheils-MacBook-Pro DMS % 
(base) soheilsanati@Soheils-MacBook-Pro DMS % source .venv/bin/activate && python
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % source .venv/bin/activate &&
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % 
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % git log -n 3 --oneline && git diff --name
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % git status
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % git log -n 5 --oneline
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % ls -la dms/bin/activate
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % 
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % rm -rf dms
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % 
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % ./.venv/bin/python verify_setup.py
============================================================
  DMS Environment Verification
============================================================
  ✅ PASS  Python 3.12.2  (3.10+ recommended)
  ✅ PASS  opencv-python
  ✅ PASS  mediapipe
  ✅ PASS  numpy
  ✅ PASS  torch
  ✅ PASS  torchvision
  ✅ PASS  ultralytics
pygame 2.5.2 (SDL 2.28.3, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html
  ✅ PASS  pygame
  ✅ PASS  scipy
  ✅ PASS  filterpy
  ✅ PASS  Pillow
  ✅ PASS  timm
  ✅ PASS  deepface
  ✅ PASS  imutils

  ⚠️  WARN  No CUDA GPU — will run on CPU (slower but functional)
  ✅ PASS  Webcam (index 0) accessible
  ✅ PASS  config.py loaded successfully
  ✅ PASS  Folder: core/
  ✅ PASS  Folder: dms_engine/
  ✅ PASS  Folder: ui/
  ✅ PASS  Folder: simulation/
  ✅ PASS  Folder: assets/
  ✅ PASS  Folder: models/
  ✅ PASS  Folder: docs/

============================================================
  Verification complete. Fix any ❌ before proceeding.
============================================================

(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS %
(base) soheilsanati@Soheils-MacBook-Pro DMS % conda deactivate

(base) soheilsanati@Soheils-MacBook-Pro DMS % source .venv/bin/activate 
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % python verify_setup.py

============================================================
  DMS Environment Verification
============================================================
  ✅ PASS  Python 3.12.2  (3.10+ recommended)
  ✅ PASS  opencv-python
  ✅ PASS  mediapipe
  ✅ PASS  numpy
  ✅ PASS  torch
  ✅ PASS  torchvision
  ✅ PASS  ultralytics
pygame 2.5.2 (SDL 2.28.3, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html
  ✅ PASS  pygame
  ✅ PASS  scipy
  ✅ PASS  filterpy
  ✅ PASS  Pillow
  ✅ PASS  timm
  ✅ PASS  deepface
  ✅ PASS  imutils

  ⚠️  WARN  No CUDA GPU — will run on CPU (slower but functional)
  ✅ PASS  Webcam (index 0) accessible
  ✅ PASS  config.py loaded successfully
  ✅ PASS  Folder: core/
  ✅ PASS  Folder: dms_engine/
  ✅ PASS  Folder: ui/
  ✅ PASS  Folder: simulation/
  ✅ PASS  Folder: assets/
  ✅ PASS  Folder: models/
  ✅ PASS  Folder: docs/

============================================================
  Verification complete. Fix any ❌ before proceeding.
============================================================

-----------------------------------------------------------------------------------------------------------------------------------------------

(base) soheilsanati@Soheils-MacBook-Pro DMS % python test_geometry.py
Traceback (most recent call last):
  File "/Users/soheilsanati/Downloads/DMS/test_geometry.py", line 7, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
(base) soheilsanati@Soheils-MacBook-Pro DMS % source .venv/bin/activate 
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % python test_geometry.py  
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1772287742.592886 5757565 gl_context.cc:357] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M2
17:09:02 | INFO     | dms_engine.geometry_tracker | GeometryTracker initialized (640×480)

[test_geometry] Running — press Q to quit

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772287742.612286 5757934 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772287742.618058 5757928 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/soheilsanati/Downloads/DMS/.venv/lib/python3.12/site-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.
  warnings.warn('SymbolDatabase.GetPrototype() is deprecated. Please '
Traceback (most recent call last):
  File "/Users/soheilsanati/Downloads/DMS/test_geometry.py", line 129, in <module>
    main()
  File "/Users/soheilsanati/Downloads/DMS/test_geometry.py", line 114, in main
    draw_hud(frame, state)
  File "/Users/soheilsanati/Downloads/DMS/test_geometry.py", line 41, in draw_hud
    put(f"Face Detected: YES")
TypeError: draw_hud.<locals>.put() missing 1 required positional argument: 'row'
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % 

Solution:

@terminal:zsh please rectify, and provide me with the steps on how you preformed your rectification 
The error is clear: in test_geometry.py, the put() function inside draw_hud() requires 2 arguments (text and row), but the first call put(f"Face Detected: YES") only passes 1. I need to fix the call to include the row number 0.

The bug is clear. The 

put()
 function inside 

draw_hud
 requires two arguments — text and row — but the first call 

put(f"Face Detected: YES")
 is missing the row argument.

 source .venv/bin/activate && python test_geometry.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1772287841.571352 5759212 gl_context.cc:357] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M2
17:10:41 | INFO     | dms_engine.geometry_tracker | GeometryTracker initialized (640×480)

[test_geometry] Running — press Q to quit

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772287841.587585 5759432 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772287841.593055 5759432 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/soheilsanati/Downloads/DMS/.venv/lib/python3.12/site-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.
  warnings.warn('SymbolDatabase.GetPrototype() is deprecated. Please '

Checked command status
The tracker is running with no crash — the [test_geometry] Running — press Q to quit message is live and MediaPipe has initialized successfully. The window should be open on your screen right now.

What the bug was & how I fixed it
The error:

TypeError: draw_hud.<locals>.put() missing 1 required positional argument: 'row'

Root cause: Inside 

draw_hud()
, a helper 

put(text, row, color=...)
 was defined. It needs two positional arguments — the text string and the row number (which controls the vertical position). The very first call after detecting a face was written as:

 put(f"Face Detected: YES")     # ❌ missing row
All the other 10+ calls below it correctly passed a row number (1, 2, 3...), but this first one did not.

The fix — one character change:
put(f"Face Detected: YES", 0)  # ✅ row=0 = first line
Row 0 places it at the top of the HUD panel, which is exactly where it belongs logically (first thing displayed). The fix was minimal and surgical — no other code needed to change.

-----------------------------------------------------------------------------------------
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % python test_analytics.py

[test_analytics] Initializing DMS Core …

20:45:52 | INFO     | dms_engine.dms_core | Initializing DMSCore …
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1772300752.763483 5976265 gl_context.cc:357] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M2
20:45:52 | INFO     | dms_engine.geometry_tracker | GeometryTracker initialized (640×480)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772300752.766158 5976620 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772300752.777433 5976622 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
20:45:52 | INFO     | dms_engine.detection_module | Loading YOLOv8n weights …
20:45:52 | INFO     | dms_engine.detection_module | DetectionModule using device: mps
20:45:53 | INFO     | dms_engine.detection_module | DetectionModule ready.
20:45:53 | WARNING  | dms_engine.action_module | ActionModule: No fine-tuned weights found at /Users/soheilsanati/Downloads/DMS/models/action_efficientnet.pth. Running in HEURISTIC mode (uses YOLO detections as proxy). For full accuracy, train on StateFarm dataset and place weights at the path above.
20:45:53 | INFO     | dms_engine.action_module | ActionModule ready (device=mps, heuristic=True)
20:45:53 | INFO     | dms_engine.fer_module | Initializing FERModule (DeepFace) …
20:45:54 | INFO     | dms_engine.fer_module | FERModule ready.
20:45:54 | INFO     | dms_engine.dl_pipeline | DLPipeline initialized (target=15fps)
20:45:54 | INFO     | dms_engine.analytics | AnalyticsEngine initialized.
20:45:54 | INFO     | dms_engine.state_machine | DriverStateMachine initialized.
20:45:54 | INFO     | dms_engine.dms_core | DMSCore initialized.
20:45:54 | INFO     | dms_engine.dl_pipeline | DLPipeline background thread started.
20:45:54 | INFO     | dms_engine.dms_core | DMSCore started.
[test_analytics] Running — press Q to quit
  → Slowly close your eyes for 3s to trigger DROWSY state
  → Cover the camera to test obstruction detection
  → Hold your phone up to test action detection

Traceback (most recent call last):
  File "/Users/soheilsanati/Downloads/DMS/test_analytics.py", line 168, in <module>
    main()
  File "/Users/soheilsanati/Downloads/DMS/test_analytics.py", line 153, in main
    draw_analytics_hud(frame, state)
  File "/Users/soheilsanati/Downloads/DMS/test_analytics.py", line 77, in draw_analytics_hud
    badge(state.driver_state,    15,       h - 65, d_col)
  File "/Users/soheilsanati/Downloads/DMS/test_analytics.py", line 70, in badge
    cv2.rectangle(frame, (bx - 5, by - 22), (bx + tw + 5, by + 5), color, -1)
                                             ~~~^~~~
TypeError: unsupported operand type(s) for +: 'int' and 'tuple'
(.venv) (base) soheilsanati@Soheils-MacBook-Pro DMS % 


Solution: 

@terminal:zsh Error: "TypeError: unsupported operand type(s) for +: 'int' and 'tuple'"

please rectify!

thought for 6s

The error is clear. In the badge() function, cv2.getTextSize() returns ((width, height), baseline), so tw is actually the full tuple (width, height) from the first element, not just the width integer. The code does cv2.getTextSize(text, ...)[0:2] which gives ((width, height), baseline), then unpacks as tw, _ where tw = (width, height) (a tuple). Then bx + tw + 5 fails because you can't add int + tuple.

The fix is to unpack properly: (tw, th), _ = cv2.getTextSize(...) or access tw as the width only.

The bug is in the 

badge()
 function. cv2.getTextSize() returns 

((width, height), baseline)
, but the code unpacks it as tw, _ — making tw a tuple 

(width, height)
 instead of just the integer width. Then bx + tw + 5 crashes because you can't add an int to a tuple.

 Edited test_analytics +1 -1

 Now re-running it:

~/Downloads/DMS $ source .venv/bin/activate && python test_analytics.py

<truncated 104 lines>
20:51:15 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.27)
20:51:16 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.71)
20:51:17 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.28)
20:51:20 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.69)
20:51:31 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.23)
20:51:31 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.60)
20:51:50 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.20)
20:51:58 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.65)
20:51:59 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.51, perclos=0.23)
20:51:59 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.35, perclos=0.06)
20:52:05 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.00)
20:52:06 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.48, perclos=0.36)
20:52:07 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.65, perclos=0.54)
20:52:17 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.38, perclos=0.23)
20:52:18 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.22, perclos=0.08)
20:52:20 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.44, perclos=0.36)
20:52:21 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.63, perclos=0.56)
20:52:23 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.52)
20:52:25 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.36)
20:52:32 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.53, perclos=0.33)
20:52:33 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.70, perclos=0.51)
20:52:34 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.60)
20:52:40 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.41)
20:52:43 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.34, perclos=0.30)
20:52:44 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.24, perclos=0.12)
20:52:49 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.60)
20:52:53 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.27, perclos=0.23)
20:52:55 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.36, perclos=0.17)
20:52:56 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.46, perclos=0.29)
20:52:56 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.31)
20:52:59 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.79, perclos=0.52)
20:53:05 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.54)
20:53:07 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.25)
20:53:15 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.51)
20:53:17 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → ALERT (score=0.00, perclos=0.00)
20:53:17 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.00)
20:53:17 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.48, perclos=0.32)
20:53:17 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.84)
20:53:18 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.17, perclos=0.12)
20:53:19 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.22)
20:53:19 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.34, perclos=0.20)
20:53:20 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.52, perclos=0.46)
20:53:21 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.50, perclos=0.34)
20:53:24 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.31, perclos=0.14)
20:53:26 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.53, perclos=0.33)
20:53:26 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.60, perclos=0.46)
20:53:28 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.52, perclos=0.33)
20:53:29 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.27, perclos=0.11)
20:53:40 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.48)
20:53:43 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.25)
20:53:43 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.44, perclos=0.33)
20:53:44 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.77, perclos=0.54)
20:53:45 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.45)
20:53:48 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.33)
20:53:50 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.34, perclos=0.24)
20:53:51 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.20, perclos=0.10)
20:53:52 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.45)
20:53:54 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.36, perclos=0.30)
20:53:54 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.29)
20:53:54 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.72, perclos=0.56)
20:54:04 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.52, perclos=0.23)
20:54:04 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.28, perclos=0.02)
20:54:06 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.47, perclos=0.23)
20:54:09 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.21, perclos=0.03)
20:54:13 | INFO     | dms_engine.state_machine | Driver state: ALERT → DROWSY (score=0.50, perclos=0.23)
20:54:14 | INFO     | dms_engine.state_machine | Driver state: DROWSY → SLEEPING (score=0.56, perclos=0.51)
20:54:38 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.46)
20:54:41 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.44)
20:54:48 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.48)
20:54:51 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.45)
20:55:04 | INFO     | dms_engine.state_machine | Attention state: FOCUSED → DISTRACTED (score=0.66)
20:55:12 | INFO     | dms_engine.state_machine | Attention state: DISTRACTED → FOCUSED (score=0.07)
20:55:25 | INFO     | dms_engine.state_machine | Driver state: SLEEPING → DROWSY (score=0.50, perclos=0.37)
20:55:27 | INFO     | dms_engine.state_machine | Driver state: DROWSY → ALERT (score=0.32, perclos=0.07)
[test_analytics] Shutting down …
20:55:28 | INFO     | dms_engine.dl_pipeline | DLPipeline background thread stopped.
20:55:28 | INFO     | dms_engine.geometry_tracker | GeometryTracker released.
20:55:28 | INFO     | dms_engine.dms_core | DMSCore stopped.
[test_analytics] Done.

Checked command status
It's running perfectly! The live output confirms the full DMS pipeline is working:

[test_analytics] Running — press Q to quit
Driver state: ALERT → DROWSY (score=0.38, perclos=0.21)
Attention state: FOCUSED → DISTRACTED (score=0.60)
Attention state: DISTRACTED → FOCUSED (score=0.35)
Driver state: DROWSY → ALERT (score=0.19, perclos=0.03)

The state machine is detecting real transitions in real time from your camera!

What the bug was
cv2.getTextSize() has this return signature:
((width, height), baseline) = cv2.getTextSize(text, font, scale, thickness)

The original code sliced [0:2] and unpacked into tw, _ — this gave tw the value 

(width, height)
 (a tuple), not a plain integer. Then the line bx + tw + 5 failed because you can't do int + tuple.

The fix — one character change in the unpacking:

# Before (broken): tw = (width, height) tuple
tw, _ = cv2.getTextSize(...)[0:2]

# After (correct): tw = width integer
(tw, _), _ = cv2.getTextSize(...)
