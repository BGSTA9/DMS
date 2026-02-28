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