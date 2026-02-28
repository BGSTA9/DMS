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