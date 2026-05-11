[app]
title = Twin Face Match
package.name = twinfacematch
package.domain = org.twinfacematch

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,pth,onnx,xml

version = 2.0

# ─────────────────────────────────────────────────────
# Requirements
# Note: PyTorch is too large for mobile — inference uses
# ONNX Runtime instead (exported from trained checkpoint).
# ─────────────────────────────────────────────────────
requirements = python3,kivy==2.3.0,pillow,numpy,opencv,onnxruntime,plyer

orientation = portrait
fullscreen = 0

# Include the ONNX model and Haar cascade
source.include_patterns = checkpoints/model.onnx,utils/*.xml

android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,CAMERA
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.allow_backup = False

# Release signing (fill in before Play Store submission)
# android.keystore = my-release-key.jks
# android.keystore_passwd = <password>
# android.keyalias = my-key-alias
# android.keyalias_passwd = <password>

p4a.bootstrap = sdl2

[buildozer]
log_level = 2
warn_on_root = 1
