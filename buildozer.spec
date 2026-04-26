[app]
title = Twin Face Match
package.name = twinfacematch
package.domain = org.test

source.dir = .
version = 0.1

requirements = python3,kivy,pillow,numpy,opencv-python,plyer,android

orientation = portrait
fullscreen = 0

android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,CAMERA

android.api = 30
android.minapi = 21
# REMOVE android.ndk line - let buildozer auto-detect
# REMOVE android.sdk line
# REMOVE android.ndk_api line

android.archs = arm64-v8a

p4a.bootstrap = sdl2
android.allow_backup = True