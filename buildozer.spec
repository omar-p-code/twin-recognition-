[app]
title = Twin Face Match
package.name = twinfacematch
package.domain = org.test

source.dir = .
version = 0.1


requirements = python3,kivy==2.3.0,pillow,numpy,opencv-python-headless,plyer,android

orientation = portrait
fullscreen = 0

android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,CAMERA

android.api = 33
android.minapi = 24
android.sdk = 34.0.0
android.ndk = 25.2.9519653
android.ndk_api = 21

android.arch = arm64-v8a

p4a.bootstrap = sdl2
android.gradle_dependencies = 'androidx.core:core:1.12.0'
android.allow_backup = True
android.bootstrap_timeout = 3600