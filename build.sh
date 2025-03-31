#!/bin/bash
# Script pour exécuter PyInstaller avec les options spécifiées

echo "Exécution de PyInstaller..."

pyinstaller --onefile \
  --add-data "stereo_sys.py:." \
  --add-data "camera.py:." \
  main.py

echo "PyInstaller terminé !"

chmod +x dist/main

