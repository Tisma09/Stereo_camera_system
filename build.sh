#!/bin/bash
# Script pour exécuter PyInstaller avec les options spécifiées

echo "Exécution de PyInstaller..."

pyinstaller --onefile \
  --add-data "config.py:." \
  --add-data "camera.py:." \
  --add-data "stereo_sys.py:." \
  main.py

echo "PyInstaller terminé !"

chmod +x dist/main

