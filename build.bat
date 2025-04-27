@echo off
REM Script pour exécuter PyInstaller avec les options spécifiées

echo Exécution de PyInstaller...

pyinstaller --onefile ^
  --add-data "config.py;." ^
  --add-data "camera.py;." ^
  --add-data "stereo_sys.py;." ^
  main.py

echo PyInstaller terminé !

REM Le fichier exécutable se trouve dans le répertoire dist
