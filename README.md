# Système de Stéréovision

Ce projet permet de réaliser une reconstruction 3D à partir de deux caméras calibrées.

## Prérequis

- Python 3.x
- OpenCV (cv2)
- NumPy
- Open3D
- Matplotlib
- PyInstaller (pour la compilation)

Installation des dépendances:
```bash
pip install opencv-python numpy open3d matplotlib pyinstaller
```

## Structure du projet

- `main.py` - Point d'entrée du programme
- `camera.py` - Classe pour gérer une caméra individuelle
- `stereo_sys.py` - Classe pour le système stéréo complet
- `config.py` - Paramètres de configuration
- `build.bat`/`build.sh` - Scripts de compilation

## Utilisation

1. Calibration des caméras:
   - Préparez un damier d'échiquier (7x9 par défaut)
   - Lancez le programme et suivez les instructions
   - Prenez plusieurs photos du damier sous différents angles

2. Reconstruction 3D:
   - Placez un objet fixes devant les caméras
   - Prenez une photo stéréo
   - Le programme générera un nuage de points 3D

## Fonctionnalités

- Calibration automatique des caméras
- Détection de points d'intérêt SIFT
- Rectification stéréo
- Calcul de carte de disparité
- Génération de nuage de points 3D
- Visualisation 3D interactive

## Compilation

Windows:
```bash
.\build.bat
```

Linux/Mac:
```bash
chmod +x build.sh
./build.sh
```

L'exécutable sera généré dans le dossier `dist/`.
