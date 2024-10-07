import sys
from pathlib import Path
import pytest

# Agregar la carpeta src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Ejecutar todas las pruebas de los módulos de prueba
if __name__ == "__main__":
    # Ejecutar pytest y buscar automáticamente todos los archivos de prueba
    pytest.main(["-q", "--tb=short", "tests/"])
