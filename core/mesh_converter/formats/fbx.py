"""FBX Mesh Format Converter"""

from core.mesh_loader import MeshData

NAME = "Kaydara (FBX) Format"
EXTENSION = ".fbx"


def convert(mesh_data: MeshData):
    """
    Converts MeshData to FBX 7.4 ASCII string.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.

    Returns:
    - bytes: FBX file content as bytes
    """
    return bytes([0])
