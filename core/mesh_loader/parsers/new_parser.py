import io
from typing import Any, BinaryIO, Iterable

import numpy as np

from core.binary_readers import (
    read_float,
    read_half_float,
    read_uint8,
    read_uint16,
    read_uint32,
)
from core.logger import get_logger
from core.mesh_loader.types import BaseMeshParser, MeshData


def identify_mesh_type(
    size: int,
    vertex_count: int,
    face_count: int,
    meshes_inside_data: int,
    uv_total_data: int,
    bone_type: int,
    version: int,
) -> int:
    # Smallest final size calculation for vertex bones and vertex weights, 4x8u + 4xfloat (20 bytes)
    test = 0
    uv_total_data = uv_total_data * 8

    size = size - meshes_inside_data

    # Assume bone_type 1 or 4 has 20 bytes per vertex
    # vertex_bones 4*uint8 and vertex weights 4xfloat
    if bone_type == 1 or bone_type == 4:
        size = size - 20 * vertex_count

    match bone_type:
        case 0 | 1:
            test = size - vertex_count * 24 - face_count * 6 - uv_total_data - 2
            if not test or test == 32:
                return 1

            test = test - vertex_count * 12
            if not test or test == 32:
                return 1

            # vertex_bones is uint16
            test = test - vertex_count * 4
            if test > 0 and test < vertex_count - 1:
                return 2

            # bones has extra data + vertex is uint16
            test = test - vertex_count * 4
            if test > 0 and test < vertex_count - 1:
                return 3

            if version == 4:
                # No bones
                test = size - vertex_count * 14 - face_count * 6 - (uv_total_data // 2)
                if test >= 0 and test < vertex_count - 1:
                    return -1
                # Vertex is +1 because vertex_weight can be half_float
                test = (
                    size - vertex_count * 3 - face_count * 6 - (uv_total_data // 2) - 1
                )
                if not test or test == 32:
                    return -1
                # UV data: U, V, 4 bytes each, (8 bytes total, for vertex_count and extra)
                test = size - vertex_count * 11 - face_count * 6 - uv_total_data - 1
                if not test:
                    return -1

                # vertex positions (4*uint16?), normals (3*half_float) 14 bytes per vertex, vertex_weights (4xhalf_float), uv (2*half_float)
                test = (
                    size - vertex_count * 6 - face_count * 6 - uv_total_data // 2 - 32
                )
                if test >= 0 and test < vertex_count - 1:
                    return -1

                test = size - vertex_count * 14 - face_count * 6 - uv_total_data
                # vertex positions are 8 bytes whole (the 3 coordinates, somehow), and normals are quantized ints (2 bytes each), 14 bytes per vertex, and 300 bytes of some bullshit data in the middle
                if test >= 0 and test < vertex_count - 1:
                    return -1

        case 3 | 4:
            test = size - vertex_count * 12 - face_count * 6 - uv_total_data - 2
            if not test or test == 32:
                return 4

            # vertex_bones is uint16
            test = test - vertex_count * 4
            if not test or test == 32:
                return 5

            # _flag is 1
            test = test - vertex_count * 2
            if not test or test == 32:
                return 4

            # vertex_bones is uint16 AND _flag is 1
            test = test - vertex_count * 4
            if not test or test == 32:
                return 5

            return 100

        case _:
            return -98
    return -99


# Try to identify if the following data is uint8 or uint16, if its 16 bits, return True, else False
def bones_is_16(f: BinaryIO, bones: int) -> bool:
    test = f.read(bones * 2)
    f.seek(-bones * 2, 1)
    return b"\x00\x00\x00\x00" not in test


def dequant(value: int, min_float: float, max_float: float) -> float:
    normalized = (value + 32768) / 65535.0
    return min_float + (normalized * (max_float - min_float))


class MeshParser0(BaseMeshParser):
    """Standard mesh parser for typical mesh formats with bone hierarchies."""

    def parse(self, data: bytes) -> MeshData:
        """Parse mesh."""

        f = io.BytesIO(data)

        raw_model = self._parse_mesh_testing(f)
        return self._standardize_mesh_data(raw_model)

    def _parse_mesh_testing(self, f: BinaryIO) -> dict[str, Iterable[Any] | int]:
        model: dict[str, Iterable[Any] | int] = {}
        model["mesh"] = {}
        model["mesh"]["extra"] = []
        model["mesh"]["multiple_offsets"] = []
        parent_nodes: list[int] = []
        MAX_PARENT_NODE = 255

        bone_int_reader = read_uint8
        vertex_float_reader = read_float
        vertex_int_reader = read_uint8
        total_uv_data = 0
        meshes_inside = 1
        meshes_inside_data = 0
        total_uv_data = 0

        _magic_number = read_uint32(f)
        model["mesh_version"] = read_uint16(f)
        read_uint16(f)  # always_0x0500
        model["has_bones"] = read_uint16(f)
        read_uint16(f)  # always_0x0000
        get_logger().info(
            f"MESH: Version {model['mesh_version']} | Bone Type: {model['has_bones']}"
        )

        if model["has_bones"] == 1 or model["has_bones"] == 4:
            model["bone_count"] = bone_count = read_uint16(f)
            if model["mesh_version"] == 4 and bones_is_16(f, bone_count):
                bone_int_reader = read_uint16
                MAX_PARENT_NODE = 65535
            self._validate_bone_count(bone_count)

            for _ in range(bone_count):
                parent_node: int = bone_int_reader(f)
                if parent_node == MAX_PARENT_NODE:
                    parent_node = -1
                parent_nodes.append(parent_node)
            model["bone_parent"] = parent_nodes

            bone_names: list[str] = []
            for _ in range(bone_count):
                bone_name = f.read(32)
                bone_name = bone_name.decode().replace("\0", "").replace(" ", "_")
                bone_names.append(bone_name)
            model["bone_name"] = bone_names

            bone_extra_info = read_uint8(f)
            if bone_extra_info:
                f.seek(28 * bone_count, 1)

            model["bone_matrix"] = []
            for _ in range(bone_count):
                matrix = [read_float(f) for _ in range(16)]
                matrix = np.array(matrix).reshape(4, 4)
                model["bone_matrix"].append(matrix)

            # creates a new "dummy_root" as a default bone for the rest of the starters to link to
            if parent_nodes.count(-1) > 1:
                num = len(model["bone_parent"])
                model["bone_parent"] = list(
                    map(lambda x: num if x == -1 else x, model["bone_parent"])
                )
                model["bone_parent"].append(-1)
                model["bone_name"].append("dummy_root")
                model["bone_matrix"].append(np.identity(4))

            flag1 = read_uint8(f)
            if flag1 != 0:
                raise ValueError(
                    f"Unexpected _flag value {flag1} at position {hex(f.tell() - 1)}"
                )

        ending_address = read_uint32(f)
        f.seek(ending_address, 0)
        model["mesh"]["multiple"] = meshes_inside = read_uint16(f)
        for i in range(meshes_inside):
            model["mesh"]["multiple_offsets"].append(read_uint32(f))

        main_data_offset = model["mesh"]["multiple_offsets"].pop(0)

        # Temporary handling, calculate the size and just ignore the data
        if meshes_inside > 1:
            for i in model["mesh"]["multiple_offsets"]:
                f.seek(i, 0)
                small = f.tell()
                v = read_uint32(f)
                f.seek(v * 12, 1)
                v = read_uint32(f)
                f.seek(v * 2, 1)
                v = read_uint32(f)
                f.seek(v * 4, 1)
                meshes_inside_data += f.tell() - small + 1

        f.seek(main_data_offset)

        it = 0  # iterations
        while True:
            vertex_count = read_uint32(f)
            face_count = read_uint32(f)
            uv_layers = read_uint8(f)
            unknown = read_uint8(f)
            must_be_one = read_uint16(f)
            model["mesh"]["extra"].append(
                (vertex_count, face_count, uv_layers, unknown)
            )
            if must_be_one == 1:
                vertex_count = read_uint32(f)
                face_count = read_uint32(f)
                break
            else:
                it += 1
                f.seek(-2, 1)

        for vertex, _, uv, _ in model["mesh"]["extra"]:
            total_uv_data += vertex * uv
            # if add_again and uv == 1:
            #    total_uv_data += vertex

        # Temporary handling, remove if its problematic:
        total_uv_data = (
            vertex_count
            if vertex_count * 2 == total_uv_data and uv_layers == 1
            else total_uv_data
        )

        mesh_data_size = ending_address - f.tell()

        type = identify_mesh_type(
            mesh_data_size,
            vertex_count,
            face_count,
            meshes_inside_data,
            total_uv_data,
            model["has_bones"],
            model["mesh_version"],
        )
        get_logger().info(
            f"MESH: VERTS: {vertex_count} | FACES: {face_count} | UV_DATA: {total_uv_data} | UV_LAYERS: {'No' if uv_layers == 0 else uv_layers} | TYPE: {type}"
        )

        if meshes_inside_data != 0:
            get_logger().warning(
                f"MESH: This is a multiple mesh type ({meshes_inside} total) - COMPATIBILITY NOT GURANTEED"
            )

        if type == 100:
            vertex_float_reader = read_half_float
            get_logger().warning(
                "MESH: This mesh has a non-standard UV count - UV , vertex_bone and vertex_weights will be missing!"
            )
        elif type == -1:
            raise NotImplementedError("MESH: This mesh type is not yet implemented")

        elif type == 4 or type == 5:
            vertex_float_reader = read_half_float
        elif type == 5 or type == 2 or type == 3:
            vertex_int_reader = read_uint16

        model["mesh"]["data"] = []
        model["mesh"]["data"].append(
            (
                vertex_count,
                face_count,
                uv_layers,
            )
        )

        model["mesh"]["position"] = []
        # vertex position
        for _ in range(vertex_count):
            x = vertex_float_reader(f)
            y = vertex_float_reader(f)
            z = vertex_float_reader(f)
            model["mesh"]["position"].append((x, y, z))

        model["mesh"]["normal"] = []

        # vertex normal
        for _ in range(vertex_count):
            x = vertex_float_reader(f)
            y = vertex_float_reader(f)
            z = vertex_float_reader(f)
            model["mesh"]["normal"].append((x, y, z))

        _flag = read_uint16(f)

        if _flag == 1 and (type == 4 or type == 5 or type == 7 or type == 100):
            f.seek(vertex_count * 6, 1)
        elif _flag == 1:
            f.seek(vertex_count * 12, 1)
        elif _flag > 1:
            f.seek(_flag * 4, 1)

        model["mesh"]["face"] = []
        # face index table
        for _ in range(face_count):
            v1 = read_uint16(f)
            v2 = read_uint16(f)
            v3 = read_uint16(f)
            model["mesh"]["face"].append((v1, v2, v3))

        model["mesh"]["uv"] = []
        if uv_layers and type != 100:
            for _ in range(vertex_count):
                model["mesh"]["uv"].append((read_float(f), read_float(f)))
        else:
            for _ in range(total_uv_data):
                model["mesh"]["uv"].append((0.0, 0.0))

        if model["has_bones"] == 1 or model["has_bones"] == 4 and type != 100:
            model["vertex_bone"] = []
            for _ in range(vertex_count):
                model["vertex_bone"].append([vertex_int_reader(f) for _ in range(4)])

            model["vertex_weight"] = []
            for _ in range(vertex_count):
                model["vertex_weight"].append([read_float(f) for _ in range(4)])

        return model
