# from https://github.com/davidstutz/mesh-voxelization
# modified by Joerg Stueckler 2019 MPI for Intelligent Systems

import os
import math
import numpy as np
import trimesh

def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices = [[]], faces = [[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype = float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype = int)
        """ (numpy.ndarray) Faces. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0]*3
        max = [0]*3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]

    @staticmethod
    def from_off(filepath):
        """
        Read a mesh from OFF.

        :param filepath: path to OFF file
        :type filepath: str
        :return: mesh
        :rtype: Mesh
        """

        vertices, faces = read_off(filepath)

        real_faces = []
        for face in faces:
            assert len(face) == 4
            real_faces.append([face[1], face[2], face[3]])

        return Mesh(vertices, real_faces)

    def to_off(self, filepath):
        """
        Write mesh to OFF.

        :param filepath: path to write file to
        :type filepath: str
        """

        faces = np.ones((self.faces.shape[0], 4), dtype = int)*3
        faces[:, 1:4] = self.faces[:, :]

        write_off(filepath, self.vertices.tolist(), faces.tolist())


def scale_off(input, output, padding=0.1, height=2, width=2, depth=2):

    if not os.path.exists(input):
        print('Input directory does not exist.')
        return

    if not os.path.exists(output):
        os.makedirs(output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    scale = np.max([height, width, depth])   # scale=2. (default), basic cube for sdfs later is in range [-1, 1] (?)

    filelist = []

    for filename in os.listdir(input):
        filepath = os.path.join(input, filename)
        filelist.append(filename)
        print(os.path.basename(filepath))

        out_path = os.path.join(output, filename)
        if os.path.exists(out_path):
            continue
        if not '.off' in filename:
            continue

        print(filepath)

        # check if mesh is watertight
        # mesh = trimesh.load(filepath)
        # mesh.show()
        # if not mesh.is_watertight:
        #    print("mesh not watertight")
        #    continue

        mesh = Mesh.from_off(filepath)

        # Get extents of model.
        min, max = mesh.extents()

        print(' extents before %f - %f (%f), %f - %f (%f), %f - %f (%f)' %
              (min[0], max[0], max[0]-min[0], min[1], max[1], max[1]-min[1],  min[2], max[2], max[2]-min[2]))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2
        )

        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        mesh.translate(translation)

        # Updated extents of model
        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Scales all dimensions equally.
        sizes = (
            total_max - total_min,
            total_max - total_min,
            total_max - total_min
        )

        # scales = (
        #     1 / (sizes[0] + 2 * padding * sizes[0]),
        #     1 / (sizes[1] + 2 * padding * sizes[1]),
        #     1 / (sizes[2] + 2 * padding * sizes[2])
        # )
        pad = 2.*padding/(1-2*padding)
        scales = (
            scale / (sizes[0] + pad * sizes[0]),
            scale / (sizes[1] + pad * sizes[1]),
            scale / (sizes[2] + pad * sizes[2])
        )
        mesh.scale(scales)

        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))
        print(' final extents %f - %f (%f), %f - %f (%f), %f - %f (%f); total %f - %f (%f)' %
              (min[0], max[0], max[0]-min[0], min[1], max[1], max[1]-min[1],  min[2], max[2], max[2]-min[2], total_min, total_max, total_max-total_min))

        mesh.to_off(out_path)

    return filelist


def scale_off_output(filepath, output, s, grid_size=64):

    if not os.path.exists(output):
        os.makedirs(output)
        print('Created output directory.')

    filename = os.path.basename(filepath)
    out_path = os.path.join(output, filename)
    # if os.path.exists(out_path):
    #     return

    print(filepath)

    mesh = Mesh.from_off(filepath)

    def print_extensions(message):

        min, max = mesh.extents()
        print(message + ' %f - %f (%f), %f - %f (%f), %f - %f (%f)' %
              (min[0], max[0], max[0] - min[0], min[1], max[1], max[1] - min[1], min[2], max[2], max[2] - min[2]))

    # print_extensions('extents before')

    translation = (-grid_size/2, -grid_size/2, -grid_size/2)
    mesh.translate(translation)
    # print_extensions('after translation')

    scales = (2. / grid_size, 2. / grid_size, 2. / grid_size)
    mesh.scale(scales)
    # print_extensions('after scaling')

    min, _ = mesh.extents()
    z_min = min[2]
    translation = (0, 0, -z_min)
    mesh.translate(translation)
    # print_extensions('after translation no2')

    scales = (s, s, s)
    mesh.scale(scales)

    min, max = mesh.extents()
    total_min = np.min(np.array(min))
    total_max = np.max(np.array(max))
    # print(' final extents %f - %f (%f), %f - %f (%f), %f - %f (%f); total %f - %f (%f)' %
    #       (min[0], max[0], max[0] - min[0], min[1], max[1], max[1] - min[1], min[2], max[2], max[2] - min[2],
    #        total_min, total_max, total_max - total_min))

    mesh.to_off(out_path)
