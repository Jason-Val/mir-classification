import numpy as np

"""
writes vertices as a .ply file; useful for making scatterplots
"""
def write_ply(vertices, name):
    with open('./visualization/pca/ply/{}.ply'.format(name), 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(vertices)))
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for vertex in vertices:
            f.write('{} {} {}\n'.format(vertex[2], vertex[1], vertex[0]))        #pca1->z, pca2->y, pca3->x
