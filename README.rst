Cython-CSG
==============


Cython port of Evan Wallace's Javascript https://github.com/evanw/csg.js/ and Tim Knip raw Python version https://github.com/timknip/pycsg.


What is Cython-CSG
==================

CSG stands for Constructive Solid Geometry. It allow Boolean operations to be made on a 3D object like subtraction.


Usage
=====

Subtraction
-----------

A simple subtraction is as followed

.. code-block:: python

   from _cython_csg import CSG

   a = CSG.cube()
   b = CSG.cube([0.5, 0.5, 0.0])
   c = a - b
   c.saveVTK('subtract.vtk')


Union
-----

A Simple Union would is as followed


.. code-block:: python

    from _cython_csg import CSG
    a = CSG.sphere(center=[0.5, 0.5, 0.5], radius=0.5, slices=8, stacks=4)
    b = CSG.cylinder(start=[0.,0.,0.], end=[1.,0.,0.], radius=0.3, slices=16)
    a.union(b).saveVTK('union.vtk')



Custom objects
--------------


You can make custom object here is a example

.. code-block:: python

    from _cython_csg import BSPNode, Polygon, Vertex
    v0 = Vertex([0., 0., 0.])
    v1 = Vertex([1., 0., 0.])
    v2 = Vertex([1., 1., 0.])
    p0 = Polygon([v0, v1, v2])
    polygons = [p0]
    node = BSPNode(polygons)


if you then want to convert it back to a CSG object you can do

    CSG.fromPolygons(node)



Install
=======

To build run


.. code-block:: bash

    python setup.py build
    python setup.py install


Other Notes
===========


To view the output I would recommend a program like ParaView


Help required
=============

Patches are welcome for the source code or for the documentation

