from setuptools import Extension, setup
from Cython.Distutils import build_ext
import io
from os.path import exists

__version__ = "0.5.2"


ext_modules = [
    Extension("cython_csg", ["cython_csg/cython_csg.pyx"]),
]

setup(
    name='cython-csg',
    version=__version__,
    author='Thomas Turner',
    author_email='tom@twt.me.uk',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    scripts=[],
    packages=[
        'cython_csg',
    ],
    url='https://github.com/tomturner/cython-csg',
    license='MIT',
    description='Constructive solid geometry on meshes using BSP trees in Cython CSG.',
    long_description=io.open('README.rst', encoding='utf-8').read() if exists("README.rst") else "",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'License :: OSI Approved :: MIT License',
        'Framework :: Django',
        'Framework :: Django :: 2.0',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    install_requires=[
        'Cython >= 0.27',
    ],
    zip_safe=False,
)
